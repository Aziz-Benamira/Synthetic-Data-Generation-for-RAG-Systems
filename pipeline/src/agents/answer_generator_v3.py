"""
Answer Generator V3
===================

Génère des réponses ancrées avec validation automatique (CriticV4 Phase 2) :
- AnswerCompleteness : vérifie la couverture de tous les aspects
- AnswerAnchoring    : détecte les hallucinations hors-chunk

Architecture:
    generate(question, chunk)
    ├── [1] Appeler le LLM → réponse candidate
    ├── [2] Valider avec AnswerCompleteness + AnswerAnchoring
    │   ├── PASS → retourner la réponse validée
    │   └── REJECT → régénérer avec feedback (max max_retries fois)
    └── [fallback] Retourner la meilleure réponse si échec total

Input  : question (str) + chunk (dict)
Output : dict {answer, phase2_completeness_score, phase2_anchoring_score,
               attempts, status, feedback}
"""

import re
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


# ─── Helpers (réutilisés depuis V2) ─────────────────────────────────────────

from .question_generator_v2 import strip_think_tags, extract_json


# ─── Prompts ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un assistant pédagogique expert (niveau Master 2).

RÈGLES ABSOLUES:
1. Utilise STRICTEMENT ET UNIQUEMENT le contenu fourni pour répondre
2. JAMAIS de connaissances externes — même si tu les connais
3. Si l'information n'est pas dans le contenu, dis-le clairement
4. Cite des passages EXACTS du texte source entre guillemets
5. Réponse claire, structurée, en français académique
6. 3-6 phrases — complète mais concise

FORMAT DE SORTIE: UN JSON strict
{
  "answer": "La réponse complète",
  "supporting_quotes": ["citation exacte 1", "citation exacte 2"]
}

IMPORTANT: Réponds UNIQUEMENT avec le JSON, rien d'autre."""


_FIRST_ATTEMPT_TEMPLATE = """Réponds à la question en utilisant UNIQUEMENT le contenu ci-dessous.

=== QUESTION ===
{question}

=== CONTENU SOURCE (seule information autorisée) ===
Chapitre: {chapter}
Section: {section}
Pages: {pages}

{content}

=== SORTIE ===
{{"answer": "...", "supporting_quotes": ["..."]}}"""


_REGENERATE_TEMPLATE = """La réponse précédente a été évaluée par le Critic. Génère une MEILLEURE réponse.

=== QUESTION ===
{question}

=== RÉPONSE PRÉCÉDENTE (tentative {attempt}) ===
{previous_answer}

=== FEEDBACK DU CRITIC ===
{feedback}

=== CONTENU SOURCE (seule information autorisée) ===
Chapitre: {chapter}
Section: {section}

{content}

=== INSTRUCTIONS DE CORRECTION ===
- Corrige PRÉCISÉMENT les problèmes mentionnés dans le feedback
- Utilise UNIQUEMENT le contenu source ci-dessus
- N'INVENTE RIEN — chaque affirmation doit être dans le texte
- Cite des passages EXACTS entre guillemets
- Garde les éléments corrects de la réponse précédente

{{"answer": "...", "supporting_quotes": ["..."]}}"""


# ─── AnswerGeneratorV3 ───────────────────────────────────────────────────────

class AnswerGeneratorV3:
    """
    Génère des réponses avec validation automatique Phase 2 (CriticV4).

    Usage:
        from src.llm import LLMManager
        from src.critic_v4.metrics import AnswerCompleteness, AnswerAnchoring

        llm_mgr = LLMManager.from_direct_llamacpp(model_path, n_gpu_layers=-1)
        llm = llm_mgr.provider.llm

        gen = AnswerGeneratorV3(
            llm=llm,
            completeness_evaluator=AnswerCompleteness(llm=llm),
            anchoring_evaluator=AnswerAnchoring(llm=llm),
        )

        result = gen.generate(question="...", chunk={...})
        # result["answer"]                    → str
        # result["phase2_completeness_score"] → float (0-3)
        # result["phase2_anchoring_score"]    → float (0-3)
        # result["status"]                    → "accepted" | "fallback"
        # result["attempts"]                  → int
    """

    # Poids Phase 2 (miroir de CriticV4)
    WEIGHT_COMPLETENESS = 0.6
    WEIGHT_ANCHORING = 0.4

    def __init__(
        self,
        llm: Any,
        completeness_evaluator=None,  # AnswerCompleteness | None
        anchoring_evaluator=None,     # AnswerAnchoring | None
        temperature: float = 0.3,
        max_tokens: int = 600,
        max_retries: int = 2,
    ):
        """
        Args:
            llm: Instance LLM directe (llm_manager.provider.llm)
            completeness_evaluator: AnswerCompleteness de CriticV4 (optionnel)
            anchoring_evaluator: AnswerAnchoring de CriticV4 (optionnel)
            temperature: 0.3 (faible = réponses ancrées et cohérentes)
            max_tokens: Tokens max par génération
            max_retries: Tentatives max (Phase 2 est plus coûteuse → défaut 2)
        """
        self.llm = llm
        self.completeness_evaluator = completeness_evaluator
        self.anchoring_evaluator = anchoring_evaluator
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self, question: str, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Génère une réponse validée pour une question donnée.

        Args:
            question: La question à répondre
            chunk: Dict avec au minimum "content".
                   Clés utiles: content, chapter, section, page_range

        Returns:
            dict avec:
            - answer                    : str — la réponse finale
            - phase2_completeness_score : float — score complétude (0-3, ou None)
            - phase2_anchoring_score    : float — score ancrage (0-3, ou None)
            - phase2_score              : float — score global Phase 2 (0-1)
            - attempts                  : int — nombre de tentatives
            - status                    : "accepted" | "fallback"
            - feedback                  : str — dernier feedback (si applicable)
        """
        best_answer: Optional[str] = None
        best_phase2_score: float = 0.0
        best_completeness: Optional[float] = None
        best_anchoring: Optional[float] = None
        last_feedback: str = ""
        previous_answer: Optional[str] = None

        for attempt in range(1, self.max_retries + 1):
            logger.info(f"[AnswerGeneratorV3] Tentative {attempt}/{self.max_retries}")

            # ── Générer une réponse candidate ─────────────────────
            if attempt == 1:
                user_prompt = self._build_first_prompt(question, chunk)
            else:
                user_prompt = self._build_regen_prompt(
                    question=question,
                    chunk=chunk,
                    previous_answer=previous_answer,
                    feedback=last_feedback,
                    attempt=attempt - 1,
                )

            raw = self._call_llm(user_prompt)
            answer = self._parse_answer(raw)

            if not answer:
                logger.warning(f"  Tentative {attempt}: impossible de parser la réponse")
                continue

            logger.info(f"  Réponse: {answer[:100]}...")

            # ── Valider avec CriticV4 Phase 2 ─────────────────────
            if self.completeness_evaluator is None and self.anchoring_evaluator is None:
                # Pas de validateur → retourner directement
                return {
                    "answer": answer,
                    "phase2_completeness_score": None,
                    "phase2_anchoring_score": None,
                    "phase2_score": None,
                    "attempts": attempt,
                    "status": "accepted",
                    "feedback": "",
                }

            completeness_result = None
            anchoring_result = None
            completeness_score = 0.0
            anchoring_score = 0.0
            rejected = False
            feedbacks = []

            # AnswerCompleteness
            if self.completeness_evaluator is not None:
                completeness_result = self.completeness_evaluator.evaluate(
                    question=question,
                    answer=answer,
                    llm=self.llm,
                )
                completeness_score = completeness_result.get("score", 0.0)
                if completeness_result["decision"] == "reject":
                    rejected = True
                    feedbacks.append(completeness_result.get("feedback", ""))
                    logger.info(f"  Complétude: REJECT (score={completeness_score:.1f}/3)")
                else:
                    logger.info(f"  Complétude: PASS (score={completeness_score:.1f}/3)")

            # AnswerAnchoring (uniquement si complétude passe)
            if self.anchoring_evaluator is not None and not rejected:
                anchoring_result = self.anchoring_evaluator.evaluate(
                    chunk_content=chunk["content"],
                    question=question,
                    answer=answer,
                    llm=self.llm,
                )
                anchoring_score = anchoring_result.get("score", 0.0)
                if anchoring_result["decision"] == "reject":
                    rejected = True
                    feedbacks.append(anchoring_result.get("feedback", ""))
                    logger.info(f"  Ancrage: REJECT (score={anchoring_score:.1f}/3)")
                else:
                    logger.info(f"  Ancrage: PASS (score={anchoring_score:.1f}/3)")

            # Score Phase 2 normalisé (0-1)
            phase2_score = (
                (completeness_score / 3.0) * self.WEIGHT_COMPLETENESS
                + (anchoring_score / 3.0) * self.WEIGHT_ANCHORING
            )

            # Garder la meilleure réponse en cas de fallback
            if phase2_score > best_phase2_score:
                best_phase2_score = phase2_score
                best_answer = answer
                best_completeness = completeness_score
                best_anchoring = anchoring_score

            if not rejected:
                # ✓ Réponse validée
                return {
                    "answer": answer,
                    "phase2_completeness_score": completeness_score,
                    "phase2_anchoring_score": anchoring_score,
                    "phase2_score": round(phase2_score, 3),
                    "attempts": attempt,
                    "status": "accepted",
                    "feedback": "",
                }

            # Réponse rejetée → préparer la régénération
            last_feedback = " | ".join(feedbacks)
            previous_answer = answer
            logger.info(f"  Rejeté — feedback: {last_feedback[:100]}")

        # ── Fallback ─────────────────────────────────────────────────────────
        logger.warning(
            f"[AnswerGeneratorV3] {self.max_retries} tentatives épuisées. "
            f"Retour de la meilleure réponse (phase2_score={best_phase2_score:.2f})."
        )

        return {
            "answer": best_answer or "",
            "phase2_completeness_score": best_completeness,
            "phase2_anchoring_score": best_anchoring,
            "phase2_score": round(best_phase2_score, 3),
            "attempts": self.max_retries,
            "status": "fallback",
            "feedback": last_feedback,
        }

    # ── Private ──────────────────────────────────────────────────────────────

    def _build_first_prompt(self, question: str, chunk: Dict[str, Any]) -> str:
        return _FIRST_ATTEMPT_TEMPLATE.format(
            question=question,
            chapter=chunk.get("chapter", "N/A"),
            section=chunk.get("section", "N/A"),
            pages=self._format_pages(chunk.get("page_range")),
            content=chunk["content"][:3000],
        )

    def _build_regen_prompt(
        self,
        question: str,
        chunk: Dict[str, Any],
        previous_answer: str,
        feedback: str,
        attempt: int,
    ) -> str:
        return _REGENERATE_TEMPLATE.format(
            question=question,
            previous_answer=previous_answer,
            feedback=feedback,
            attempt=attempt,
            chapter=chunk.get("chapter", "N/A"),
            section=chunk.get("section", "N/A"),
            content=chunk["content"][:3000],
        )

    def _call_llm(self, user_prompt: str) -> str:
        """Appelle le LLM directement (accès llama_cpp)."""
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response["choices"][0]["message"]["content"]

    def _parse_answer(self, raw: str) -> Optional[str]:
        """Parse la réponse depuis la sortie LLM avec fallbacks."""
        cleaned = strip_think_tags(raw)

        # 1. JSON
        data = extract_json(cleaned)
        if data and "answer" in data:
            answer = data["answer"].strip()
            if len(answer) > 20:
                return answer

        # 2. Texte brut nettoyé
        text = re.sub(r'[{}":]', '', cleaned).strip()
        text = re.sub(r'^answer\s*', '', text, flags=re.IGNORECASE).strip()
        if len(text) > 30:
            logger.warning("Utilisation du texte brut comme réponse (JSON parse échoué)")
            return text

        logger.warning(f"Impossible de parser: {cleaned[:200]}")
        return None

    @staticmethod
    def _format_pages(page_range) -> str:
        if isinstance(page_range, (list, tuple)) and len(page_range) >= 2:
            return f"{page_range[0]}-{page_range[1]}"
        return str(page_range or "N/A")
