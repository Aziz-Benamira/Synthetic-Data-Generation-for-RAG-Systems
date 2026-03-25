"""
Question Generator V3
=====================

Génère des questions pédagogiques avec :
1. Intégration ScopedMemory  → diversité garantie (évite redondances)
2. Boucle de validation interne (CriticV4 Phase 1) → qualité garantie
3. Feedback-driven regeneration → corrige les questions rejetées

Architecture:
    generate(chunk)
    ├── [1] Construire le prompt avec hints de diversité (ScopedMemory)
    ├── [2] Appeler le LLM → question candidate
    ├── [3] Valider avec QuestionEvaluator (Phase 1 CriticV4)
    │   ├── PASS → enregistrer dans ScopedMemory → retourner
    │   └── REJECT → régénérer avec feedback (max max_retries fois)
    └── [fallback] Retourner la moins mauvaise question si échec total

Input  : chunk dict {chunk_id, content, chapter, section, ...}
Output : dict {question, key_concepts, phase1_score, attempts, status}
"""

import json
import re
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


# ─── Helpers (réutilisés depuis V2) ─────────────────────────────────────────

from .question_generator_v2 import (
    strip_think_tags,
    extract_json,
    extract_question_text,
)


# ─── Prompts ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un expert en création de questions pédagogiques pour des étudiants niveau Master 2.

RÈGLES ABSOLUES:
1. Génère EXACTEMENT UNE question en français académique
2. La question DOIT être répondable UNIQUEMENT avec le contenu fourni
3. La question doit être auto-suffisante (compréhensible sans voir le chunk)
4. Pas de questions triviales (oui/non) — exige explication, analyse ou démonstration
5. Le sujet mathématique/technique doit être précis et ciblé

TYPES DE QUESTIONS POSSIBLES:
- Conceptuel  : "Expliquez pourquoi...", "Quelle est la signification de..."
- Application : "Comment applique-t-on... pour calculer..."
- Démonstration: "Démontrez que...", "Montrez que..."
- Comparaison  : "Comparez... et... en termes de..."
- Causal       : "Pourquoi... implique-t-il..."

FORMAT DE SORTIE: UN JSON strict
{"question": "La question complète et auto-suffisante ?"}

IMPORTANT: Réponds UNIQUEMENT avec le JSON, rien d'autre."""


_FIRST_ATTEMPT_TEMPLATE = """Génère UNE question académique de qualité à partir du contenu suivant.

=== MÉTADONNÉES ===
Chapitre: {chapter}
Section: {section}
Type: {semantic_type}
Pages: {pages}

=== CONTENU ===
{content}

{diversity_hint}=== SORTIE ===
{{"question": "..."}}"""


_REGENERATE_TEMPLATE = """La question précédente a été rejetée. Génère une MEILLEURE question en corrigeant les problèmes identifiés.

=== QUESTION REJETÉE (tentative {attempt}) ===
{previous_question}

=== FEEDBACK DU CRITIC ===
{feedback}

=== CONTENU SOURCE ===
Chapitre: {chapter}
Section: {section}

{content}

{diversity_hint}=== INSTRUCTIONS ===
- Corrige PRÉCISÉMENT les problèmes du feedback
- La question DOIT être répondable avec CE contenu
- Formulation académique claire et auto-suffisante

{{"question": "..."}}"""


# ─── QuestionGeneratorV3 ─────────────────────────────────────────────────────

class QuestionGeneratorV3:
    """
    Génère des questions avec validation automatique (CriticV4 Phase 1)
    et intégration de la diversité (ScopedMemory).

    Usage:
        from src.llm import LLMManager
        from src.utils.scoped_memory import ScopedMemory
        from src.critic_v4 import QuestionEvaluator

        llm_mgr = LLMManager.from_direct_llamacpp(model_path, n_gpu_layers=-1)
        llm = llm_mgr.provider.llm

        memory = ScopedMemory()
        evaluator = QuestionEvaluator(llm=llm)

        gen = QuestionGeneratorV3(
            llm=llm,
            scoped_memory=memory,
            question_evaluator=evaluator,
        )

        result = gen.generate(chunk)
        # result["question"]     → str
        # result["phase1_score"] → float (0-1)
        # result["status"]       → "accepted" | "fallback"
        # result["attempts"]     → int
    """

    def __init__(
        self,
        llm: Any,
        scoped_memory=None,           # ScopedMemory | None
        question_evaluator=None,      # QuestionEvaluator | None
        temperature: float = 0.7,
        max_tokens: int = 300,
        max_retries: int = 3,
    ):
        """
        Args:
            llm: Instance LLM directe (llm_manager.provider.llm)
            scoped_memory: ScopedMemory pour hints de diversité (optionnel)
            question_evaluator: QuestionEvaluator CriticV4 Phase 1 (optionnel)
            temperature: Température de génération (0.7 = créatif)
            max_tokens: Tokens max par génération
            max_retries: Nombre max de tentatives en cas de rejet
        """
        self.llm = llm
        self.scoped_memory = scoped_memory
        self.question_evaluator = question_evaluator
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Génère une question validée à partir d'un chunk.

        Args:
            chunk: dict avec au minimum la clé "content".
                   Clés utiles: chunk_id, content, chapter, section,
                                semantic_type, page_range

        Returns:
            dict avec:
            - question      : str — la question finale
            - key_concepts  : list[str] — concepts clés (pour ScopedMemory)
            - phase1_score  : float — score CriticV4 Phase 1 (0-1, ou None)
            - attempts      : int — nombre de tentatives effectuées
            - status        : "accepted" | "fallback"
            - feedback      : str — dernier feedback du critic (si applicable)
        """
        # Mettre à jour le scope de la mémoire avec ce chunk
        if self.scoped_memory is not None:
            self.scoped_memory.update_scope(chunk)

        best_question: Optional[str] = None
        best_score: float = 0.0
        best_concepts: List[str] = []
        last_feedback: str = ""
        previous_question: Optional[str] = None

        for attempt in range(1, self.max_retries + 1):
            logger.info(f"[QuestionGeneratorV3] Tentative {attempt}/{self.max_retries}")

            # ── Générer une question candidate ────────────────────
            if attempt == 1:
                user_prompt = self._build_first_prompt(chunk)
            else:
                user_prompt = self._build_regen_prompt(
                    chunk=chunk,
                    previous_question=previous_question,
                    feedback=last_feedback,
                    attempt=attempt - 1,
                )

            raw = self._call_llm(user_prompt)
            question = self._parse_question(raw)

            if not question:
                logger.warning(f"  Tentative {attempt}: impossible de parser la question")
                continue

            logger.info(f"  Question: {question[:100]}")

            # ── Valider avec CriticV4 Phase 1 ─────────────────────
            if self.question_evaluator is None:
                # Pas de validateur → retourner directement
                if self.scoped_memory is not None:
                    self.scoped_memory.register_question(question, [])
                return {
                    "question": question,
                    "key_concepts": [],
                    "phase1_score": None,
                    "attempts": attempt,
                    "status": "accepted",
                    "feedback": "",
                }

            eval_result = self.question_evaluator.evaluate(
                chunk_content=chunk["content"],
                question=question,
                llm=self.llm,
            )

            score = eval_result["global_score"]
            decision = eval_result["decision"]
            concepts = eval_result.get("key_concepts", [])
            last_feedback = eval_result.get("feedback", "")

            logger.info(
                f"  Phase1: {decision.upper()} "
                f"(score={score:.2f}, short_circuit={eval_result.get('short_circuited', False)})"
            )

            # Garder la meilleure question en cas de fallback total
            if score > best_score:
                best_score = score
                best_question = question
                best_concepts = concepts

            if decision == "pass":
                # ✓ Question acceptée → enregistrer dans ScopedMemory
                if self.scoped_memory is not None:
                    self.scoped_memory.register_question(question, concepts)
                    logger.info(f"  Enregistré dans ScopedMemory ({len(concepts)} concepts)")

                return {
                    "question": question,
                    "key_concepts": concepts,
                    "phase1_score": score,
                    "attempts": attempt,
                    "status": "accepted",
                    "feedback": "",
                }

            # Question rejetée → préparer la régénération
            previous_question = question
            logger.info(f"  Rejeté — feedback: {last_feedback[:100]}")

        # ── Fallback : retourner la meilleure question même si rejetée ────
        logger.warning(
            f"[QuestionGeneratorV3] {self.max_retries} tentatives épuisées. "
            f"Retour de la meilleure question (score={best_score:.2f})."
        )

        return {
            "question": best_question or "",
            "key_concepts": best_concepts,
            "phase1_score": best_score,
            "attempts": self.max_retries,
            "status": "fallback",
            "feedback": last_feedback,
        }

    # ── Private ──────────────────────────────────────────────────────────────

    def _build_first_prompt(self, chunk: Dict[str, Any]) -> str:
        """Construit le prompt de première génération."""
        diversity_hint = self._get_diversity_hint()
        return _FIRST_ATTEMPT_TEMPLATE.format(
            chapter=chunk.get("chapter", "N/A"),
            section=chunk.get("section", "N/A"),
            semantic_type=chunk.get("semantic_type", "N/A"),
            pages=self._format_pages(chunk.get("page_range")),
            content=chunk["content"][:3000],
            diversity_hint=diversity_hint,
        )

    def _build_regen_prompt(
        self,
        chunk: Dict[str, Any],
        previous_question: str,
        feedback: str,
        attempt: int,
    ) -> str:
        """Construit le prompt de régénération avec feedback."""
        diversity_hint = self._get_diversity_hint()
        return _REGENERATE_TEMPLATE.format(
            previous_question=previous_question,
            feedback=feedback,
            attempt=attempt,
            chapter=chunk.get("chapter", "N/A"),
            section=chunk.get("section", "N/A"),
            content=chunk["content"][:3000],
            diversity_hint=diversity_hint,
        )

    def _get_diversity_hint(self) -> str:
        """Récupère les hints de diversité depuis ScopedMemory (si disponible)."""
        if self.scoped_memory is None:
            return ""
        hint = self.scoped_memory.get_diversity_prompt()
        if not hint:
            return ""
        return f"=== CONTRAINTE DE DIVERSITÉ ===\n{hint}\n\n"

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

    def _parse_question(self, raw: str) -> Optional[str]:
        """Parse la question depuis la sortie LLM avec fallbacks."""
        cleaned = strip_think_tags(raw)

        # 1. JSON
        data = extract_json(cleaned)
        if data and "question" in data:
            q = data["question"].strip()
            if len(q) > 20:
                return q

        # 2. Phrase avec ?
        q = extract_question_text(cleaned)
        if q and len(q) > 20:
            return q

        logger.warning(f"Impossible de parser: {cleaned[:200]}")
        return None

    @staticmethod
    def _format_pages(page_range) -> str:
        if isinstance(page_range, (list, tuple)) and len(page_range) >= 2:
            return f"{page_range[0]}-{page_range[1]}"
        return str(page_range or "N/A")
