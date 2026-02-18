"""
QuestionEvaluator - Orchestrateur Phase 1 du Critic V4

Orchestre les deux métriques de filtrage de questions:
1. ContextualAnswerability: Le chunk contient-il les informations?
2. PedagogicalValue: La question a-t-elle une bonne qualité pédagogique?

Décision finale:
- PASS: les deux métriques passent
- REJECT: au moins une métrique échoue

La décision est accompagnée de:
- un score global (moyenne pondérée)
- un feedback détaillé pour le Question Generator
- les concepts clés extraits du chunk (pour ScopedMemory)
"""

import logging
from typing import Dict, Any, Optional, List

from .metrics import ContextualAnswerability, PedagogicalValue

logger = logging.getLogger(__name__)


class QuestionEvaluator:
    """
    Orchestrateur Phase 1: filtre les questions AVANT la génération de réponses.

    Combine ContextualAnswerability + PedagogicalValue pour prendre une décision
    binaire PASS/REJECT sur chaque question générée.

    Logique de décision:
    - PASS seulement si les DEUX métriques passent
    - REJECT si l'une ou l'autre échoue (court-circuit possible pour économiser du temps)

    Poids pour le score global:
    - contextual_answerability: 60% (ancrage dans le chunk = plus critique)
    - pedagogical_value: 40% (qualité pédagogique)
    """

    WEIGHT_CONTEXTUAL = 0.6
    WEIGHT_PEDAGOGICAL = 0.4

    def __init__(
        self,
        llm: Any = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        short_circuit: bool = True,
    ):
        """
        Initialise le QuestionEvaluator.

        Args:
            llm: Instance LLM (llama_cpp.Llama ou compatible)
            temperature: Température pour les évaluations LLM
            max_tokens: Nombre maximum de tokens par évaluation
            short_circuit: Si True, skip PedagogicalValue si ContextualAnswerability REJECT
                           (économise ~15s par question rejetée sur le contexte)
        """
        self.llm = llm
        self.short_circuit = short_circuit

        self.contextual_evaluator = ContextualAnswerability(
            llm=llm,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.pedagogical_evaluator = PedagogicalValue(
            llm=llm,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def evaluate(
        self,
        chunk_content: str,
        question: str,
        llm: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Évalue une question en Phase 1 (avant génération de réponse).

        Args:
            chunk_content: Contenu du chunk de cours
            question: Question à évaluer
            llm: Instance LLM optionnelle (override self.llm)

        Returns:
            Dict avec:
            - decision: "pass" ou "reject"
            - global_score: score global pondéré (0.0 à 1.0)
            - contextual: résultat complet de ContextualAnswerability
            - pedagogical: résultat complet de PedagogicalValue (ou None si court-circuit)
            - feedback: feedback consolidé pour le Question Generator
            - key_concepts: concepts clés extraits du chunk (pour ScopedMemory)
            - short_circuited: True si PedagogicalValue skippé
        """
        active_llm = llm or self.llm
        if active_llm is None:
            raise ValueError("Un LLM doit être fourni soit dans __init__ soit dans evaluate()")

        logger.info(f"QuestionEvaluator Phase 1: '{question[:80]}...'")

        # --- Étape 1: Contextual Answerability ---
        logger.info("  → Évaluation Contextual Answerability...")
        contextual_result = self.contextual_evaluator.evaluate(
            chunk_content=chunk_content,
            question=question,
            llm=active_llm,
        )

        # Court-circuit: si le chunk ne contient pas l'info, inutile d'évaluer la pédagogie
        if self.short_circuit and contextual_result["decision"] == "reject":
            logger.info("  → Court-circuit: ContextualAnswerability REJECT, skip PedagogicalValue")
            global_score = contextual_result["normalized_score"] * self.WEIGHT_CONTEXTUAL
            return {
                "decision": "reject",
                "global_score": global_score,
                "contextual": contextual_result,
                "pedagogical": None,
                "feedback": contextual_result["feedback"],
                "key_concepts": [],
                "short_circuited": True,
            }

        # --- Étape 2: Pedagogical Value ---
        logger.info("  → Évaluation Pedagogical Value...")
        pedagogical_result = self.pedagogical_evaluator.evaluate(
            chunk_content=chunk_content,
            question=question,
            llm=active_llm,
        )

        # --- Décision finale ---
        contextual_pass = contextual_result["decision"] == "pass"
        pedagogical_pass = pedagogical_result["decision"] == "pass"
        final_decision = "pass" if (contextual_pass and pedagogical_pass) else "reject"

        # Score global pondéré
        global_score = (
            contextual_result["normalized_score"] * self.WEIGHT_CONTEXTUAL
            + pedagogical_result["score"] * self.WEIGHT_PEDAGOGICAL
        )

        # Feedback consolidé
        feedback = self._build_feedback(
            final_decision,
            contextual_result,
            pedagogical_result,
        )

        # Extraire les concepts clés des passages pertinents (pour ScopedMemory)
        key_concepts = self._extract_key_concepts(
            contextual_result.get("passages_pertinents", [])
        )

        result = {
            "decision": final_decision,
            "global_score": round(global_score, 3),
            "contextual": contextual_result,
            "pedagogical": pedagogical_result,
            "feedback": feedback,
            "key_concepts": key_concepts,
            "short_circuited": False,
        }

        logger.info(
            f"  → Décision finale: {final_decision.upper()} "
            f"(score global={global_score:.2f}, "
            f"contextual={contextual_result['score']:.1f}/3, "
            f"pedagogical={pedagogical_result['score']:.2f}/1)"
        )

        return result

    def _build_feedback(
        self,
        decision: str,
        contextual: Dict[str, Any],
        pedagogical: Dict[str, Any],
    ) -> str:
        """
        Construit un feedback consolidé pour le Question Generator.

        En cas de REJECT, le feedback explique précisément pourquoi
        et donne des instructions concrètes pour améliorer la question.
        """
        if decision == "pass":
            return (
                f"Question acceptée (score={contextual['score']:.0f}/3 ancrage, "
                f"{pedagogical['score']:.0f}/1.0 pédagogie). Bonne qualité pour le dataset Gold."
            )

        parts = ["Question rejetée en Phase 1:"]

        # Raison contextuelle
        if contextual["decision"] == "reject":
            manquements = contextual.get("manquements", [])
            parts.append(
                f"  • Ancrage insuffisant (score={contextual['score']:.1f}/3): "
                + (f"Manquements: {', '.join(manquements)}." if manquements else contextual["feedback"])
            )

        # Raison pédagogique
        if pedagogical["decision"] == "reject":
            false_criteria = [k for k, v in pedagogical["criteria"].items() if not v]
            suggestions = pedagogical.get("suggestions", "")
            parts.append(
                f"  • Qualité pédagogique insuffisante (score={pedagogical['score']:.2f}/1.0, "
                f"critères échoués: {', '.join(false_criteria)})."
            )
            if suggestions:
                parts.append(f"  • Suggestion: {suggestions}")

        parts.append("→ Génère une nouvelle question mieux ancrée dans le contenu du chunk.")
        return "\n".join(parts)

    def _extract_key_concepts(self, passages: List[str]) -> List[str]:
        """
        Extrait des concepts clés simples depuis les passages pertinents.

        Utilisé par ScopedMemory pour tracker les concepts déjà couverts.
        Implémentation simple V1: extrait les noms de 4+ caractères.

        Args:
            passages: Liste de passages extraits du chunk

        Returns:
            Liste de concepts clés (max 10)
        """
        if not passages:
            return []

        # Extraction simple: mots significatifs (≥4 chars, pas de mots vides)
        STOP_WORDS = {
            "dans", "avec", "pour", "mais", "donc", "elle", "ils", "entre",
            "cette", "plus", "sont", "leur", "leur", "tout", "peut", "aussi",
            "être", "avoir", "faire", "bien", "très", "même", "dont", "comme",
            "ainsi", "selon", "lors", "après", "avant", "pendant", "depuis",
            "partir", "vers", "jusqu", "sans", "sous", "lors", "quand"
        }

        concepts = set()
        for passage in passages:
            words = passage.replace(",", " ").replace(".", " ").replace(";", " ").split()
            for word in words:
                clean = word.strip("«»()[]{}'\"-").lower()
                if len(clean) >= 4 and clean not in STOP_WORDS and clean.isalpha():
                    concepts.add(clean)

        # Retourner max 10 concepts les plus longs (indicatif de termes techniques)
        sorted_concepts = sorted(concepts, key=len, reverse=True)
        return sorted_concepts[:10]
