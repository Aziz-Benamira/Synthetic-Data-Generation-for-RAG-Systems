"""
Answer Anchoring Metric (Phase 2 - Answer Validation)

Vérifie que la réponse est bien ancrée dans le chunk source et ne contient
pas d'informations inventées (hallucinations).

Score: 0 (majoritairement halluciné) → 3 (parfaitement ancré)
Seuil: ≥2.0 pour PASS
"""

import json
import logging
from typing import Dict, Any, Optional

from ..prompts.answer_anchoring_prompt import get_answer_anchoring_prompt

logger = logging.getLogger(__name__)


class AnswerAnchoring:
    """
    Évalue si la réponse est ancrée dans le chunk (pas d'hallucinations).

    Phase 2 du Critic V4: valide la réponse APRÈS sa génération.

    Échelle:
    - 0: Majoritairement halluciné (>50% hors-chunk)
    - 1: Partiellement ancré (25-50% hors-chunk)
    - 2: Bien ancré avec extrapolations mineures (<25% hors-chunk)
    - 3: Parfaitement ancré (tout dans le chunk)

    Seuil: ≥2.0 pour PASS
    """

    SCORE_THRESHOLD = 2.0

    def __init__(
        self,
        llm: Optional[Any] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ):
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens

    def evaluate(
        self,
        chunk_content: str,
        question: str,
        answer: str,
        llm: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Évalue l'ancrage de la réponse dans le chunk.

        Args:
            chunk_content: Le chunk source (référence)
            question: La question posée
            answer: La réponse à évaluer
            llm: Instance LLM optionnelle

        Returns:
            Dict avec:
            - decision: "pass" ou "reject"
            - score: score brut (0.0 à 3.0)
            - normalized_score: score normalisé (0.0 à 1.0)
            - affirmations_ancrees: affirmations supportées par le chunk
            - affirmations_non_ancrees: affirmations inventées (hallucinations)
            - affirmations_extrapolations: déductions logiques acceptables
            - justification: explication
            - feedback: pour le Answer Generator
        """
        active_llm = llm or self.llm
        if active_llm is None:
            raise ValueError("Un LLM doit être fourni soit dans __init__ soit dans evaluate()")

        prompts = get_answer_anchoring_prompt(chunk_content, question, answer)
        logger.info(f"Évaluation Answer Anchoring...")

        try:
            response = active_llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": prompts["system"]},
                    {"role": "user", "content": prompts["user"]},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            llm_output = response["choices"][0]["message"]["content"]
            evaluation = self._parse_llm_response(llm_output)

            raw_score = float(evaluation["score"])
            decision = "pass" if raw_score >= self.SCORE_THRESHOLD else "reject"

            result = {
                "decision": decision,
                "score": raw_score,
                "normalized_score": raw_score / 3.0,
                "affirmations_ancrees": evaluation.get("affirmations_ancrees", []),
                "affirmations_non_ancrees": evaluation.get("affirmations_non_ancrees", []),
                "affirmations_extrapolations": evaluation.get("affirmations_extrapolations", []),
                "justification": evaluation.get("justification", ""),
                "feedback": self._generate_feedback(decision, raw_score, evaluation),
            }

            logger.info(f"Answer Anchoring: {decision.upper()} (score={raw_score:.1f}/3)")
            return result

        except Exception as e:
            logger.error(f"Erreur Answer Anchoring: {e}")
            raise RuntimeError(f"Échec de l'évaluation: {e}") from e

    def _parse_llm_response(self, llm_output: str) -> Dict[str, Any]:
        cleaned = llm_output.strip()
        for marker in ["```json", "```"]:
            if cleaned.startswith(marker):
                cleaned = cleaned[len(marker):]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            evaluation = json.loads(cleaned)
            if "score" not in evaluation:
                raise ValueError("Champ 'score' manquant")
            score = evaluation["score"]
            if not isinstance(score, (int, float)) or not (0 <= score <= 3):
                raise ValueError(f"Score invalide: {score}")
            return evaluation
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Réponse LLM invalide (JSON attendu): {e}") from e

    def _generate_feedback(self, decision: str, score: float, evaluation: Dict[str, Any]) -> str:
        if decision == "pass":
            return f"Réponse acceptée: bien ancrée dans le chunk (score={score:.1f}/3)."
        hallucinations = evaluation.get("affirmations_non_ancrees", [])
        feedback = f"Réponse rejetée: trop d'affirmations hors-chunk (score={score:.1f}/3). "
        if hallucinations:
            feedback += f"Affirmations non-ancrées: {', '.join(hallucinations[:3])}. "
        feedback += "Reformule la réponse en te basant uniquement sur le contenu du chunk."
        return feedback
