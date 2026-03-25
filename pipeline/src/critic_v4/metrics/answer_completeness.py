"""
Answer Completeness Metric (Phase 2 - Answer Validation)

Vérifie que la réponse générée couvre bien tous les aspects
importants requis par la question.

Score: 0 (hors-sujet) → 3 (complète)
Seuil: ≥2.0 pour PASS
"""

import json
import logging
from typing import Dict, Any, Optional

from ..prompts.answer_completeness_prompt import get_answer_completeness_prompt

logger = logging.getLogger(__name__)


class AnswerCompleteness:
    """
    Évalue si la réponse couvre tous les aspects requis par la question.

    Phase 2 du Critic V4: valide la réponse APRÈS sa génération.

    Échelle:
    - 0: Réponse vide ou hors-sujet
    - 1: Réponse partielle (aspects importants manquants)
    - 2: Réponse suffisante (aspects principaux couverts)
    - 3: Réponse complète (tous les aspects couverts en profondeur)

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
        question: str,
        answer: str,
        llm: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Évalue la complétude de la réponse.

        Args:
            question: La question posée
            answer: La réponse générée
            llm: Instance LLM optionnelle

        Returns:
            Dict avec:
            - decision: "pass" ou "reject"
            - score: score brut (0.0 à 3.0)
            - normalized_score: score normalisé (0.0 à 1.0)
            - aspects_requis: aspects attendus
            - aspects_couverts: aspects présents dans la réponse
            - aspects_manquants: aspects absents
            - justification: explication
            - feedback: pour le Answer Generator
        """
        active_llm = llm or self.llm
        if active_llm is None:
            raise ValueError("Un LLM doit être fourni soit dans __init__ soit dans evaluate()")

        prompts = get_answer_completeness_prompt(question, answer)
        logger.info(f"Évaluation Answer Completeness...")

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
                "aspects_requis": evaluation.get("aspects_requis", []),
                "aspects_couverts": evaluation.get("aspects_couverts", []),
                "aspects_manquants": evaluation.get("aspects_manquants", []),
                "justification": evaluation.get("justification", ""),
                "feedback": self._generate_feedback(decision, raw_score, evaluation),
            }

            logger.info(f"Answer Completeness: {decision.upper()} (score={raw_score:.1f}/3)")
            return result

        except Exception as e:
            logger.error(f"Erreur Answer Completeness: {e}")
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
            return f"Réponse acceptée: complète (score={score:.1f}/3)."
        manquants = evaluation.get("aspects_manquants", [])
        feedback = f"Réponse rejetée: incomplète (score={score:.1f}/3). "
        if manquants:
            feedback += f"Aspects manquants: {', '.join(manquants)}. "
        feedback += "Complète la réponse en abordant tous les aspects de la question."
        return feedback
