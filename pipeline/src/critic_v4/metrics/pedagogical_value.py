"""
Pedagogical Value Metric (Phase 1 - Question Evaluation)

Évalue la qualité pédagogique de la question pour détecter les questions
circulaires, triviales, ou sans valeur éducative.

Critères:
- tests_understanding: Teste la compréhension conceptuelle?
- non_trivial: Question non-triviale?
- educational_utility: Valeur éducative?

Score: count(True) / 3
Seuil: ≥0.67 pour PASS (au moins 2/3 critères)
"""

import json
import logging
from typing import Dict, Any, Optional

from ..prompts.pedagogical_value_prompt import get_pedagogical_value_prompt

logger = logging.getLogger(__name__)


class PedagogicalValue:
    """
    Évalue la qualité pédagogique de la question.
    
    Cette métrique fait partie de la Phase 1 du Critic V4, qui filtre les questions
    de faible qualité AVANT de générer les réponses.
    
    Critères binaires (True/False):
    - tests_understanding: La question teste-t-elle la compréhension conceptuelle?
    - non_trivial: La question est-elle non-triviale?
    - educational_utility: La question a-t-elle une valeur éducative?
    
    Score = nombre de True / 3
    Seuil de rejet: score < 0.67 (moins de 2 critères sur 3)
    """
    
    SCORE_THRESHOLD = 0.67  # ~2/3 critères doivent être True
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ):
        """
        Initialise l'évaluateur de Pedagogical Value.
        
        Args:
            llm: Instance Llama (si None, doit être fourni lors de evaluate())
            temperature: Température pour la génération LLM (bas = plus déterministe)
            max_tokens: Nombre maximum de tokens pour la réponse LLM
        """
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    def evaluate(
        self,
        chunk_content: str,
        question: str,
        llm: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Évalue la qualité pédagogique de la question.
        
        Args:
            chunk_content: Contenu du chunk de cours
            question: Question à évaluer
            llm: Instance Llama (optionnel si fourni dans __init__)
            
        Returns:
            Dict avec:
            - decision: "pass" ou "reject"
            - score: score normalisé (0.0 à 1.0)
            - criteria: dict des 3 critères (tests_understanding, non_trivial, educational_utility)
            - justification: explication de l'évaluation
            - suggestions: suggestions d'amélioration
            - feedback: feedback pour le Question Generator
            
        Raises:
            ValueError: Si ni self.llm ni llm fourni
            RuntimeError: Si erreur lors de l'appel LLM ou parsing JSON
        """
        # Vérifier qu'on a un LLM
        active_llm = llm or self.llm
        if active_llm is None:
            raise ValueError("Un LLM doit être fourni soit dans __init__ soit dans evaluate()")
        
        # Générer les prompts
        prompts = get_pedagogical_value_prompt(chunk_content, question)
        
        logger.info(f"Évaluation Pedagogical Value pour question: {question[:100]}...")
        
        try:
            # Appel LLM
            response = active_llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": prompts["system"]},
                    {"role": "user", "content": prompts["user"]},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            # Extraire le contenu de la réponse
            llm_output = response["choices"][0]["message"]["content"]
            logger.debug(f"Réponse LLM brute: {llm_output}")
            
            # Parser le JSON
            evaluation = self._parse_llm_response(llm_output)
            
            # Calculer le score (nombre de True / 3)
            criteria = {
                "tests_understanding": evaluation["tests_understanding"],
                "non_trivial": evaluation["non_trivial"],
                "educational_utility": evaluation["educational_utility"],
            }
            
            num_true = sum(1 for v in criteria.values() if v)
            score = num_true / 3.0
            
            # Décision pass/reject
            decision = "pass" if score >= self.SCORE_THRESHOLD else "reject"
            
            # Construire le résultat
            result = {
                "decision": decision,
                "score": score,
                "criteria": criteria,
                "justification": evaluation.get("justification", ""),
                "suggestions": evaluation.get("suggestions", ""),
                "feedback": self._generate_feedback(decision, score, criteria, evaluation),
            }
            
            logger.info(
                f"Pedagogical Value: {decision.upper()} "
                f"(score={score:.2f}, critères={num_true}/3, seuil={self.SCORE_THRESHOLD})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation Pedagogical Value: {e}")
            raise RuntimeError(f"Échec de l'évaluation: {e}") from e
    
    def _parse_llm_response(self, llm_output: str) -> Dict[str, Any]:
        """
        Parse la réponse JSON du LLM.
        
        Args:
            llm_output: Réponse brute du LLM
            
        Returns:
            Dict parsé contenant tests_understanding, non_trivial, educational_utility,
            justification, suggestions
            
        Raises:
            RuntimeError: Si le parsing JSON échoue
        """
        try:
            # Nettoyer la sortie (retirer markdown potential ```json ... ```)
            cleaned = llm_output.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            # Parser le JSON
            evaluation = json.loads(cleaned)
            
            # Normaliser les clés connues susceptibles de fautes de frappe
            # "educational_utily", "educational_utiliy", etc. → "educational_utility"
            normalized: Dict[str, Any] = {}
            for k, v in evaluation.items():
                if k.startswith("educational_util"):
                    normalized["educational_utility"] = v
                else:
                    normalized[k] = v
            evaluation = normalized

            # Valider les champs requis
            required_fields = ["tests_understanding", "non_trivial", "educational_utility"]
            for field in required_fields:
                if field not in evaluation:
                    raise ValueError(f"Champ '{field}' manquant dans la réponse LLM")
                if not isinstance(evaluation[field], bool):
                    raise ValueError(f"Champ '{field}' doit être un booléen")

            return evaluation
            
        except json.JSONDecodeError as e:
            logger.error(f"Échec du parsing JSON: {e}\nSortie LLM: {llm_output}")
            raise RuntimeError(f"Réponse LLM invalide (JSON attendu): {e}") from e
        except (ValueError, KeyError) as e:
            logger.error(f"Réponse LLM mal formée: {e}\nSortie: {llm_output}")
            raise RuntimeError(f"Structure JSON invalide: {e}") from e
    
    def _generate_feedback(
        self,
        decision: str,
        score: float,
        criteria: Dict[str, bool],
        evaluation: Dict[str, Any],
    ) -> str:
        """
        Génère un feedback pour le Question Generator.
        
        Args:
            decision: "pass" ou "reject"
            score: Score normalisé
            criteria: Dict des 3 critères booléens
            evaluation: Dict d'évaluation parsé
            
        Returns:
            Feedback textuel
        """
        if decision == "reject":
            false_criteria = [k for k, v in criteria.items() if not v]
            feedback = f"Question rejetée: qualité pédagogique insuffisante (score={score:.2f}/1.0). "
            feedback += f"Critères échoués: {', '.join(false_criteria)}. "
            
            suggestions = evaluation.get("suggestions", "")
            if suggestions:
                feedback += f"Suggestions: {suggestions}"
            else:
                feedback += "Génère une question qui teste la compréhension conceptuelle et a une vraie valeur éducative."
            
            return feedback
        else:
            num_true = sum(1 for v in criteria.values() if v)
            return f"Question acceptée: qualité pédagogique suffisante ({num_true}/3 critères, score={score:.2f})."
