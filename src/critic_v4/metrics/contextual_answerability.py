"""
Contextual Answerability Metric (Phase 1 - Question Evaluation)

Évalue si le chunk de cours contient suffisamment d'informations pour permettre
de répondre à la question posée.

Score: 0 (aucune info) → 3 (info complète)
Seuil: ≥2.0 pour PASS
"""

import json
import logging
from typing import Dict, Any, Optional

from ..prompts.contextual_answerability_prompt import get_contextual_answerability_prompt

logger = logging.getLogger(__name__)


class ContextualAnswerability:
    """
    Évalue si le chunk contient les informations nécessaires pour répondre à la question.
    
    Cette métrique fait partie de la Phase 1 du Critic V4, qui filtre les questions
    AVANT de générer les réponses, économisant du temps de calcul.
    
    Échelle de notation:
    - 0: Aucune information pertinente
    - 1: Informations partielles et insuffisantes
    - 2: Informations suffisantes mais incomplètes
    - 3: Informations complètes et précises
    
    Seuil de rejet: score < 2.0
    """
    
    SCORE_THRESHOLD = 2.0  # Score minimum pour accepter une question
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ):
        """
        Initialise l'évaluateur de Contextual Answerability.
        
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
        Évalue si le chunk contient assez d'informations pour répondre à la question.
        
        Args:
            chunk_content: Contenu du chunk de cours
            question: Question à évaluer
            llm: Instance Llama (optionnel si fourni dans __init__)
            
        Returns:
            Dict avec:
            - decision: "pass" ou "reject"
            - score: score brut (0.0 à 3.0)
            - normalized_score: score normalisé (0.0 à 1.0)
            - passages_pertinents: liste des passages extraits
            - justification: explication du score
            - manquements: liste des éléments manquants
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
        prompts = get_contextual_answerability_prompt(chunk_content, question)
        
        logger.info(f"Évaluation Contextual Answerability pour question: {question[:100]}...")
        
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
            
            # Extraire le score
            raw_score = float(evaluation["score"])
            
            # Décision pass/reject
            decision = "pass" if raw_score >= self.SCORE_THRESHOLD else "reject"
            
            # Construire le résultat
            result = {
                "decision": decision,
                "score": raw_score,
                "normalized_score": raw_score / 3.0,  # Normaliser sur [0, 1]
                "passages_pertinents": evaluation.get("passages_pertinents", []),
                "justification": evaluation.get("justification", ""),
                "manquements": evaluation.get("manquements", []),
                "feedback": self._generate_feedback(decision, raw_score, evaluation),
            }
            
            logger.info(
                f"Contextual Answerability: {decision.upper()} "
                f"(score={raw_score:.1f}/{3}, seuil={self.SCORE_THRESHOLD})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation Contextual Answerability: {e}")
            raise RuntimeError(f"Échec de l'évaluation: {e}") from e
    
    def _parse_llm_response(self, llm_output: str) -> Dict[str, Any]:
        """
        Parse la réponse JSON du LLM.
        
        Args:
            llm_output: Réponse brute du LLM
            
        Returns:
            Dict parsé contenant score, passages_pertinents, justification, manquements
            
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
            
            # Valider les champs requis
            if "score" not in evaluation:
                raise ValueError("Champ 'score' manquant dans la réponse LLM")
            
            score = evaluation["score"]
            if not isinstance(score, (int, float)) or score < 0 or score > 3:
                raise ValueError(f"Score invalide: {score} (doit être 0-3)")
            
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
        evaluation: Dict[str, Any],
    ) -> str:
        """
        Génère un feedback pour le Question Generator.
        
        Args:
            decision: "pass" ou "reject"
            score: Score brut
            evaluation: Dict d'évaluation parsé
            
        Returns:
            Feedback textuel
        """
        if decision == "reject":
            manquements = evaluation.get("manquements", [])
            feedback = f"Question rejetée: le chunk ne contient pas assez d'informations (score={score}/3). "
            if manquements:
                feedback += f"Manquements: {', '.join(manquements)}. "
            feedback += "Génère une question plus ancrée dans le contenu disponible."
            return feedback
        else:
            return f"Question acceptée: le chunk contient les informations nécessaires (score={score}/3)."
