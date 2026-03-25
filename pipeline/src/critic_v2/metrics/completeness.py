"""
Completeness Metric - Adapté de Ragas Context Recall
====================================================

PRIORITÉ: MOYENNE

Adapté du pattern Ragas Context Recall:
  - Identifier les aspects/sous-questions de la question
  - Vérifier que chaque aspect est adressé dans la réponse
  - Score = aspects adressés / aspects totaux

Ce pattern est plus fiable qu'un scoring global car:
- On identifie EXACTEMENT quels aspects manquent
- Le feedback est actionnable (on sait quoi ajouter)

Few-shot: exemples GÉNÉRIQUES
"""

import logging
from typing import Dict, Any, List

from ..base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT = """Tu es un expert en évaluation de la complétude des réponses.
Ta tâche est de vérifier que la réponse adresse TOUS les aspects de la question.

Tu dois d'abord identifier les aspects/sous-questions, puis vérifier chacun.
Raisonne step-by-step."""


USER_PROMPT = """Évalue la COMPLÉTUDE de la réponse par rapport à la question.

=== CONTEXTE SOURCE ===
{chunk_content}

=== QUESTION ===
{question}

=== RÉPONSE ===
{answer}

=== MÉTHODE D'ÉVALUATION ===

Étape 1: Identifie tous les aspects/sous-questions implicites dans la question.
Étape 2: Pour chaque aspect, vérifie s'il est adressé dans la réponse.
Étape 3: Calcule le score.

=== EXEMPLES ===

Exemple 1 (Complet):
Question: "Qu'est-ce que la loi normale et quels sont ses paramètres?"
Aspects: [1. Définition de la loi normale, 2. Ses paramètres]
Réponse: "La loi normale est une distribution de probabilité continue définie par sa fonction de densité en forme de cloche. Ses deux paramètres sont la moyenne μ (espérance) et l'écart-type σ."
→ Aspect 1: ✓ Définition donnée
→ Aspect 2: ✓ Paramètres μ et σ identifiés
→ Score: 2/2 = 1.0

Exemple 2 (Partiel):
Question: "Quelles sont les causes et les conséquences du réchauffement climatique?"
Aspects: [1. Causes du réchauffement, 2. Conséquences du réchauffement]
Réponse: "Le réchauffement climatique est causé par les émissions de gaz à effet de serre, principalement le CO2 et le méthane."
→ Aspect 1: ✓ Causes mentionnées
→ Aspect 2: ✗ Conséquences non traitées
→ Score: 1/2 = 0.5

Exemple 3 (Incomplet):
Question: "Décrivez les trois lois de Newton."
Aspects: [1. Première loi, 2. Deuxième loi, 3. Troisième loi]
Réponse: "La première loi de Newton dit qu'un objet reste en mouvement uniforme sauf si une force s'applique."
→ Aspect 1: ✓ Première loi décrite
→ Aspect 2: ✗ Manquant
→ Aspect 3: ✗ Manquant
→ Score: 1/3 = 0.33

=== FORMAT DE SORTIE (JSON) ===
{{
  "reasoning": "Analyse step-by-step",
  "aspects": [
    {{"aspect": "Description de l'aspect", "addressed": true, "evidence": "Extrait de la réponse"}},
    {{"aspect": "Description de l'aspect", "addressed": false, "evidence": ""}}
  ],
  "addressed_count": 2,
  "total_aspects": 3
}}

Pense step-by-step. Génère UNIQUEMENT le JSON."""


class CompletenessMetric(BaseMetric):
    """
    Completeness - Ragas Context Recall Pattern Adapté
    
    Vérifie que CHAQUE aspect de la question est adressé dans la réponse.
    
    Score = aspects adressés / aspects totaux → [0, 1]
    """
    
    name = "completeness"
    description = "Vérifie que tous les aspects de la question sont adressés"
    priority = "MEDIUM"
    
    def evaluate(
        self,
        question: str,
        answer: str,
        chunk_content: str,
        **kwargs
    ) -> MetricResult:
        """
        Évaluer la complétude en identifiant les aspects et vérifiant chacun.
        """
        
        try:
            response = self._call_llm(
                prompt=USER_PROMPT.format(
                    chunk_content=chunk_content,
                    question=question,
                    answer=answer
                ),
                system_prompt=SYSTEM_PROMPT,
                temperature=0.1
            )
            
            data = self._parse_json_response(response)
            aspects = data.get("aspects", [])
            reasoning = data.get("reasoning", "")
            
        except Exception as e:
            logger.error(f"[{self.name}] Evaluation error: {e}")
            aspects = []
            reasoning = f"Evaluation error: {e}"
            response = ""
        
        # --- Calcul du score ---
        if aspects:
            addressed = sum(1 for a in aspects if a.get("addressed", False))
            total = len(aspects)
            score = addressed / total if total > 0 else 0.5
            
            # Construire la liste des aspects manquants
            missing = [
                a.get("aspect", "?") 
                for a in aspects 
                if not a.get("addressed", False)
            ]
            
            if missing:
                reasoning += f"\n\nAspects manquants: {', '.join(missing)}"
        else:
            # Fallback
            score = 0.5
            addressed = 0
            total = 0
            missing = []
            reasoning += "\nImpossible d'identifier les aspects de la question."
        
        return MetricResult(
            metric_name=self.name,
            score=round(score, 3),
            reasoning=reasoning,
            details={
                "total_aspects": total if aspects else 0,
                "addressed_aspects": addressed if aspects else 0,
                "missing_aspects": missing,
                "aspects_detail": aspects
            },
            raw_llm_output=response[:500] if response else ""
        )
