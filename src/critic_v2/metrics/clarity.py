"""
Clarity Metric - Pattern G-Eval (Rubrique Explicite, échelle 1-3)
=================================================================

PRIORITÉ: MOYENNE

Adapté du pattern G-Eval:
  - Rubrique explicite avec critères clairs
  - Demander un score ET une justification step-by-step
  - Échelle courte (1-3) pour réduire la variance

L'échelle 1-3 est préférée à 1-5 car:
- Moins de variance dans les réponses
- Plus facile à calibrer
- Les LLM sont plus fiables sur des échelles courtes

Few-shot: exemples GÉNÉRIQUES (pas de domaine spécifique)
"""

import logging
from typing import Dict, Any

from ..base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT = """Tu es un expert en évaluation de la qualité linguistique et pédagogique.
Ta tâche est d'évaluer la CLARTÉ d'une paire Question-Réponse selon une rubrique précise.

Tu dois raisonner step-by-step avant de donner ton score."""


USER_PROMPT = """Évalue la CLARTÉ de la paire Question-Réponse suivante selon la rubrique ci-dessous.

=== PAIRE QA ===
Question: {question}
Réponse: {answer}

=== RUBRIQUE DE CLARTÉ ===

Score 3 - CLAIRE ET PRÉCISE:
  ✓ La question est compréhensible sans contexte externe
  ✓ La question utilise un vocabulaire précis et académique
  ✓ La réponse est bien structurée (introduction, développement si nécessaire)
  ✓ Pas de termes vagues ("truc", "chose", "ça")
  ✓ Pas de formulation orale ou familière
  ✓ La réponse répond directement à la question posée

Score 2 - PARTIELLEMENT CLAIRE:
  ~ La question est compréhensible mais pourrait être plus précise
  ~ La réponse contient quelques formulations maladroites
  ~ Structure acceptable mais pas optimale
  ~ Vocabulaire correct mais pas toujours précis

Score 1 - PAS CLAIRE:
  ✗ La question est vague, ambiguë, ou incompréhensible sans contexte
  ✗ La question utilise un langage oral/familier
  ✗ La réponse est confuse ou mal structurée
  ✗ Le lien question-réponse n'est pas direct

=== EXEMPLES ===

Exemple 1 (Score 3):
Q: "Quelle est la définition formelle d'une variable aléatoire continue?"
R: "Une variable aléatoire continue est une variable aléatoire dont l'ensemble des valeurs possibles est un intervalle de ℝ (ou une union d'intervalles). Sa loi de probabilité est caractérisée par une fonction de densité f telle que P(a ≤ X ≤ b) = ∫[a,b] f(x)dx."
→ Score: 3 (vocabulaire précis, structure claire, réponse complète)

Exemple 2 (Score 2):
Q: "Comment on calcule une probabilité avec la loi normale?"
R: "Pour calculer une probabilité avec la loi normale, on utilise la table de la loi normale centrée réduite. Il faut d'abord centrer et réduire la variable."
→ Score: 2 ("comment on" est légèrement oral, la réponse est correcte mais manque de détails sur le centrage-réduction)

Exemple 3 (Score 1):
Q: "C'est quoi le truc avec les probabilités conditionnelles?"
R: "En gros c'est quand on calcule la proba d'un événement sachant qu'un autre s'est passé, ça change la probabilité."  
→ Score: 1 ("c'est quoi le truc" = oral/vague, "en gros" = familier, "ça" = vague)

=== FORMAT DE SORTIE (JSON) ===
{{
  "reasoning": "Analyse step-by-step de la clarté de la question et de la réponse",
  "question_clarity": {{
    "is_precise": true,
    "is_academic": true,
    "issues": []
  }},
  "answer_clarity": {{
    "is_structured": true,
    "is_direct": true,
    "issues": []
  }},
  "score": 3
}}

Pense step-by-step. Génère UNIQUEMENT le JSON."""


class ClarityMetric(BaseMetric):
    """
    Clarity - G-Eval Rubric Pattern
    
    Évalue la clarté et la qualité linguistique de la paire QA
    avec une rubrique explicite et une échelle 1-3.
    
    Score normalisé: (score - 1) / 2 → [0, 1]
      1 → 0.0
      2 → 0.5
      3 → 1.0
    """
    
    name = "clarity"
    description = "Évalue la clarté et qualité linguistique (rubrique G-Eval)"
    priority = "MEDIUM"
    
    def evaluate(
        self,
        question: str,
        answer: str,
        chunk_content: str,
        **kwargs
    ) -> MetricResult:
        """
        Évaluer la clarté avec rubrique explicite.
        """
        
        try:
            response = self._call_llm(
                prompt=USER_PROMPT.format(
                    question=question,
                    answer=answer
                ),
                system_prompt=SYSTEM_PROMPT,
                temperature=0.2  # Un peu plus de température pour évaluation subjective
            )
            
            data = self._parse_json_response(response)
            raw_score = self._clamp_score(data.get("score", 2))
            reasoning = data.get("reasoning", "")
            
            # Extraire les détails
            q_clarity = data.get("question_clarity", {})
            a_clarity = data.get("answer_clarity", {})
            
        except Exception as e:
            logger.error(f"[{self.name}] Evaluation error: {e}")
            raw_score = 2
            reasoning = f"Evaluation error: {e}"
            q_clarity = {}
            a_clarity = {}
            response = ""
        
        # --- Score normalisé [0, 1] ---
        # 1 → 0.0, 2 → 0.5, 3 → 1.0
        score = (raw_score - 1) / 2.0
        
        # --- Construire les issues ---
        issues = []
        if q_clarity.get("issues"):
            issues.extend([f"Question: {i}" for i in q_clarity["issues"]])
        if a_clarity.get("issues"):
            issues.extend([f"Réponse: {i}" for i in a_clarity["issues"]])
        
        return MetricResult(
            metric_name=self.name,
            score=round(score, 3),
            reasoning=reasoning,
            details={
                "raw_score": raw_score,
                "scale": "1-3",
                "question_clarity": q_clarity,
                "answer_clarity": a_clarity,
                "issues": issues
            },
            raw_llm_output=response[:500] if response else ""
        )
    
    @staticmethod
    def _clamp_score(score) -> int:
        """Forcer le score dans {1, 2, 3}"""
        try:
            s = int(score)
        except (TypeError, ValueError):
            return 2
        return max(1, min(3, s))
