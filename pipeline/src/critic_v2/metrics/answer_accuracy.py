"""
Answer Accuracy Metric - Pattern Nvidia Double Judge (0-2-4)
============================================================

PRIORITÉ: HAUTE

Adapté du pattern Nvidia Answer Accuracy:
  Judge 1: Compare réponse vs contexte → rating
  Judge 2: Compare contexte vs réponse → rating (symétrique)
  Score = moyenne normalisée des deux ratings

L'évaluation symétrique est plus robuste car:
- Judge 1 détecte les hallucinations (réponse dit des choses pas dans le contexte)
- Judge 2 détecte les omissions (contexte contient des infos absentes de la réponse)
- La moyenne réduit le biais de position

Échelle: 0 (incorrect), 2 (partiellement correct), 4 (correct)
Score normalisé: rating / 4 → [0, 1]

Few-shot examples: GÉNÉRIQUES (pas de domaine spécifique)
"""

import logging
from typing import Dict, Any

from ..base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT = """Tu es un expert en vérification factuelle.
Ta tâche est de comparer deux textes et d'évaluer leur concordance.

Tu utilises une échelle stricte:
- 0 = INCORRECT : Les textes se contredisent ou sont sans rapport
- 2 = PARTIELLEMENT CORRECT : Accord partiel, informations manquantes ou imprécises
- 4 = CORRECT : Les textes concordent pleinement, pas de contradiction

Tu dois raisonner step-by-step avant de donner ton verdict."""


USER_PROMPT_JUDGE1 = """Compare la RÉPONSE avec le CONTEXTE SOURCE. 
La réponse est-elle fidèle et exacte par rapport au contexte ?

=== CONTEXTE SOURCE ===
{chunk_content}

=== QUESTION ===
{question}

=== RÉPONSE À ÉVALUER ===
{answer}

=== EXEMPLES DE NOTATION ===

Exemple 1 (Score 4 - CORRECT):
Contexte: "La vitesse de la lumière dans le vide est de 299 792 458 m/s."
Question: "Quelle est la vitesse de la lumière?"
Réponse: "La vitesse de la lumière dans le vide est d'environ 299 792 458 m/s."
→ Rating: 4 (information exacte, fidèle au contexte)

Exemple 2 (Score 2 - PARTIEL):
Contexte: "L'eau bout à 100°C sous pression atmosphérique normale (1 atm)."
Question: "À quelle température l'eau bout-elle?"
Réponse: "L'eau bout à 100°C."
→ Rating: 2 (correct mais omet la condition "pression atmosphérique normale")

Exemple 3 (Score 0 - INCORRECT):
Contexte: "Le fer fond à 1538°C."
Question: "Quel est le point de fusion du fer?"
Réponse: "Le fer fond à 1083°C."
→ Rating: 0 (valeur incorrecte, 1083°C est le cuivre)

=== INSTRUCTIONS ===
1. Lis attentivement le contexte
2. Compare chaque affirmation de la réponse avec le contexte
3. Note les concordances ET les divergences
4. Donne ton rating

=== FORMAT DE SORTIE (JSON) ===
{{
  "reasoning": "Analyse step-by-step des concordances et divergences",
  "concordances": ["Point de concordance 1", "Point de concordance 2"],
  "divergences": ["Divergence 1", "Divergence 2"],
  "rating": 0
}}

Pense step-by-step. Génère UNIQUEMENT le JSON."""


USER_PROMPT_JUDGE2 = """Compare le CONTEXTE SOURCE avec la RÉPONSE (sens inverse).
Le contexte contient-il les informations nécessaires pour la réponse donnée ?

=== RÉPONSE ===
{answer}

=== QUESTION ===
{question}

=== CONTEXTE SOURCE ===
{chunk_content}

=== INSTRUCTIONS ===
1. Lis la réponse d'abord
2. Vérifie que CHAQUE affirmation de la réponse trouve son support dans le contexte
3. Identifie les affirmations sans support dans le contexte
4. Donne ton rating

Échelle:
- 0 = La réponse contient des informations ABSENTES du contexte (hallucination)
- 2 = La réponse est partiellement supportée par le contexte
- 4 = La réponse est ENTIÈREMENT supportée par le contexte

=== FORMAT DE SORTIE (JSON) ===
{{
  "reasoning": "Analyse step-by-step de chaque affirmation",
  "supported_claims": ["Affirmation supportée 1"],
  "unsupported_claims": ["Affirmation non supportée 1"],
  "rating": 0
}}

Pense step-by-step. Génère UNIQUEMENT le JSON."""


class AnswerAccuracyMetric(BaseMetric):
    """
    Answer Accuracy - Nvidia Double Judge Pattern
    
    Évalue la fidélité de la réponse par rapport au contexte source
    avec une double évaluation symétrique pour robustesse.
    
    Score = (rating_judge1 + rating_judge2) / 8  → [0, 1]
    """
    
    name = "answer_accuracy"
    description = "Vérifie la fidélité factuelle de la réponse (double judge)"
    priority = "HIGH"
    
    def evaluate(
        self,
        question: str,
        answer: str,
        chunk_content: str,
        **kwargs
    ) -> MetricResult:
        """
        Double évaluation symétrique:
        Judge 1: contexte → réponse (détecte hallucinations)
        Judge 2: réponse → contexte (détecte omissions)
        """
        
        # --- Judge 1: Réponse fidèle au contexte ? ---
        try:
            judge1_response = self._call_llm(
                prompt=USER_PROMPT_JUDGE1.format(
                    chunk_content=chunk_content,
                    question=question,
                    answer=answer
                ),
                system_prompt=SYSTEM_PROMPT,
                temperature=0.1
            )
            judge1_data = self._parse_json_response(judge1_response)
            rating1 = self._clamp_rating(judge1_data.get("rating", 2))
            reasoning1 = judge1_data.get("reasoning", "")
            
        except Exception as e:
            logger.error(f"[{self.name}] Judge 1 error: {e}")
            rating1 = 2
            reasoning1 = f"Judge 1 error: {e}"
        
        # --- Judge 2: Contexte supporte la réponse ? ---
        try:
            judge2_response = self._call_llm(
                prompt=USER_PROMPT_JUDGE2.format(
                    chunk_content=chunk_content,
                    question=question,
                    answer=answer
                ),
                system_prompt=SYSTEM_PROMPT,
                temperature=0.1
            )
            judge2_data = self._parse_json_response(judge2_response)
            rating2 = self._clamp_rating(judge2_data.get("rating", 2))
            reasoning2 = judge2_data.get("reasoning", "")
            
        except Exception as e:
            logger.error(f"[{self.name}] Judge 2 error: {e}")
            rating2 = 2
            reasoning2 = f"Judge 2 error: {e}"
        
        # --- Score normalisé ---
        # Chaque rating est 0, 2, ou 4. Score max = 4+4 = 8
        raw_score = rating1 + rating2
        score = raw_score / 8.0
        
        # --- Raisonnement combiné ---
        reasoning = (
            f"Judge 1 (fidélité): {rating1}/4 - {reasoning1}\n"
            f"Judge 2 (support): {rating2}/4 - {reasoning2}\n"
            f"Score combiné: {raw_score}/8 = {score:.2f}"
        )
        
        return MetricResult(
            metric_name=self.name,
            score=round(score, 3),
            reasoning=reasoning,
            details={
                "judge1_rating": rating1,
                "judge2_rating": rating2,
                "raw_score": raw_score,
                "judge1_concordances": judge1_data.get("concordances", []) if 'judge1_data' in dir() else [],
                "judge1_divergences": judge1_data.get("divergences", []) if 'judge1_data' in dir() else [],
                "judge2_supported": judge2_data.get("supported_claims", []) if 'judge2_data' in dir() else [],
                "judge2_unsupported": judge2_data.get("unsupported_claims", []) if 'judge2_data' in dir() else [],
            },
            raw_llm_output=f"J1: {judge1_response[:200] if 'judge1_response' in dir() else ''}...\nJ2: {judge2_response[:200] if 'judge2_response' in dir() else ''}..."
        )
    
    @staticmethod
    def _clamp_rating(rating) -> int:
        """Forcer le rating dans {0, 2, 4}"""
        try:
            r = int(rating)
        except (TypeError, ValueError):
            return 2
        
        if r <= 1:
            return 0
        elif r <= 3:
            return 2
        else:
            return 4
