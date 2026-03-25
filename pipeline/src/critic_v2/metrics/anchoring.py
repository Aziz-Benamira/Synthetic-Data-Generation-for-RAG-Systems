"""
Anchoring Metric - Pattern Ragas Faithfulness (NLI à 2 étapes)
==============================================================

PRIORITÉ: CRITIQUE

Adapté du pattern Ragas Faithfulness:
  Step 1: Décomposer la réponse en statements atomiques
  Step 2: Pour chaque statement, verdict binaire (supported/not supported) 
  Score = moyenne des verdicts

Ce pattern est supérieur à un scoring global car:
- Chaque statement est évalué individuellement (pas de dilution)
- Le LLM ne peut pas "compenser" un statement faux par un vrai
- On identifie EXACTEMENT quels statements posent problème

Few-shot examples: GÉNÉRIQUES (Einstein, Super Bowl, pas de domaine spécifique)
"""

import logging
from typing import Dict, Any, List

from ..base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT_STEP1 = """Tu es un expert en analyse linguistique. 
Ta tâche est de décomposer une réponse en statements atomiques (affirmations simples et indépendantes).

Règles:
- Chaque statement doit être une phrase simple avec UN SEUL fait
- Ne pas ajouter d'interprétation
- Garder les statements fidèles au texte original
- Si la réponse contient des formules, les inclure comme statements"""


USER_PROMPT_STEP1 = """Décompose la réponse suivante en statements atomiques (affirmations simples et indépendantes).

=== RÉPONSE À DÉCOMPOSER ===
{answer}

=== EXEMPLES DE DÉCOMPOSITION ===

Exemple 1:
Réponse: "Albert Einstein est né en 1879 à Ulm, en Allemagne. Il a développé la théorie de la relativité restreinte en 1905."
Statements:
1. Albert Einstein est né en 1879.
2. Albert Einstein est né à Ulm.
3. Ulm se trouve en Allemagne.
4. Einstein a développé la théorie de la relativité restreinte.
5. La théorie de la relativité restreinte a été développée en 1905.

Exemple 2:
Réponse: "L'eau bout à 100°C sous pression atmosphérique normale, ce qui est essentiel pour la cuisson."
Statements:
1. L'eau bout à 100°C.
2. Cette température est valable sous pression atmosphérique normale.
3. L'ébullition de l'eau est essentielle pour la cuisson.

=== FORMAT DE SORTIE (JSON) ===
{{
  "statements": [
    "Statement atomique 1",
    "Statement atomique 2",
    "Statement atomique 3"
  ]
}}

Génère UNIQUEMENT le JSON."""


SYSTEM_PROMPT_STEP2 = """Tu es un expert en Natural Language Inference (NLI).
Ta tâche est de vérifier si chaque statement est SUPPORTÉ par le contexte source.

Un statement est SUPPORTÉ si et seulement si:
- L'information est EXPLICITEMENT présente dans le contexte
- Pas d'inférence, pas de déduction, pas d'extrapolation
- Les faits, chiffres, termes correspondent EXACTEMENT

Un statement est NON SUPPORTÉ si:
- L'information est absente du contexte
- C'est une inférence ou déduction non explicite
- Les détails ne correspondent pas (chiffres, noms, conditions)
- C'est une connaissance externe ajoutée"""


USER_PROMPT_STEP2 = """Pour chaque statement, détermine s'il est SUPPORTÉ (1) ou NON SUPPORTÉ (0) par le contexte.

=== CONTEXTE SOURCE ===
{chunk_content}

=== STATEMENTS À VÉRIFIER ===
{statements_formatted}

=== EXEMPLES DE VÉRIFICATION ===

Exemple (contexte: "La Terre tourne autour du Soleil en 365.25 jours"):
- "La Terre tourne autour du Soleil" → 1 (explicitement dans le contexte)
- "La Terre met 365.25 jours pour une orbite complète" → 1 (reformulation fidèle)
- "La Terre tourne plus vite que Mars" → 0 (information absente du contexte)
- "Cette rotation cause les saisons" → 0 (inférence non explicite)

=== FORMAT DE SORTIE (JSON) ===
{{
  "verdicts": [
    {{"statement": "Statement 1", "verdict": 1, "reason": "Raison courte"}},
    {{"statement": "Statement 2", "verdict": 0, "reason": "Raison courte"}}
  ]
}}

Pense step-by-step pour chaque statement. Génère UNIQUEMENT le JSON."""


class AnchoringMetric(BaseMetric):
    """
    Anchoring/Groundedness - Ragas Faithfulness Pattern
    
    Vérifie que CHAQUE affirmation de la réponse est ancrée dans le chunk source.
    Utilise le pattern NLI à 2 étapes pour une évaluation granulaire.
    
    Score = nombre de statements supportés / nombre total de statements
    """
    
    name = "anchoring"
    description = "Vérifie que chaque affirmation est ancrée dans le chunk source"
    priority = "CRITICAL"
    
    def evaluate(
        self,
        question: str,
        answer: str,
        chunk_content: str,
        **kwargs
    ) -> MetricResult:
        """
        Évaluer l'ancrage en 2 étapes:
        1. Décomposer la réponse en statements atomiques
        2. Vérifier chaque statement contre le contexte
        """
        total_tokens = 0
        
        # --- Step 1: Décomposer en statements ---
        try:
            step1_response = self._call_llm(
                prompt=USER_PROMPT_STEP1.format(answer=answer),
                system_prompt=SYSTEM_PROMPT_STEP1,
                temperature=0.1
            )
            
            step1_data = self._parse_json_response(step1_response)
            statements = step1_data.get("statements", [])
            
            if not statements:
                # Fallback: traiter la réponse entière comme un seul statement
                statements = [answer]
                logger.warning(f"[{self.name}] Step 1 failed to extract statements, using full answer")
                
        except Exception as e:
            logger.error(f"[{self.name}] Step 1 error: {e}")
            statements = [answer]
        
        # --- Step 2: Vérifier chaque statement ---
        try:
            statements_formatted = "\n".join(
                f"{i+1}. {s}" for i, s in enumerate(statements)
            )
            
            step2_response = self._call_llm(
                prompt=USER_PROMPT_STEP2.format(
                    chunk_content=chunk_content,
                    statements_formatted=statements_formatted
                ),
                system_prompt=SYSTEM_PROMPT_STEP2,
                temperature=0.1
            )
            
            step2_data = self._parse_json_response(step2_response)
            verdicts = step2_data.get("verdicts", [])
            
        except Exception as e:
            logger.error(f"[{self.name}] Step 2 error: {e}")
            verdicts = []
        
        # --- Calcul du score ---
        if verdicts:
            supported = sum(1 for v in verdicts if v.get("verdict", 0) == 1)
            total = len(verdicts)
            score = supported / total if total > 0 else 0.0
            
            # Construire le raisonnement
            unsupported = [
                v for v in verdicts if v.get("verdict", 0) == 0
            ]
            
            if unsupported:
                reasoning_parts = [f"{supported}/{total} statements supportés par le contexte."]
                reasoning_parts.append("Statements non supportés:")
                for v in unsupported:
                    reasoning_parts.append(
                        f"  - \"{v.get('statement', '?')}\" → {v.get('reason', 'non supporté')}"
                    )
                reasoning = "\n".join(reasoning_parts)
            else:
                reasoning = f"Tous les {total} statements sont supportés par le contexte."
        else:
            # Fallback si le parsing échoue
            score = 0.5
            reasoning = "Évaluation partielle: impossible de vérifier les statements individuels."
            supported = 0
            total = len(statements)
            unsupported = []
        
        return MetricResult(
            metric_name=self.name,
            score=round(score, 3),
            reasoning=reasoning,
            details={
                "total_statements": total if verdicts else len(statements),
                "supported_statements": supported if verdicts else 0,
                "unsupported_statements": [
                    {"statement": v.get("statement"), "reason": v.get("reason")}
                    for v in unsupported
                ] if unsupported else [],
                "all_statements": statements
            },
            raw_llm_output=f"Step1: {step1_response[:200] if 'step1_response' in dir() else ''}...\nStep2: {step2_response[:200] if 'step2_response' in dir() else ''}...",
            tokens_used=total_tokens
        )
