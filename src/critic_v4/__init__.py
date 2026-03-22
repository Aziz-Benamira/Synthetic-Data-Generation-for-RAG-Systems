"""
Critic V4 - Architecture 2-Phases pour Évaluation QA

Phase 1 (Question Filtering):
- Contextual Answerability: Vérifie si le chunk contient les informations nécessaires
- Pedagogical Value: Évalue la qualité pédagogique de la question
- QuestionEvaluator: Orchestrateur Phase 1 (combine les deux métriques ci-dessus)

Phase 2 (Answer Validation):
- Answer Completeness: Vérifie que la réponse couvre tous les aspects
- Answer Anchoring: Vérifie l'ancrage de la réponse dans le chunk

Phase 3 (Question Difficulty — optionnelle):
- DifficultyGrader: Évalue le niveau cognitif 1–5 selon la taxonomie de Bloom
"""

from .question_evaluator import QuestionEvaluator
from .critic_v4 import CriticV4
from .metrics import (
    ContextualAnswerability,
    PedagogicalValue,
    AnswerCompleteness,
    AnswerAnchoring,
    DifficultyGrader,
)

__version__ = "4.1.0"
__all__ = [
    "CriticV4",
    "QuestionEvaluator",
    "ContextualAnswerability",
    "PedagogicalValue",
    "AnswerCompleteness",
    "AnswerAnchoring",
    "DifficultyGrader",
]
