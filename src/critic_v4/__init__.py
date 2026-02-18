"""
Critic V4 - Architecture 2-Phases pour Évaluation QA

Phase 1 (Question Filtering):
- Contextual Answerability: Vérifie si le chunk contient les informations nécessaires
- Pedagogical Value: Évalue la qualité pédagogique de la question

Phase 2 (Answer Validation):
- Answer Completeness: Vérifie que la réponse couvre tous les aspects
- Answer Anchoring: Vérifie l'ancrage de la réponse dans le chunk
"""

__version__ = "4.0.0"
