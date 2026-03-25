"""
Prompts système pour les métriques Critic V4

Phase 1 (Question Evaluation):
- contextual_answerability_prompt: Évalue si le chunk contient les informations nécessaires
- pedagogical_value_prompt: Évalue la qualité pédagogique de la question

Phase 2 (Answer Validation):
- answer_completeness_prompt (à implémenter)
- answer_anchoring_prompt (à implémenter)
"""

from .contextual_answerability_prompt import get_contextual_answerability_prompt
from .pedagogical_value_prompt import get_pedagogical_value_prompt
from .answer_completeness_prompt import get_answer_completeness_prompt
from .answer_anchoring_prompt import get_answer_anchoring_prompt

__all__ = [
    "get_contextual_answerability_prompt",
    "get_pedagogical_value_prompt",
    "get_answer_completeness_prompt",
    "get_answer_anchoring_prompt",
]
