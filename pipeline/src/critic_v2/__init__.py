"""
Critic V2 - Évaluation Professionnelle par Métrique Individuelle
================================================================

Refonte complète du critic basée sur les patterns de Ragas, G-Eval et Nvidia.

Architecture:
- 1 prompt spécialisé par métrique (attention non-diluée)
- Patterns professionnels: NLI 2-step, Double Judge, G-Eval Rubric
- Compatible DeepSeek R1 reasoning mode
- Seuils calibrables empiriquement

Métriques:
- Anchoring (CRITIQUE) : Ragas Faithfulness - NLI à 2 étapes
- Answer Accuracy (HAUTE) : Nvidia Double Judge - 0/2/4
- Clarity (MOYENNE) : G-Eval Rubric - échelle 1-3
- Completeness (MOYENNE) : Context Recall adapté

Usage:
    from src.critic_v2 import CriticV2
    from src.llm import LLMManager
    
    llm = LLMManager.from_llamacpp("deepseek-r1-distill-qwen-32b")
    critic = CriticV2(llm)
    
    result = critic.evaluate(question, answer, chunk_content)
    print(result.decision)       # "pass" / "reject" / "improve"
    print(result.overall_score)  # 0.0 - 1.0
"""

from .base import (
    MetricResult,
    EvaluationResult,
    BaseMetric,
    Decision,
    ScoreBand
)

from .config import CriticV2Config

from .critic import CriticV2

from .metrics import (
    AnchoringMetric,
    AnswerAccuracyMetric,
    ClarityMetric,
    CompletenessMetric
)

__all__ = [
    "CriticV2",
    "CriticV2Config",
    "MetricResult",
    "EvaluationResult",
    "BaseMetric",
    "Decision",
    "ScoreBand",
    "AnchoringMetric",
    "AnswerAccuracyMetric",
    "ClarityMetric",
    "CompletenessMetric",
]
