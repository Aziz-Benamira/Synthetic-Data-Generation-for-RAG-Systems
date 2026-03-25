"""
Critic V2 - Configuration
==========================

Seuils, poids, et configuration des métriques.

⚠️ Les seuils doivent être calibrés empiriquement sur un échantillon du dataset !
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class MetricConfig:
    """Configuration d'une métrique individuelle"""
    enabled: bool = True
    weight: float = 1.0          # Poids dans le score global
    pass_threshold: float = 0.5  # Seuil minimum pour passer
    temperature: float = 0.1     # Température LLM pour cette métrique


@dataclass
class CriticV2Config:
    """
    Configuration globale du Critic V2.
    
    Seuils Recommandés (à calibrer empiriquement):
        < 0.3  → REJETER (BAD)
        0.3-0.5 → AMÉLIORER (MEDIOCRE)
        0.5-0.7 → ACCEPTABLE
        0.7-0.85 → BON
        > 0.85 → EXCELLENT
    """
    
    # --- Seuils de décision globaux ---
    reject_threshold: float = 0.3    # Score global < 0.3 → REJECT
    improve_threshold: float = 0.55  # Score global 0.3-0.55 → IMPROVE
    pass_threshold: float = 0.55     # Score global >= 0.55 → PASS
    
    # --- Configuration par métrique ---
    metrics: Dict[str, MetricConfig] = field(default_factory=lambda: {
        "anchoring": MetricConfig(
            enabled=True,
            weight=2.0,           # CRITIQUE - poids double
            pass_threshold=0.5,
            temperature=0.1
        ),
        "answer_accuracy": MetricConfig(
            enabled=True,
            weight=1.5,           # HAUTE priorité
            pass_threshold=0.5,
            temperature=0.1
        ),
        "clarity": MetricConfig(
            enabled=True,
            weight=1.0,           # MOYENNE priorité
            pass_threshold=0.5,
            temperature=0.2
        ),
        "completeness": MetricConfig(
            enabled=True,
            weight=1.0,           # MOYENNE priorité
            pass_threshold=0.5,
            temperature=0.1
        ),
    })
    
    # --- Mode strict ---
    strict_mode: bool = False      # True = ALL métriques doivent passer
    # Si False, on utilise le score pondéré global
    
    # --- Langue ---
    language: str = "fr"
    
    # --- Logging ---
    verbose: bool = True
    log_raw_llm: bool = False      # Logger les réponses brutes du LLM
    
    def get_metric_config(self, metric_name: str) -> MetricConfig:
        """Obtenir la config d'une métrique"""
        return self.metrics.get(metric_name, MetricConfig())
    
    @classmethod
    def strict(cls) -> "CriticV2Config":
        """Config stricte : all metrics must pass, seuils hauts"""
        config = cls()
        config.strict_mode = True
        config.reject_threshold = 0.4
        config.improve_threshold = 0.6
        config.pass_threshold = 0.6
        return config
    
    @classmethod
    def lenient(cls) -> "CriticV2Config":
        """Config permissive : pour calibration initiale"""
        config = cls()
        config.strict_mode = False
        config.reject_threshold = 0.2
        config.improve_threshold = 0.4
        config.pass_threshold = 0.4
        return config
