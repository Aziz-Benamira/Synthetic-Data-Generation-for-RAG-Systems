"""
Critic V2 - Classes de Base
============================

Dataclasses et classes abstraites pour le système d'évaluation par métrique.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class Decision(Enum):
    """Décision finale pour une paire QA"""
    PASS = "pass"         # Inclure dans le dataset
    REJECT = "reject"     # Supprimer
    IMPROVE = "improve"   # Renvoyer pour amélioration


class ScoreBand(Enum):
    """Bandes de score avec interprétation"""
    EXCELLENT = "excellent"   # > 0.85
    GOOD = "good"             # 0.7 - 0.85
    ACCEPTABLE = "acceptable" # 0.5 - 0.7
    MEDIOCRE = "mediocre"     # 0.3 - 0.5
    BAD = "bad"               # < 0.3

    @classmethod
    def from_score(cls, score: float) -> "ScoreBand":
        """Convertir un score en bande"""
        if score > 0.85:
            return cls.EXCELLENT
        elif score > 0.70:
            return cls.GOOD
        elif score > 0.50:
            return cls.ACCEPTABLE
        elif score > 0.30:
            return cls.MEDIOCRE
        else:
            return cls.BAD


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class MetricResult:
    """
    Résultat d'une métrique individuelle.
    
    Chaque métrique retourne un score normalisé [0, 1], 
    une justification step-by-step, et des détails spécifiques.
    """
    metric_name: str
    score: float                     # 0.0 - 1.0 normalisé
    reasoning: str                   # Raisonnement step-by-step du LLM
    details: Dict[str, Any] = field(default_factory=dict)  # Détails spécifiques à la métrique
    raw_llm_output: str = ""         # Sortie brute du LLM (pour debug)
    tokens_used: int = 0             # Tokens consommés
    
    @property
    def band(self) -> ScoreBand:
        return ScoreBand.from_score(self.score)
    
    @property
    def passed(self) -> bool:
        """Un score >= 0.5 est considéré passé (seuil ajustable dans config)"""
        return self.score >= 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric_name,
            "score": round(self.score, 3),
            "band": self.band.value,
            "reasoning": self.reasoning,
            "details": self.details,
            "tokens_used": self.tokens_used
        }


@dataclass
class EvaluationResult:
    """
    Résultat complet de l'évaluation d'une paire QA.
    
    Agrège les résultats de toutes les métriques individuelles.
    """
    question: str
    answer: str
    chunk_content: str
    
    # Résultats par métrique
    metrics: Dict[str, MetricResult] = field(default_factory=dict)
    
    # Décision finale
    decision: Decision = Decision.REJECT
    overall_score: float = 0.0
    
    # Détails
    rejection_reasons: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    
    # Méta
    total_tokens: int = 0
    
    @property
    def band(self) -> ScoreBand:
        return ScoreBand.from_score(self.overall_score)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "decision": self.decision.value,
            "overall_score": round(self.overall_score, 3),
            "band": self.band.value,
            "metrics": {
                name: result.to_dict() 
                for name, result in self.metrics.items()
            },
            "rejection_reasons": self.rejection_reasons,
            "improvement_suggestions": self.improvement_suggestions,
            "total_tokens": self.total_tokens
        }
    
    def format_feedback(self) -> str:
        """Formater le feedback pour renvoyer au générateur"""
        lines = ["=== FEEDBACK CRITIC V2 ==="]
        for name, result in self.metrics.items():
            icon = "✅" if result.passed else "❌"
            lines.append(f"{icon} {name}: {result.score:.2f} ({result.band.value})")
            if not result.passed:
                lines.append(f"   → {result.reasoning}")
        
        if self.improvement_suggestions:
            lines.append("\n📋 Suggestions d'amélioration:")
            for s in self.improvement_suggestions:
                lines.append(f"   - {s}")
        
        return "\n".join(lines)


# =============================================================================
# CLASSE ABSTRAITE POUR LES MÉTRIQUES
# =============================================================================

class BaseMetric(ABC):
    """
    Classe abstraite pour toutes les métriques d'évaluation.
    
    Chaque métrique implémente:
    - Un prompt spécialisé unique
    - Un parser de réponse LLM
    - Un calcul de score normalisé [0, 1]
    
    Design Pattern: 1 prompt par métrique pour attention non-diluée.
    """
    
    # Nom de la métrique (à override)
    name: str = "base_metric"
    description: str = "Base metric"
    priority: str = "MEDIUM"  # CRITICAL, HIGH, MEDIUM
    
    def __init__(self, llm_manager):
        """
        Args:
            llm_manager: Instance de LLMManager pour les appels LLM
        """
        self.llm = llm_manager
    
    @abstractmethod
    def evaluate(
        self, 
        question: str, 
        answer: str, 
        chunk_content: str, 
        **kwargs
    ) -> MetricResult:
        """
        Évaluer une paire QA sur cette métrique.
        
        Args:
            question: La question
            answer: La réponse
            chunk_content: Le contenu du chunk source
            **kwargs: Paramètres additionnels
            
        Returns:
            MetricResult avec score, raisonnement et détails
        """
        pass
    
    def _call_llm(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.1
    ) -> str:
        """
        Appel LLM centralisé via LLMManager.
        
        Args:
            prompt: Prompt utilisateur
            system_prompt: Instruction système
            temperature: Température (basse pour évaluation)
            
        Returns:
            Contenu de la réponse LLM
        """
        from src.llm import LLMConfig
        
        config = LLMConfig(
            temperature=temperature,
            max_tokens=2000,
            top_p=0.95
        )
        
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            config=config
        )
        
        return response.content
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parser robuste de réponse JSON depuis un LLM.
        
        Gère les cas où le LLM entoure le JSON de texte ou
        utilise des balises <think>.
        """
        import json
        import re
        
        # Nettoyer les balises <think>...</think> (DeepSeek R1)
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        # Essayer de parser directement
        try:
            return json.loads(cleaned.strip())
        except json.JSONDecodeError:
            pass
        
        # Chercher un bloc JSON
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Chercher dans un bloc markdown ```json ... ```
        md_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
        if md_match:
            try:
                return json.loads(md_match.group(1))
            except json.JSONDecodeError:
                pass
        
        logger.warning(f"[{self.name}] Failed to parse JSON from LLM response")
        return {}
    
    def __repr__(self):
        return f"<{self.__class__.__name__}(priority={self.priority})>"
