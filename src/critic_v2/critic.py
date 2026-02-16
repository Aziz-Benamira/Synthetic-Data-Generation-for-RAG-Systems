"""
Critic V2 - Orchestrateur Principal
====================================

Orchestre les 4 métriques spécialisées et agrège les résultats.

Usage:
    from src.critic_v2 import CriticV2
    from src.llm import LLMManager
    
    # Créer le LLM manager (DeepSeek R1 via llama.cpp)
    llm = LLMManager.from_llamacpp("deepseek-r1-distill-qwen-32b")
    
    # Créer le critic avec config par défaut
    critic = CriticV2(llm)
    
    # Évaluer une paire QA
    result = critic.evaluate(
        question="Qu'est-ce que la loi normale?",
        answer="La loi normale est une distribution...",
        chunk_content="La loi normale, aussi appelée..."
    )
    
    print(result.decision)       # Decision.PASS / REJECT / IMPROVE
    print(result.overall_score)  # 0.78
    print(result.format_feedback())
    
    # Évaluer un batch
    results, stats = critic.evaluate_batch(qa_pairs)
"""

import logging
import time
from typing import List, Dict, Any, Tuple, Optional

from .base import (
    BaseMetric,
    MetricResult,
    EvaluationResult,
    Decision,
    ScoreBand
)
from .config import CriticV2Config
from .metrics import (
    AnchoringMetric,
    AnswerAccuracyMetric,
    ClarityMetric,
    CompletenessMetric
)

logger = logging.getLogger(__name__)


class CriticV2:
    """
    Orchestrateur Critic V2 - Évaluation par métrique individuelle.
    
    Architecture:
    1. Chaque métrique a son propre prompt spécialisé
    2. Les métriques sont exécutées séquentiellement
    3. Les scores sont agrégés avec pondération
    4. La décision finale utilise les seuils configurés
    
    Patterns professionnels utilisés:
    - Anchoring: Ragas Faithfulness NLI 2-step
    - Answer Accuracy: Nvidia Double Judge 0-2-4
    - Clarity: G-Eval Rubric 1-3
    - Completeness: Context Recall adapté
    """
    
    def __init__(
        self,
        llm_manager,
        config: Optional[CriticV2Config] = None
    ):
        """
        Args:
            llm_manager: Instance de LLMManager (src.llm.LLMManager)
            config: Configuration (seuils, poids, etc.)
        """
        self.llm = llm_manager
        self.config = config or CriticV2Config()
        
        # Initialiser les métriques
        self.metrics: Dict[str, BaseMetric] = {}
        self._init_metrics()
        
        logger.info(
            f"CriticV2 initialized with {len(self.metrics)} metrics: "
            f"{list(self.metrics.keys())}"
        )
    
    def _init_metrics(self):
        """Initialiser les métriques activées selon la config"""
        metric_classes = {
            "anchoring": AnchoringMetric,
            "answer_accuracy": AnswerAccuracyMetric,
            "clarity": ClarityMetric,
            "completeness": CompletenessMetric,
        }
        
        for name, cls in metric_classes.items():
            metric_cfg = self.config.get_metric_config(name)
            if metric_cfg.enabled:
                self.metrics[name] = cls(llm_manager=self.llm)
                logger.debug(f"  Metric '{name}' enabled (weight={metric_cfg.weight})")
    
    def evaluate(
        self,
        question: str,
        answer: str,
        chunk_content: str,
        **kwargs
    ) -> EvaluationResult:
        """
        Évaluer une paire QA sur toutes les métriques.
        
        Args:
            question: La question
            answer: La réponse
            chunk_content: Le contenu du chunk source
            **kwargs: Paramètres additionnels passés aux métriques
            
        Returns:
            EvaluationResult complet avec tous les scores et la décision
        """
        start_time = time.time()
        metric_results: Dict[str, MetricResult] = {}
        total_tokens = 0
        
        # --- Exécuter chaque métrique ---
        for name, metric in self.metrics.items():
            metric_cfg = self.config.get_metric_config(name)
            
            if self.config.verbose:
                logger.info(f"  📊 Évaluation '{name}' (poids={metric_cfg.weight})...")
            
            try:
                result = metric.evaluate(
                    question=question,
                    answer=answer,
                    chunk_content=chunk_content,
                    **kwargs
                )
                metric_results[name] = result
                total_tokens += result.tokens_used
                
                if self.config.verbose:
                    icon = "✅" if result.score >= metric_cfg.pass_threshold else "❌"
                    logger.info(
                        f"  {icon} {name}: {result.score:.3f} ({result.band.value})"
                    )
                    
            except Exception as e:
                logger.error(f"  ❌ Metric '{name}' failed: {e}")
                metric_results[name] = MetricResult(
                    metric_name=name,
                    score=0.0,
                    reasoning=f"Metric evaluation failed: {e}",
                    details={"error": str(e)}
                )
        
        # --- Calculer le score global pondéré ---
        overall_score = self._calculate_weighted_score(metric_results)
        
        # --- Déterminer la décision ---
        decision, rejection_reasons, suggestions = self._make_decision(
            metric_results, overall_score
        )
        
        elapsed = time.time() - start_time
        
        if self.config.verbose:
            logger.info(
                f"  🏁 Décision: {decision.value} "
                f"(score={overall_score:.3f}, {elapsed:.1f}s)"
            )
        
        return EvaluationResult(
            question=question,
            answer=answer,
            chunk_content=chunk_content,
            metrics=metric_results,
            decision=decision,
            overall_score=overall_score,
            rejection_reasons=rejection_reasons,
            improvement_suggestions=suggestions,
            total_tokens=total_tokens
        )
    
    def _calculate_weighted_score(
        self, 
        results: Dict[str, MetricResult]
    ) -> float:
        """
        Calculer le score global pondéré.
        
        Score = Σ(weight_i × score_i) / Σ(weight_i)
        """
        weighted_sum = 0.0
        total_weight = 0.0
        
        for name, result in results.items():
            weight = self.config.get_metric_config(name).weight
            weighted_sum += weight * result.score
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return round(weighted_sum / total_weight, 3)
    
    def _make_decision(
        self,
        results: Dict[str, MetricResult],
        overall_score: float
    ) -> Tuple[Decision, List[str], List[str]]:
        """
        Prendre la décision finale basée sur les scores.
        
        Returns:
            (decision, rejection_reasons, improvement_suggestions)
        """
        rejection_reasons = []
        suggestions = []
        
        # Vérifier les métriques individuelles
        failed_metrics = []
        for name, result in results.items():
            threshold = self.config.get_metric_config(name).pass_threshold
            if result.score < threshold:
                failed_metrics.append(name)
                rejection_reasons.append(
                    f"{name}: {result.score:.2f} < {threshold} ({result.band.value})"
                )
                # Suggestion d'amélioration basée sur la métrique
                suggestions.append(self._get_improvement_suggestion(name, result))
        
        # Mode strict: toutes les métriques doivent passer
        if self.config.strict_mode and failed_metrics:
            return Decision.REJECT, rejection_reasons, suggestions
        
        # Mode normal: score global
        if overall_score < self.config.reject_threshold:
            return Decision.REJECT, rejection_reasons, suggestions
        elif overall_score < self.config.pass_threshold:
            return Decision.IMPROVE, rejection_reasons, suggestions
        else:
            return Decision.PASS, [], []
    
    def _get_improvement_suggestion(
        self, 
        metric_name: str, 
        result: MetricResult
    ) -> str:
        """Générer une suggestion d'amélioration pour une métrique échouée"""
        suggestions_map = {
            "anchoring": (
                "Ancrage insuffisant: Utiliser UNIQUEMENT les informations du chunk. "
                "Supprimer tout exemple, inférence ou connaissance externe."
            ),
            "answer_accuracy": (
                "Exactitude insuffisante: Vérifier que chaque fait correspond "
                "EXACTEMENT au contexte source. Attention aux chiffres et conditions."
            ),
            "clarity": (
                "Clarté insuffisante: Utiliser un vocabulaire académique précis. "
                "Éviter le langage oral ('truc', 'ça', 'comment on'). "
                "Structurer la réponse clairement."
            ),
            "completeness": (
                "Complétude insuffisante: La réponse ne couvre pas tous les aspects "
                "de la question. Développer les points manquants."
            ),
        }
        
        base = suggestions_map.get(metric_name, f"Améliorer {metric_name}")
        
        # Ajouter les détails spécifiques de la métrique
        if metric_name == "anchoring" and result.details.get("unsupported_statements"):
            statements = result.details["unsupported_statements"]
            if statements:
                first = statements[0]
                base += f" Statement non supporté: \"{first.get('statement', '?')}\""
        
        elif metric_name == "completeness" and result.details.get("missing_aspects"):
            missing = result.details["missing_aspects"]
            if missing:
                base += f" Aspects manquants: {', '.join(missing[:3])}"
        
        return base
    
    def evaluate_batch(
        self,
        qa_pairs: List[Dict[str, str]],
        progress_callback: Optional[callable] = None
    ) -> Tuple[List[EvaluationResult], Dict[str, Any]]:
        """
        Évaluer un batch de paires QA.
        
        Args:
            qa_pairs: Liste de dicts avec keys: question, answer, chunk_content
            progress_callback: Callback optionnel(current, total)
            
        Returns:
            (résultats, statistiques)
        """
        results = []
        total = len(qa_pairs)
        
        logger.info(f"🔍 Évaluation batch de {total} paires QA...")
        
        for i, pair in enumerate(qa_pairs):
            if self.config.verbose:
                logger.info(f"\n--- QA {i+1}/{total} ---")
            
            result = self.evaluate(
                question=pair["question"],
                answer=pair["answer"],
                chunk_content=pair["chunk_content"]
            )
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        # --- Statistiques ---
        stats = self._compute_batch_stats(results)
        
        logger.info(f"\n📊 Batch terminé: {stats['pass_rate']:.1%} pass rate")
        
        return results, stats
    
    def _compute_batch_stats(
        self, 
        results: List[EvaluationResult]
    ) -> Dict[str, Any]:
        """Calculer les statistiques d'un batch"""
        total = len(results)
        if total == 0:
            return {"total": 0}
        
        passed = sum(1 for r in results if r.decision == Decision.PASS)
        rejected = sum(1 for r in results if r.decision == Decision.REJECT)
        improved = sum(1 for r in results if r.decision == Decision.IMPROVE)
        
        # Moyennes par métrique
        metric_avgs = {}
        for metric_name in self.metrics:
            scores = [
                r.metrics[metric_name].score 
                for r in results 
                if metric_name in r.metrics
            ]
            if scores:
                metric_avgs[metric_name] = {
                    "mean": round(sum(scores) / len(scores), 3),
                    "min": round(min(scores), 3),
                    "max": round(max(scores), 3),
                }
        
        # Raisons de rejet les plus fréquentes
        rejection_counts = {}
        for r in results:
            for reason in r.rejection_reasons:
                metric = reason.split(":")[0] if ":" in reason else reason
                rejection_counts[metric] = rejection_counts.get(metric, 0) + 1
        
        return {
            "total": total,
            "passed": passed,
            "rejected": rejected,
            "to_improve": improved,
            "pass_rate": passed / total,
            "reject_rate": rejected / total,
            "average_score": round(
                sum(r.overall_score for r in results) / total, 3
            ),
            "metric_averages": metric_avgs,
            "rejection_breakdown": rejection_counts,
            "score_distribution": {
                band.value: sum(
                    1 for r in results if r.band == band
                )
                for band in ScoreBand
            }
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Obtenir les informations de configuration"""
        return {
            "version": "2.0",
            "metrics": list(self.metrics.keys()),
            "config": {
                "strict_mode": self.config.strict_mode,
                "reject_threshold": self.config.reject_threshold,
                "pass_threshold": self.config.pass_threshold,
                "weights": {
                    name: self.config.get_metric_config(name).weight
                    for name in self.metrics
                }
            },
            "llm": self.llm.get_info() if hasattr(self.llm, 'get_info') else {}
        }
