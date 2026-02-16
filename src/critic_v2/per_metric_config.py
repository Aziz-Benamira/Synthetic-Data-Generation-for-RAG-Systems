"""
Per-Metric Threshold Configuration for Critic V2

Instead of using weighted average, each metric has its own threshold.
If ANY metric fails → trigger feedback loop with metric-specific guidance.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class MetricThreshold:
    """Individual metric threshold configuration"""
    name: str
    threshold: float
    weight: float  # Still used for reporting overall score
    feedback_prompt: str  # What to tell LLM when this metric fails
    

class PerMetricConfig:
    """Per-metric threshold system based on baseline analysis"""
    
    # Analysis from 18 QA baseline:
    # - anchoring: 16/18 at 1.0, 2/18 at 0.5 → threshold 0.60
    # - answer_accuracy: 8 at 1.0, 9 at 0.75, 1 at 0.5 → threshold 0.60
    # - clarity: 13 at 1.0, 5 at 0.5 → threshold 0.60
    # - completeness: 7 at 1.0, 5 at 0.667, 6 at 0.333 → threshold 0.50 (KEY!)
    
    METRICS = {
        'anchoring': MetricThreshold(
            name='anchoring',
            threshold=0.60,
            weight=2.0,
            feedback_prompt=(
                "⚓ ANCHORING ISSUE: Your answer contains statements not supported by the context.\n"
                "Action: Remove hallucinations and stick strictly to the provided context.\n"
                "Focus: Verify every claim is directly supported by the source material."
            )
        ),
        
        'answer_accuracy': MetricThreshold(
            name='answer_accuracy',
            threshold=0.60,
            weight=1.5,
            feedback_prompt=(
                "🎯 ACCURACY ISSUE: Your answer is incorrect or doesn't properly address the question.\n"
                "Action: Re-read the question carefully and provide a correct, relevant answer.\n"
                "Focus: Ensure the answer directly addresses what is asked and is factually correct."
            )
        ),
        
        'clarity': MetricThreshold(
            name='clarity',
            threshold=0.60,
            weight=1.0,
            feedback_prompt=(
                "💡 CLARITY ISSUE: Your answer is vague, uses informal language, or lacks structure.\n"
                "Action: Write in clear, academic French with proper structure.\n"
                "Focus: Use precise vocabulary, avoid ambiguity, organize your explanation logically."
            )
        ),
        
        'completeness': MetricThreshold(
            name='completeness',
            threshold=0.50,
            weight=1.0,
            feedback_prompt=(
                "📋 COMPLETENESS ISSUE: Your answer is incomplete or missing key information.\n"
                "Action: Expand your answer to cover all aspects of the question.\n"
                "Focus: Include all relevant details, definitions, and explanations needed for full understanding."
            )
        )
    }
    
    @classmethod
    def get_metric_config(cls, metric_name: str) -> Optional[MetricThreshold]:
        """Get configuration for a specific metric"""
        return cls.METRICS.get(metric_name)
    
    @classmethod
    def check_pass(cls, metric_scores: Dict[str, float]) -> tuple[bool, list[str]]:
        """
        Check if answer passes ALL metric thresholds.
        
        Returns:
            (passes, failed_metrics): 
                - passes: True if all metrics pass
                - failed_metrics: List of metrics that failed
        """
        failed = []
        
        for metric_name, score in metric_scores.items():
            config = cls.get_metric_config(metric_name)
            if config and score < config.threshold:
                failed.append(metric_name)
        
        return (len(failed) == 0, failed)
    
    @classmethod
    def get_feedback_prompt(cls, metric_scores: Dict[str, float], 
                          metric_reasonings: Dict[str, str]) -> str:
        """
        Generate feedback prompt based on failed metrics.
        
        Args:
            metric_scores: Dict of metric_name -> score
            metric_reasonings: Dict of metric_name -> critic reasoning
            
        Returns:
            Formatted feedback prompt for LLM regeneration
        """
        passes, failed = cls.check_pass(metric_scores)
        
        if passes:
            return "✅ All metrics passed. No feedback needed."
        
        feedback_parts = [
            "🔄 FEEDBACK FROM CRITIC - Please improve your answer:\n",
            "=" * 80,
            ""
        ]
        
        for metric_name in failed:
            config = cls.get_metric_config(metric_name)
            reasoning = metric_reasonings.get(metric_name, "No reasoning provided")
            score = metric_scores[metric_name]
            
            feedback_parts.extend([
                f"\n{config.feedback_prompt}",
                f"",
                f"Current score: {score:.2f} / {config.threshold:.2f}",
                f"Critic's analysis: {reasoning}",
                f""
            ])
        
        feedback_parts.append("=" * 80)
        feedback_parts.append("\nPlease regenerate your answer addressing the above issues.")
        
        return "\n".join(feedback_parts)
    
    @classmethod
    def calculate_overall_score(cls, metric_scores: Dict[str, float]) -> float:
        """
        Calculate weighted average score (for reporting purposes).
        
        Note: This is NOT used for pass/fail decision anymore!
        It's only for reporting and comparing iterations.
        """
        total_weight = sum(m.weight for m in cls.METRICS.values() 
                          if m.name in metric_scores)
        
        weighted_sum = sum(
            metric_scores[name] * config.weight
            for name, config in cls.METRICS.items()
            if name in metric_scores
        )
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    @classmethod
    def get_decision(cls, metric_scores: Dict[str, float]) -> str:
        """
        Get decision based on per-metric thresholds.
        
        Returns:
            'pass', 'improve', or 'reject'
        """
        passes, failed = cls.check_pass(metric_scores)
        
        if passes:
            return 'pass'
        
        # Check if any metric is critically low (< 0.3)
        critical_failures = [
            name for name, score in metric_scores.items()
            if score < 0.3
        ]
        
        if critical_failures:
            return 'reject'
        
        return 'improve'


# Example usage
if __name__ == "__main__":
    # Test case 1: All pass
    scores1 = {
        'anchoring': 1.0,
        'answer_accuracy': 0.75,
        'clarity': 1.0,
        'completeness': 0.667
    }
    
    passes, failed = PerMetricConfig.check_pass(scores1)
    print(f"Test 1: {passes} (failed: {failed})")
    print(f"Decision: {PerMetricConfig.get_decision(scores1)}")
    print()
    
    # Test case 2: Completeness fails
    scores2 = {
        'anchoring': 1.0,
        'answer_accuracy': 0.75,
        'clarity': 1.0,
        'completeness': 0.333  # < 0.50 threshold!
    }
    
    passes, failed = PerMetricConfig.check_pass(scores2)
    print(f"Test 2: {passes} (failed: {failed})")
    print(f"Decision: {PerMetricConfig.get_decision(scores2)}")
    
    reasonings = {
        'completeness': "La réponse ne couvre que la définition, mais manque des exemples et des applications."
    }
    
    feedback = PerMetricConfig.get_feedback_prompt(scores2, reasonings)
    print("\nFeedback:")
    print(feedback)
