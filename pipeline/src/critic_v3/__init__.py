"""
Critic V3 - Per-Metric Thresholds + Evolutionary Feedback Loop
================================================================

Key features:
  1. Performance Report: Shows ALL metrics with scores, percentages, pass/fail
  2. Evolutionary Feedback: Shows deltas from previous iteration (improved/degraded)
  3. Strategic Advice: Generic advice based on patterns (oscillation, trade-offs, etc.)
  4. Multi-iteration: Up to 3 attempts with cumulative history awareness
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

from src.critic_v2 import CriticV2, CriticV2Config
from src.critic_v2.per_metric_config import PerMetricConfig
from src.llm import LLMManager, LLMConfig


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class FeedbackDecision(Enum):
    """Decision after critic evaluation"""
    PASS = "pass"
    IMPROVE = "improve"
    REJECT = "reject"
    MAX_ATTEMPTS = "max_attempts"


@dataclass
class CriticV3Result:
    """Result from one evaluation iteration"""
    decision: FeedbackDecision
    metrics: Dict[str, float]
    failed_metrics: List[str]
    passing_metrics: List[str]
    overall_score: float
    feedback_message: Optional[str]
    iteration: int
    reasonings: Dict[str, str]


# ---------------------------------------------------------------------------
# Feedback generation engine
# ---------------------------------------------------------------------------

class FeedbackEngine:
    """
    Generates feedback messages that evolve across iterations.
    
    Iteration 1 → Performance Report (scores, percentages, pass/fail)
    Iteration 2+ → Evolutionary Report (deltas, pattern detection, strategic advice)
    """

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    @staticmethod
    def generate(
        metrics: Dict[str, float],
        reasonings: Dict[str, str],
        failed_metrics: List[str],
        passing_metrics: List[str],
        iteration: int,
        history: List[CriticV3Result],
    ) -> str:
        """Build the complete feedback message for the LLM."""

        if iteration == 1 or len(history) == 0:
            return FeedbackEngine._first_iteration_report(
                metrics, reasonings, failed_metrics, passing_metrics
            )
        else:
            return FeedbackEngine._evolution_report(
                metrics, reasonings, failed_metrics, passing_metrics,
                iteration, history
            )

    # ------------------------------------------------------------------
    # Iteration 1: Performance Report
    # ------------------------------------------------------------------

    @staticmethod
    def _first_iteration_report(
        metrics: Dict[str, float],
        reasonings: Dict[str, str],
        failed_metrics: List[str],
        passing_metrics: List[str],
    ) -> str:
        lines = []
        lines.append("╔══════════════════════════════════════════════════════════════╗")
        lines.append("║              RAPPORT DE PERFORMANCE (Itération 1)           ║")
        lines.append("╚══════════════════════════════════════════════════════════════╝")
        lines.append("")
        lines.append("📊 VOS SCORES:")
        lines.append("")

        for metric_name in PerMetricConfig.METRICS:
            score = metrics.get(metric_name, 0.0)
            cfg = PerMetricConfig.METRICS[metric_name]
            pct = (score / cfg.threshold) * 100 if cfg.threshold > 0 else 0
            reasoning = reasonings.get(metric_name, "")

            if metric_name in passing_metrics:
                label = "EXCELLENT" if pct >= 150 else "BON"
                star = "⭐" if pct >= 150 else "✓"
                lines.append(f"  ✅ {metric_name.upper():20s}: {score:.2f} / {cfg.threshold:.2f} ({pct:.0f}%) - {label} {star}")
                lines.append(f"     └─ Ce qui marche: {reasoning[:150]}")
            else:
                label = "ÉCHEC" if pct < 70 else "INSUFFISANT"
                lines.append(f"  ❌ {metric_name.upper():20s}: {score:.2f} / {cfg.threshold:.2f} ({pct:.0f}%) - {label}")
                lines.append(f"     └─ Problème: {cfg.feedback_prompt.split(chr(10))[0]}")
                if reasoning:
                    lines.append(f"     └─ Détail: {reasoning[:150]}")
            lines.append("")

        # Strategy section
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append("")
        lines.append("⚠️  STRATÉGIE DE CORRECTION:")
        lines.append("")
        if passing_metrics:
            lines.append("1. PRÉSERVEZ (ne touchez pas!):")
            for m in passing_metrics:
                lines.append(f"   • Votre {m} — gardez le même niveau de rigueur")
        lines.append("")
        lines.append("2. AMÉLIOREZ:")
        for m in failed_metrics:
            cfg = PerMetricConfig.METRICS[m]
            lines.append(f"   • {m}: {cfg.feedback_prompt.split(chr(10))[0]}")
        lines.append("")
        lines.append("3. ÉQUILIBRE:")
        lines.append("   Ne sacrifiez PAS une métrique réussie pour en corriger une autre!")
        lines.append("   Ajoutez UNIQUEMENT du contenu présent dans le contexte source.")
        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Iteration 2+: Evolutionary Report with Deltas + Strategic Advice
    # ------------------------------------------------------------------

    @staticmethod
    def _evolution_report(
        metrics: Dict[str, float],
        reasonings: Dict[str, str],
        failed_metrics: List[str],
        passing_metrics: List[str],
        iteration: int,
        history: List[CriticV3Result],
    ) -> str:
        prev = history[-1]  # previous iteration's result
        lines = []

        lines.append("╔══════════════════════════════════════════════════════════════╗")
        lines.append(f"║         RAPPORT ÉVOLUTIF (Itération {iteration})                     ║")
        lines.append("╚══════════════════════════════════════════════════════════════╝")
        lines.append("")
        lines.append("📊 ÉVOLUTION DE VOS SCORES:")
        lines.append("")

        for metric_name in PerMetricConfig.METRICS:
            score = metrics.get(metric_name, 0.0)
            prev_score = prev.metrics.get(metric_name, 0.0)
            cfg = PerMetricConfig.METRICS[metric_name]
            delta = score - prev_score
            pct = (score / cfg.threshold) * 100 if cfg.threshold > 0 else 0

            status = "✅" if metric_name in passing_metrics else "❌"

            # Delta arrow
            if abs(delta) < 0.01:
                delta_str = "(stable)"
                delta_icon = "➡️"
            elif delta > 0:
                delta_str = f"(+{delta:.2f})"
                delta_icon = "🎉 AMÉLIORÉ"
            else:
                delta_str = f"({delta:.2f})"
                delta_icon = "⚠️ DÉGRADÉ"

            lines.append(
                f"  {status} {metric_name.upper():20s}: {score:.2f} / {cfg.threshold:.2f} "
                f"(était {prev_score:.2f}) {delta_str} {delta_icon}"
            )

            # Context-specific comment
            if delta < -0.1 and metric_name in prev.passing_metrics:
                lines.append(
                    f"     └─ ATTENTION! Cette métrique PASSAIT avant, vous l'avez cassée!"
                )
            elif delta > 0.1 and metric_name in failed_metrics:
                lines.append(
                    f"     └─ Progrès! Mais encore insuffisant (seuil: {cfg.threshold:.2f})"
                )
            elif delta > 0.1 and metric_name in passing_metrics:
                lines.append(
                    f"     └─ BRAVO! Continuez exactement comme ça pour {metric_name}!"
                )
            elif metric_name in passing_metrics:
                lines.append(
                    f"     └─ Bien maintenu. Ne changez rien pour {metric_name}."
                )
            else:
                reasoning = reasonings.get(metric_name, "")
                lines.append(f"     └─ Problème persistant: {reasoning[:120]}")
            lines.append("")

        # Strategic advice (pattern-based, generic)
        advice = FeedbackEngine._generate_strategic_advice(metrics, history)
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append("")
        lines.append("🧠 STRATÉGIE DE CORRECTION:")
        lines.append("")
        lines.append(advice)
        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Generic Strategic Advice Generator (pattern detection)
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_strategic_advice(
        current_metrics: Dict[str, float],
        history: List[CriticV3Result],
    ) -> str:
        """
        Analyse patterns across all iterations and generate actionable advice.
        Handles: oscillation, trade-offs, stagnation, stable metrics.
        """

        all_metric_names = list(PerMetricConfig.METRICS.keys())
        prev = history[-1]

        # Classify each metric's behaviour
        improved: List[str] = []
        degraded: List[str] = []
        stable_good: List[str] = []
        stable_bad: List[str] = []
        oscillating: List[str] = []

        for m in all_metric_names:
            threshold = PerMetricConfig.METRICS[m].threshold
            scores = [h.metrics.get(m, 0.0) for h in history] + [current_metrics.get(m, 0.0)]
            last_delta = scores[-1] - scores[-2]

            # Detect oscillation (at least 3 data points needed)
            if len(scores) >= 3:
                deltas = [scores[i + 1] - scores[i] for i in range(len(scores) - 1)]
                sign_changes = sum(
                    1 for i in range(len(deltas) - 1)
                    if (deltas[i] > 0.05 and deltas[i + 1] < -0.05)
                    or (deltas[i] < -0.05 and deltas[i + 1] > 0.05)
                )
                if sign_changes >= 1:
                    oscillating.append(m)
                    continue

            if last_delta > 0.05:
                improved.append(m)
            elif last_delta < -0.05:
                degraded.append(m)
            elif scores[-1] >= threshold:
                stable_good.append(m)
            else:
                stable_bad.append(m)

        advice_parts: List[str] = []

        # --- Pattern 1: Oscillation ---
        if oscillating:
            # Find the best iteration for oscillating metrics
            all_scores = []
            for idx in range(len(history)):
                s = sum(history[idx].metrics.get(m, 0.0) for m in oscillating)
                all_scores.append((idx + 1, s))
            s_current = sum(current_metrics.get(m, 0.0) for m in oscillating)
            all_scores.append((len(history) + 1, s_current))
            best_iter, _ = max(all_scores, key=lambda x: x[1])

            advice_parts.append(
                f"💡 OSCILLATION DÉTECTÉE sur {', '.join(oscillating)}.\n"
                f"   Votre meilleure performance pour ces métriques était à l'itération {best_iter}.\n"
                f"   Revenez à l'approche de cette itération pour ces métriques."
            )

        # --- Pattern 2: Trade-off (improved some, degraded others) ---
        if improved and degraded:
            advice_parts.append(
                f"⚖️  COMPROMIS DÉTECTÉ: Vous avez amélioré {', '.join(improved)} "
                f"mais dégradé {', '.join(degraded)}.\n"
                f"   Combinez les points forts: gardez ce qui a amélioré "
                f"{', '.join(improved)} TOUT EN préservant ce qui donnait "
                f"bon {', '.join(degraded)} avant."
            )

        # --- Pattern 3: Stable good metrics to preserve ---
        if stable_good:
            advice_parts.append(
                f"🔒 PRÉSERVATION: {', '.join(stable_good)} sont stables et bonnes.\n"
                f"   Ne changez RIEN dans les aspects qui touchent à ces métriques."
            )

        # --- Pattern 4: Stagnation (same bad score) ---
        if stable_bad:
            for m in stable_bad:
                cfg = PerMetricConfig.METRICS[m]
                advice_parts.append(
                    f"🔄 STAGNATION sur {m} — la même erreur persiste.\n"
                    f"   Essayez une approche DIFFÉRENTE: {cfg.feedback_prompt.split(chr(10))[0]}"
                )

        # --- Pattern 5: Degradation without trade-off ---
        if degraded and not improved:
            advice_parts.append(
                f"❗ RÉGRESSION: {', '.join(degraded)} ont empiré sans amélioration ailleurs.\n"
                f"   Revenez à votre réponse précédente et faites des modifications plus ciblées."
            )

        # --- Final priority ---
        current_failed = [
            m for m in all_metric_names
            if current_metrics.get(m, 0.0) < PerMetricConfig.METRICS[m].threshold
        ]
        if current_failed:
            advice_parts.append(
                f"🎯 PRIORITÉ: Concentrez vos efforts UNIQUEMENT sur "
                f"{', '.join(current_failed)}."
            )

        if not advice_parts:
            advice_parts.append(
                "Continuez sur cette lancée, vous êtes proche du seuil "
                "pour toutes les métriques."
            )

        return "\n\n".join(advice_parts)


# ---------------------------------------------------------------------------
# Main Critic V3 class
# ---------------------------------------------------------------------------

class CriticV3WithFeedback:
    """
    Critic V3 — evaluate → feedback report → regenerate → re-evaluate (up to 3×).
    """

    def __init__(
        self,
        critic_v2: CriticV2,
        config: Optional[CriticV2Config] = None,
        max_iterations: int = 3,
    ):
        self.critic = critic_v2
        self.config = config or CriticV2Config()
        self.max_iterations = max_iterations

    # ------------------------------------------------------------------
    # Single evaluation
    # ------------------------------------------------------------------

    def evaluate_with_feedback(
        self,
        question: str,
        answer: str,
        chunk_content: str,
        iteration: int = 1,
        history: Optional[List[CriticV3Result]] = None,
    ) -> CriticV3Result:
        """Evaluate answer and generate feedback if metrics fail."""

        eval_result = self.critic.evaluate(question, answer, chunk_content)

        metrics = {m: r.score for m, r in eval_result.metrics.items()}
        reasonings = {m: r.reasoning for m, r in eval_result.metrics.items()}

        passes, failed = PerMetricConfig.check_pass(metrics)
        passing = [m for m in metrics if m not in failed]
        overall = PerMetricConfig.calculate_overall_score(metrics)

        if passes:
            decision = FeedbackDecision.PASS
            feedback = None
        elif iteration >= self.max_iterations:
            decision = FeedbackDecision.MAX_ATTEMPTS
            feedback = None
        else:
            critical = any(metrics[m] < 0.3 for m in failed)
            decision = FeedbackDecision.REJECT if critical else FeedbackDecision.IMPROVE

            feedback = FeedbackEngine.generate(
                metrics=metrics,
                reasonings=reasonings,
                failed_metrics=failed,
                passing_metrics=passing,
                iteration=iteration,
                history=history or [],
            )

        return CriticV3Result(
            decision=decision,
            metrics=metrics,
            failed_metrics=failed,
            passing_metrics=passing,
            overall_score=overall,
            feedback_message=feedback,
            iteration=iteration,
            reasonings=reasonings,
        )

    # ------------------------------------------------------------------
    # Answer regeneration
    # ------------------------------------------------------------------

    def regenerate_with_feedback(
        self,
        question: str,
        current_answer: str,
        chunk_content: str,
        feedback: str,
        iteration: int,
        llm: LLMManager,
    ) -> str:
        """Generate improved answer using feedback report."""

        prompt = f"""Vous êtes un assistant pédagogique expert. Votre réponse précédente a été évaluée par un critique automatique et nécessite des améliorations.

QUESTION:
{question}

CONTEXTE SOURCE (à respecter strictement — ne rien inventer en dehors):
{chunk_content}

VOTRE RÉPONSE PRÉCÉDENTE (Tentative {iteration}):
{current_answer}

{feedback}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VOTRE TÂCHE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Régénérez une réponse AMÉLIORÉE qui:
1. Corrige TOUS les problèmes identifiés dans les métriques échouées
2. PRÉSERVE les qualités des métriques qui ont réussi (très important!)
3. Respecte STRICTEMENT le contexte source fourni (pas d'hallucination)
4. Maintient un équilibre entre tous les critères de qualité

Réponse améliorée:"""

        import re
        gen_config = LLMConfig(max_tokens=600, temperature=0.3)
        response = llm.generate(prompt, config=gen_config)
        # Strip DeepSeek R1 <think> tags
        content = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL)
        return content.strip()

    # ------------------------------------------------------------------
    # Complete feedback loop
    # ------------------------------------------------------------------

    def feedback_loop(
        self,
        question: str,
        initial_answer: str,
        chunk_content: str,
        llm: LLMManager,
        verbose: bool = True,
    ) -> Tuple[str, List[CriticV3Result]]:
        """
        Complete feedback loop: evaluate → report → regenerate (up to max_iterations).

        Returns:
            (best_answer, history)
        """
        history: List[CriticV3Result] = []
        current_answer = initial_answer

        for iteration in range(1, self.max_iterations + 1):
            if verbose:
                print(f"\n{'=' * 80}")
                print(f"ITERATION {iteration}")
                print(f"{'=' * 80}\n")

            result = self.evaluate_with_feedback(
                question=question,
                answer=current_answer,
                chunk_content=chunk_content,
                iteration=iteration,
                history=history,
            )
            history.append(result)

            if verbose:
                self._print_result(result, history)

            if result.decision == FeedbackDecision.PASS:
                if verbose:
                    print("\n✅ Answer PASSED all metrics!")
                return current_answer, history

            if result.decision == FeedbackDecision.MAX_ATTEMPTS:
                if verbose:
                    print(f"\n⚠️  Maximum iterations reached ({self.max_iterations})")
                    best_idx = max(
                        range(len(history)),
                        key=lambda i: history[i].overall_score,
                    )
                    print(
                        f"   Best iteration was #{best_idx + 1} "
                        f"(score {history[best_idx].overall_score:.3f})"
                    )
                return current_answer, history

            if verbose:
                print(f"\n🔄 Regenerating answer (attempt {iteration + 1})...")
                print(f"\n--- FEEDBACK SENT TO LLM ---")
                print(result.feedback_message)
                print(f"--- END FEEDBACK ---\n")

            current_answer = self.regenerate_with_feedback(
                question=question,
                current_answer=current_answer,
                chunk_content=chunk_content,
                feedback=result.feedback_message,
                iteration=iteration,
                llm=llm,
            )

        return current_answer, history

    # ------------------------------------------------------------------
    # Pretty-print helpers
    # ------------------------------------------------------------------

    def _print_result(self, result: CriticV3Result, history: List[CriticV3Result]):
        """Pretty print evaluation result with optional delta."""
        print(f"Overall Score: {result.overall_score:.3f}")
        print(f"Decision: {result.decision.value.upper()}")
        print()
        print("Metrics:")

        prev = history[-2] if len(history) >= 2 else None

        for metric, score in result.metrics.items():
            threshold = PerMetricConfig.METRICS[metric].threshold
            status = "✅" if metric in result.passing_metrics else "❌"

            if prev:
                delta = score - prev.metrics.get(metric, 0.0)
                if abs(delta) < 0.01:
                    delta_s = ""
                elif delta > 0:
                    delta_s = f" (+{delta:.2f} ↑)"
                else:
                    delta_s = f" ({delta:.2f} ↓)"
            else:
                delta_s = ""

            print(f"  {status} {metric:20s}: {score:.3f} / {threshold:.2f}{delta_s}")
