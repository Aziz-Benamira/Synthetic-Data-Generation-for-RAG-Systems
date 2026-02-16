#!/usr/bin/env python3
"""
Étape 4: Analyse des résultats
===============================

Analyse approfondie des résultats du Critic V2 et génération d'un rapport.

Input: results/detailed_results.json, summary_stats.json, metrics_breakdown.json
Output: results/analysis_report.md
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_results() -> tuple:
    """Charger tous les fichiers de résultats"""
    base_path = Path("experiments/critic_v2_baseline/results")
    
    with open(base_path / "detailed_results.json", 'r') as f:
        detailed = json.load(f)
    
    with open(base_path / "summary_stats.json", 'r') as f:
        stats = json.load(f)
    
    with open(base_path / "metrics_breakdown.json", 'r') as f:
        breakdown = json.load(f)
    
    return detailed, stats, breakdown


def analyze_score_distribution(results: List[Dict]) -> Dict[str, Any]:
    """Analyser la distribution des scores"""
    scores = [r['evaluation']['overall_score'] for r in results if r.get('evaluation')]
    
    # Bins de distribution
    bins = [0, 0.3, 0.5, 0.7, 0.85, 1.0]
    bin_labels = ['[0.0-0.3)', '[0.3-0.5)', '[0.5-0.7)', '[0.7-0.85)', '[0.85-1.0]']
    bin_counts = [0] * len(bin_labels)
    
    for score in scores:
        for i in range(len(bins) - 1):
            if bins[i] <= score < bins[i+1]:
                bin_counts[i] += 1
                break
        else:
            if score == 1.0:
                bin_counts[-1] += 1
    
    return {
        "bins": bin_labels,
        "counts": bin_counts,
        "percentages": [round(c/len(scores)*100, 1) for c in bin_counts]
    }


def find_edge_cases(results: List[Dict]) -> Dict[str, List[Dict]]:
    """Identifier les cas limites intéressants"""
    
    valid_results = [r for r in results if r.get('evaluation')]
    
    # Trier par score
    sorted_results = sorted(valid_results, key=lambda x: x['evaluation']['overall_score'])
    
    return {
        "worst_3": sorted_results[:3],
        "best_3": sorted_results[-3:],
        "borderline": [
            r for r in valid_results 
            if 0.45 <= r['evaluation']['overall_score'] <= 0.55
        ]
    }


def analyze_metric_correlations(breakdown: Dict) -> Dict[str, Any]:
    """Analyser les corrélations entre métriques"""
    
    metrics = list(breakdown.keys())
    n = len(breakdown[metrics[0]])
    
    # Matrice de scores
    scores_matrix = {
        metric: [item['score'] for item in breakdown[metric]]
        for metric in metrics
    }
    
    # Corrélations simples (covariance normalisée)
    correlations = {}
    for i, m1 in enumerate(metrics):
        for m2 in metrics[i+1:]:
            scores1 = scores_matrix[m1]
            scores2 = scores_matrix[m2]
            
            # Corrélation de Pearson simplifiée
            mean1 = sum(scores1) / n
            mean2 = sum(scores2) / n
            
            cov = sum((s1 - mean1) * (s2 - mean2) for s1, s2 in zip(scores1, scores2)) / n
            std1 = (sum((s - mean1)**2 for s in scores1) / n) ** 0.5
            std2 = (sum((s - mean2)**2 for s in scores2) / n) ** 0.5
            
            if std1 > 0 and std2 > 0:
                corr = cov / (std1 * std2)
                correlations[f"{m1} <-> {m2}"] = round(corr, 3)
    
    return correlations


def generate_report(detailed: Dict, stats: Dict, breakdown: Dict) -> str:
    """Générer le rapport d'analyse complet"""
    
    results = detailed['results']
    valid_results = [r for r in results if r.get('evaluation')]
    
    report = []
    report.append("# Rapport d'Analyse - Critic V2 Baseline")
    report.append(f"\n**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"**Expérience**: critic_v2_baseline")
    report.append(f"**LLM**: {detailed['metadata']['llm_model']}")
    report.append("")
    
    # === RÉSUMÉ EXÉCUTIF ===
    report.append("## 📊 Résumé Exécutif")
    report.append("")
    report.append(f"- **Total QA évaluées**: {stats['total_evaluated']}")
    report.append(f"- **Taux de passage**: {stats['pass_rate']:.1%}")
    report.append(f"- **Score moyen**: {stats['score_stats']['mean']:.3f}")
    report.append(f"- **Étendue des scores**: [{stats['score_stats']['min']:.3f}, {stats['score_stats']['max']:.3f}]")
    report.append("")
    
    # Distribution des décisions
    report.append("### Distribution des décisions")
    report.append("")
    report.append("| Décision | Nombre | Pourcentage |")
    report.append("|----------|--------|-------------|")
    for decision, count in stats['decision_distribution'].items():
        pct = count / stats['total_evaluated'] * 100
        report.append(f"| {decision.upper()} | {count} | {pct:.1f}% |")
    report.append("")
    
    # === ANALYSE PAR MÉTRIQUE ===
    report.append("## 🎯 Performance par Métrique")
    report.append("")
    report.append("| Métrique | Moy | Min | Max | Taux Pass |")
    report.append("|----------|-----|-----|-----|-----------|")
    for metric, mstats in stats['metrics_individual'].items():
        report.append(
            f"| {metric} | {mstats['mean']:.3f} | {mstats['min']:.3f} | "
            f"{mstats['max']:.3f} | {mstats['pass_rate']:.1%} |"
        )
    report.append("")
    
    # === DISTRIBUTION DES SCORES ===
    report.append("## 📈 Distribution des Scores")
    report.append("")
    score_dist = analyze_score_distribution(valid_results)
    report.append("| Plage | Nombre | Pourcentage |")
    report.append("|-------|--------|-------------|")
    for label, count, pct in zip(score_dist['bins'], score_dist['counts'], score_dist['percentages']):
        report.append(f"| {label} | {count} | {pct}% |")
    report.append("")
    
    # Visualisation ASCII
    max_count = max(score_dist['counts']) if score_dist['counts'] else 1
    report.append("```")
    for label, count in zip(score_dist['bins'], score_dist['counts']):
        bar = '█' * int(count / max_count * 40)
        report.append(f"{label:12} {bar} {count}")
    report.append("```")
    report.append("")
    
    # === RAISONS DE REJET ===
    report.append("## ❌ Raisons de Rejet (Fréquence)")
    report.append("")
    report.append("| Métrique/Raison | Occurrences |")
    report.append("|-----------------|-------------|")
    for reason, count in list(stats['rejection_reasons_freq'].items())[:10]:
        report.append(f"| {reason} | {count} |")
    report.append("")
    
    # === CAS LIMITES ===
    report.append("## 🔍 Cas Limites Intéressants")
    report.append("")
    edge_cases = find_edge_cases(valid_results)
    
    report.append("### 🏆 Top 3 Meilleurs Scores")
    report.append("")
    for i, r in enumerate(edge_cases['best_3'], 1):
        score = r['evaluation']['overall_score']
        report.append(f"**{i}. Score: {score:.3f}** (QA: {r['qa_id']})")
        report.append(f"- Q: {r['question'][:100]}...")
        report.append(f"- Type: {r['question_type']} | Chunk: {r['semantic_type']}")
        report.append("")
    
    report.append("### ⚠️ Top 3 Pires Scores")
    report.append("")
    for i, r in enumerate(edge_cases['worst_3'], 1):
        score = r['evaluation']['overall_score']
        reasons = r['evaluation']['rejection_reasons'][:2]
        report.append(f"**{i}. Score: {score:.3f}** (QA: {r['qa_id']})")
        report.append(f"- Q: {r['question'][:100]}...")
        report.append(f"- Raisons: {', '.join(reasons)}")
        report.append("")
    
    if edge_cases['borderline']:
        report.append("### 🎲 Cas Borderline (score ~0.5)")
        report.append("")
        report.append(f"Nombre de cas borderline: {len(edge_cases['borderline'])}")
        report.append("Ces cas nécessitent une attention particulière pour calibration.")
        report.append("")
    
    # === CORRÉLATIONS ENTRE MÉTRIQUES ===
    report.append("## 🔗 Corrélations entre Métriques")
    report.append("")
    correlations = analyze_metric_correlations(breakdown)
    report.append("| Paire de métriques | Corrélation |")
    report.append("|--------------------|-------------|")
    for pair, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        report.append(f"| {pair} | {corr:+.3f} |")
    report.append("")
    report.append("*Note: Corrélation proche de +1 = forte corrélation positive, -1 = négative, 0 = indépendant*")
    report.append("")
    
    # === OBSERVATIONS ET RECOMMANDATIONS ===
    report.append("## 💡 Observations et Recommandations")
    report.append("")
    
    # Observations automatiques
    report.append("### Observations")
    report.append("")
    
    # 1. Variance des scores
    score_variance = (stats['score_stats']['max'] - stats['score_stats']['min'])
    if score_variance < 0.3:
        report.append("- ⚠️ **Faible variance des scores** ({:.2f}): Les scores sont trop concentrés. "
                     "Considérer d'ajuster les seuils ou les prompts.".format(score_variance))
    else:
        report.append("- ✅ **Bonne variance des scores** ({:.2f}): Le système discrimine bien entre "
                     "bonnes et mauvaises QA.".format(score_variance))
    report.append("")
    
    # 2. Métrique la plus stricte
    strictest_metric = min(stats['metrics_individual'].items(), key=lambda x: x[1]['pass_rate'])
    report.append(f"- 📉 **Métrique la plus stricte**: {strictest_metric[0]} "
                 f"(pass_rate={strictest_metric[1]['pass_rate']:.1%})")
    report.append("")
    
    # 3. Métrique la plus permissive
    lenient_metric = max(stats['metrics_individual'].items(), key=lambda x: x[1]['pass_rate'])
    report.append(f"- 📈 **Métrique la plus permissive**: {lenient_metric[0]} "
                 f"(pass_rate={lenient_metric[1]['pass_rate']:.1%})")
    report.append("")
    
    # Recommandations
    report.append("### Recommandations")
    report.append("")
    
    if stats['pass_rate'] < 0.3:
        report.append("1. **Taux de passage trop faible** ({:.1%}): Le système est peut-être trop strict. "
                     "Considérer:".format(stats['pass_rate']))
        report.append("   - Réduire les seuils (pass_threshold de 0.5 → 0.4)")
        report.append("   - Ajuster les poids des métriques strictes")
        report.append("   - Revoir les prompts pour être moins sévères")
    elif stats['pass_rate'] > 0.8:
        report.append("1. **Taux de passage trop élevé** ({:.1%}): Le système est peut-être trop permissif. "
                     "Considérer:".format(stats['pass_rate']))
        report.append("   - Augmenter les seuils (pass_threshold de 0.5 → 0.6)")
        report.append("   - Augmenter les poids des métriques critiques")
        report.append("   - Ajouter des few-shot examples plus stricts")
    else:
        report.append("1. **Taux de passage acceptable** ({:.1%})".format(stats['pass_rate']))
    report.append("")
    
    report.append("2. **Calibration des seuils**: Analyser les cas borderline pour affiner les seuils.")
    report.append("")
    report.append("3. **Prochaines expériences**:")
    report.append("   - Tester avec différents presets (strict vs lenient)")
    report.append("   - Comparer avec l'ancien critic")
    report.append("   - Ajuster les few-shot examples selon les faiblesses observées")
    report.append("")
    
    # === FOOTER ===
    report.append("---")
    report.append("")
    report.append("*Rapport généré automatiquement par `04_analyze_results.py`*")
    
    return "\n".join(report)


def main():
    """Main execution"""
    logger.info("=" * 60)
    logger.info("ÉTAPE 4: ANALYSE DES RÉSULTATS")
    logger.info("=" * 60)
    
    # Vérifier que les résultats existent
    results_dir = Path("experiments/critic_v2_baseline/results")
    if not results_dir.exists():
        logger.error(f"❌ Results directory not found: {results_dir}")
        logger.info("Please run 03_run_critic_v2.py first")
        sys.exit(1)
    
    # Charger les résultats
    logger.info("\n📖 Loading results...")
    detailed, stats, breakdown = load_results()
    
    # Générer le rapport
    logger.info("\n📝 Generating analysis report...")
    report = generate_report(detailed, stats, breakdown)
    
    # Sauvegarder
    output_path = results_dir / "analysis_report.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"💾 Report saved to: {output_path}")
    
    # Afficher les highlights
    logger.info("\n" + "=" * 60)
    logger.info("📊 KEY INSIGHTS")
    logger.info("=" * 60)
    logger.info(f"Pass rate: {stats['pass_rate']:.1%}")
    logger.info(f"Average score: {stats['score_stats']['mean']:.3f}")
    logger.info(f"Score range: [{stats['score_stats']['min']:.3f}, {stats['score_stats']['max']:.3f}]")
    
    logger.info(f"\nStrictest metric: {min(stats['metrics_individual'].items(), key=lambda x: x[1]['pass_rate'])[0]}")
    logger.info(f"Most lenient metric: {max(stats['metrics_individual'].items(), key=lambda x: x[1]['pass_rate'])[0]}")
    
    logger.info(f"\n✅ Full report: {output_path}")
    logger.info("\n🎉 Experiment complete! Review the report to discuss thresholds and next steps.")


if __name__ == "__main__":
    main()
