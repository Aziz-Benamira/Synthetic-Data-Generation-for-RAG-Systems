#!/usr/bin/env python3
"""
Étape 3: Évaluation avec Critic V2
===================================

Évalue toutes les paires QA avec le nouveau Critic V2 (4 métriques spécialisées).
Sauvegarde les résultats détaillés pour analyse.

Input: data/qa_samples.json
Output: results/detailed_results.json, summary_stats.json, metrics_breakdown.json
"""

import sys
import os
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/experiment.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Imports
try:
    from src.llm import LLMManager, LLMConfig
    from src.critic_v2 import CriticV2, CriticV2Config
    from src.critic_v2.base import Decision, ScoreBand
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)


def load_qa_samples(qa_path: str) -> List[Dict[str, Any]]:
    """Charger les paires QA depuis JSON"""
    logger.info(f"📖 Loading QA samples from: {qa_path}")
    with open(qa_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    qa_pairs = data.get('qa_pairs', [])
    logger.info(f"  - {len(qa_pairs)} QA pairs loaded")
    return qa_pairs


def evaluate_all_qa(
    qa_pairs: List[Dict[str, Any]],
    critic: CriticV2
) -> List[Dict[str, Any]]:
    """Évaluer toutes les paires QA"""
    
    results = []
    total = len(qa_pairs)
    
    logger.info(f"\n🔍 Evaluating {total} QA pairs...")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    for i, qa in enumerate(qa_pairs):
        logger.info(f"\n[{i+1}/{total}] QA ID: {qa['qa_id']}")
        logger.info(f"  Type: {qa['question_type']} | Chunk: {qa['metadata']['semantic_type']}")
        logger.info(f"  Q: {qa['question'][:80]}...")
        
        try:
            # Évaluer avec Critic V2
            eval_start = time.time()
            
            result = critic.evaluate(
                question=qa['question'],
                answer=qa['answer'],
                chunk_content=qa['chunk_content']
            )
            
            eval_time = time.time() - eval_start
            
            # Logger le résultat
            logger.info(f"  📊 Score: {result.overall_score:.3f} ({result.band.value})")
            logger.info(f"  🎯 Decision: {result.decision.value}")
            
            for metric_name, metric_result in result.metrics.items():
                icon = "✅" if metric_result.passed else "❌"
                logger.info(
                    f"    {icon} {metric_name}: {metric_result.score:.3f} "
                    f"({metric_result.band.value})"
                )
            
            if result.decision != Decision.PASS:
                logger.info(f"  ⚠️  Reasons: {', '.join(result.rejection_reasons[:2])}")
            
            logger.info(f"  ⏱️  Time: {eval_time:.1f}s")
            
            # Stocker le résultat
            results.append({
                "qa_id": qa['qa_id'],
                "chunk_id": qa['chunk_id'],
                "question": qa['question'],
                "answer": qa['answer'],
                "question_type": qa['question_type'],
                "semantic_type": qa['metadata']['semantic_type'],
                "evaluation": result.to_dict(),
                "evaluation_time_seconds": round(eval_time, 2)
            })
            
        except Exception as e:
            logger.error(f"  ❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Stocker l'échec
            results.append({
                "qa_id": qa['qa_id'],
                "chunk_id": qa['chunk_id'],
                "question": qa['question'],
                "answer": qa['answer'],
                "error": str(e),
                "evaluation": None
            })
    
    total_time = time.time() - start_time
    logger.info(f"\n⏱️  Total evaluation time: {total_time:.1f}s ({total_time/total:.1f}s per QA)")
    
    return results


def compute_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculer les statistiques agrégées"""
    
    valid_results = [r for r in results if r.get('evaluation') is not None]
    
    if not valid_results:
        return {"error": "No valid results"}
    
    # Décisions
    decisions = [r['evaluation']['decision'] for r in valid_results]
    decision_counts = {
        'pass': decisions.count('pass'),
        'reject': decisions.count('reject'),
        'improve': decisions.count('improve')
    }
    
    # Scores
    scores = [r['evaluation']['overall_score'] for r in valid_results]
    
    # Distribution par bande
    bands = [r['evaluation']['band'] for r in valid_results]
    band_counts = {}
    for band in bands:
        band_counts[band] = band_counts.get(band, 0) + 1
    
    # Stats par métrique
    metrics_stats = {}
    metric_names = ['anchoring', 'answer_accuracy', 'clarity', 'completeness']
    
    for metric_name in metric_names:
        metric_scores = [
            r['evaluation']['metrics'][metric_name]['score']
            for r in valid_results
        ]
        metrics_stats[metric_name] = {
            'mean': round(sum(metric_scores) / len(metric_scores), 3),
            'min': round(min(metric_scores), 3),
            'max': round(max(metric_scores), 3),
            'pass_rate': round(
                sum(1 for s in metric_scores if s >= 0.5) / len(metric_scores), 3
            )
        }
    
    # Raisons de rejet les plus fréquentes
    rejection_reasons = []
    for r in valid_results:
        if r['evaluation']['decision'] in ['reject', 'improve']:
            rejection_reasons.extend(r['evaluation']['rejection_reasons'])
    
    reason_counts = {}
    for reason in rejection_reasons:
        # Extraire le nom de la métrique (avant le ':')
        metric = reason.split(':')[0] if ':' in reason else reason
        reason_counts[metric] = reason_counts.get(metric, 0) + 1
    
    return {
        "total_evaluated": len(valid_results),
        "total_failed": len(results) - len(valid_results),
        "decision_distribution": decision_counts,
        "pass_rate": round(decision_counts['pass'] / len(valid_results), 3),
        "score_stats": {
            "mean": round(sum(scores) / len(scores), 3),
            "min": round(min(scores), 3),
            "max": round(max(scores), 3),
            "median": round(sorted(scores)[len(scores) // 2], 3)
        },
        "band_distribution": band_counts,
        "metrics_individual": metrics_stats,
        "rejection_reasons_freq": dict(sorted(
            reason_counts.items(), key=lambda x: x[1], reverse=True
        ))
    }


def save_results(
    results: List[Dict[str, Any]],
    stats: Dict[str, Any],
    output_dir: str
):
    """Sauvegarder tous les résultats"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Résultats détaillés
    detailed_path = output_path / "detailed_results.json"
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "evaluation_date": datetime.now().isoformat(),
                "critic_version": "2.0",
                "llm_model": "deepseek-r1-distill-qwen-32b"
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Detailed results: {detailed_path}")
    
    # Statistiques
    stats_path = output_path / "summary_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Summary stats: {stats_path}")
    
    # Breakdown par métrique (pour analyse)
    metrics_breakdown = {}
    for result in results:
        if result.get('evaluation'):
            for metric_name, metric_data in result['evaluation']['metrics'].items():
                if metric_name not in metrics_breakdown:
                    metrics_breakdown[metric_name] = []
                metrics_breakdown[metric_name].append({
                    "qa_id": result['qa_id'],
                    "score": metric_data['score'],
                    "band": metric_data['band'],
                    "reasoning": metric_data['reasoning']
                })
    
    breakdown_path = output_path / "metrics_breakdown.json"
    with open(breakdown_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_breakdown, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Metrics breakdown: {breakdown_path}")


def main():
    """Main execution"""
    logger.info("\n" + "=" * 60)
    logger.info("ÉTAPE 3: ÉVALUATION AVEC CRITIC V2")
    logger.info("=" * 60)
    
    # Vérifier que qa_samples.json existe
    project_root = Path(__file__).parent.parent.parent
    qa_path = project_root / "experiments/critic_v2_baseline/data/qa_samples.json"
    if not qa_path.exists():
        logger.error(f"❌ qa_samples.json not found: {qa_path}")
        logger.info("Please run 02_generate_qa_samples.py first")
        sys.exit(1)
    
    # Charger QA
    qa_pairs = load_qa_samples(str(qa_path))
    
    # Setup LLM pour Critic V2 (chargement direct GGUF)
    logger.info("\n🤖 Loading DeepSeek R1 Distill Qwen 32B directly...")
    model_path = "~/models/deepseek-r1-distill-qwen-32b/DeepSeek-R1-Distill-Qwen-32B-IQ3_M.gguf"
    
    llm = LLMManager.from_direct_llamacpp(
        model_path=model_path,
        n_gpu_layers=-1,  # All layers on GPU
        n_ctx=4096,
        verbose=False
    )
    logger.info("✅ Model loaded")
    
    # Setup Critic V2
    logger.info("\n🛠️  Initializing Critic V2...")
    config = CriticV2Config()  # Config par défaut
    config.verbose = True      # Logs détaillés
    
    critic = CriticV2(llm, config)
    logger.info(f"  Metrics: {list(critic.metrics.keys())}")
    logger.info(f"  Thresholds: reject<{config.reject_threshold}, pass>={config.pass_threshold}")
    
    # Évaluer
    results = evaluate_all_qa(qa_pairs, critic)
    
    # Statistiques
    logger.info("\n📊 Computing statistics...")
    stats = compute_statistics(results)
    
    # Sauvegarder
    logger.info("\n💾 Saving results...")
    save_results(results, stats, str(project_root / "experiments/critic_v2_baseline/results"))
    
    # Résumé final
    logger.info("\n" + "=" * 60)
    logger.info("📊 RÉSUMÉ FINAL")
    logger.info("=" * 60)
    logger.info(f"QA pairs evaluated: {stats['total_evaluated']}")
    logger.info(f"Pass rate: {stats['pass_rate']:.1%}")
    logger.info(f"Average score: {stats['score_stats']['mean']:.3f}")
    logger.info(f"\nDecision distribution:")
    for decision, count in stats['decision_distribution'].items():
        logger.info(f"  - {decision}: {count}")
    
    logger.info(f"\nScore distribution by band:")
    for band, count in stats['band_distribution'].items():
        logger.info(f"  - {band}: {count}")
    
    logger.info(f"\nMetrics performance:")
    for metric, mstats in stats['metrics_individual'].items():
        logger.info(f"  - {metric}: mean={mstats['mean']:.3f}, pass_rate={mstats['pass_rate']:.1%}")
    
    if stats['rejection_reasons_freq']:
        logger.info(f"\nTop rejection reasons:")
        for reason, count in list(stats['rejection_reasons_freq'].items())[:5]:
            logger.info(f"  - {reason}: {count}")
    
    logger.info(f"\n✅ Results saved to: {project_root}/experiments/critic_v2_baseline/results/")
    logger.info("\n➡️  Next: Run 04_analyze_results.py")


if __name__ == "__main__":
    main()
