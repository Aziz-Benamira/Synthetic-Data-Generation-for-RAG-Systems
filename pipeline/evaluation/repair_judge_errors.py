"""
Repair Judge Errors
====================
Reexécute le LLM-as-Judge uniquement sur les questions où il a échoué
(context window exceeded). Charge le detailed JSON existant, corrige
les -1, re-sauvegarde tout + rapport lisible.

Usage:
  python3 evaluation/repair_judge_errors.py \
      --detailed evaluation/results/detailed_20260303_161834.json \
      --n-ctx 3072
"""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import llm_judge_score, compute_aggregate_metrics, evaluate_single_qa

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detailed", required=True, help="Path to detailed_*.json")
    parser.add_argument("--judge-model", default=None, help="Override judge model path")
    parser.add_argument("--n-ctx", type=int, default=3072)
    args = parser.parse_args()

    detailed_path = Path(args.detailed)
    judge_model = args.judge_model or str(Path.home() / "models/deepseek-r1-distill-qwen-32b/DeepSeek-R1-Distill-Qwen-32B-IQ3_M.gguf")

    logger.info(f"Loading detailed results: {detailed_path}")
    with open(detailed_path) as f:
        data = json.load(f)

    # Trouver les erreurs
    errors = [d for d in data if d["metrics"].get("llm_judge", {}).get("score_moyen", 0) == -1]
    logger.info(f"Found {len(errors)} judge errors to repair: {[d['index'] for d in errors]}")

    if not errors:
        logger.info("Nothing to repair.")
        return

    # Charger DeepSeek R1
    from llama_cpp import Llama
    logger.info(f"Loading judge: {judge_model} (n_ctx={args.n_ctx})")
    judge_llm = Llama(
        model_path=judge_model,
        n_gpu_layers=-1,
        n_ctx=args.n_ctx,
        verbose=False,
        chat_format="chatml"
    )
    logger.info("Judge loaded.")

    # Réparer chaque entrée
    for d in errors:
        idx = d["index"]
        logger.info(f"  Repairing Q{idx:02d}...")

        score = llm_judge_score(
            question=d["question"],
            gold_answer=d["gold_answer"],
            generated_answer=d["generated_answer"],
            context=d["metrics"]["retrieval"].get("context_used", "")
                    or d.get("context_used", ""),
            llm=judge_llm,
            temperature=0.1,
            max_tokens=300
        )

        d["metrics"]["llm_judge"] = score
        logger.info(f"    → Q{idx:02d}: {score['score_moyen']}/5 "
                    f"(exact={score['exactitude']} compl={score['completude']} "
                    f"fidel={score['fidelite']} clair={score['clarte']})")

    # Re-calculer les agrégats
    logger.info("Recomputing aggregate metrics...")
    all_results = [d["metrics"] for d in data]
    aggregate = compute_aggregate_metrics(all_results)

    # Sauvegarder le detailed JSON corrigé
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved repaired detailed: {detailed_path}")

    # Recréer le summary JSON
    summary_path = detailed_path.parent / detailed_path.name.replace("detailed_", "summary_")
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        summary["aggregate_metrics"] = aggregate
        summary["judge_repaired"] = True
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved repaired summary: {summary_path}")

    # Recréer le rapport lisible
    report_path = detailed_path.parent / detailed_path.name.replace("detailed_", "rapport_lisible_").replace(".json", ".txt")
    _write_report(report_path, data, aggregate)
    logger.info(f"Saved repaired rapport: {report_path}")

    # Afficher résumé final
    print("\n" + "="*60)
    print("  RÉSULTATS FINAUX (après réparation)")
    print("="*60)
    r = aggregate["retrieval"]
    g = aggregate["generation"]
    print(f"  Hit@5        : {r['hit_rate@5']:.1%}")
    print(f"  Hit@3        : {r['hit_rate@3']:.1%}")
    print(f"  Hit@1        : {r['hit_rate@1']:.1%}")
    print(f"  MRR          : {r['mrr']:.4f}")
    print(f"  ROUGE-L F1   : {g['rouge_l_f1_mean']:.4f}")
    print(f"  BERTScore F1 : {g['bert_score_f1_mean']:.4f}")
    if "llm_judge" in aggregate:
        j = aggregate["llm_judge"]
        print(f"  LLM Judge    : {j['score_moyen_mean']:.2f}/5  (n={j['count']})")
    print("="*60)


def _write_report(path, data, aggregate):
    """Recrée le rapport lisible .txt complet."""
    sep  = "=" * 80
    sep2 = "-" * 80
    lines = []
    lines.append(sep)
    lines.append("  RAPPORT D'ÉVALUATION RAG — LECTURE MANUELLE (repaired)")
    lines.append(sep)
    lines.append(f"  Questions  : {len(data)}")
    lines.append(sep)

    for d in data:
        ret = d["metrics"]["retrieval"]
        gen = d["metrics"]["generation"]
        judge = d["metrics"].get("llm_judge", None)

        lines.append("")
        lines.append(f"╔══ QUESTION {d['index']:02d}/{len(data)} {'═' * 60}")
        lines.append(f"║  Chapitre : {d.get('chapter', '')}")
        lines.append(f"║  Section  : {d.get('section', '')}")
        lines.append(f"║  Gold chunk: {d['gold_chunk_id']}  (gold score: {d.get('gold_score', 0):.3f})")
        lines.append("╚" + "═" * 70)

        lines.append("")
        lines.append("▶ QUESTION :")
        lines.append(f"  {d['question']}")

        lines.append("")
        lines.append("▶ RÉPONSE GOLD :")
        lines.append(f"  {d['gold_answer']}")

        lines.append("")
        lines.append("▶ RÉPONSE RAG :")
        lines.append(f"  {d['generated_answer']}")

        lines.append("")
        lines.append("▶ RETRIEVAL :")
        hit = "✅" if ret["hit_rate_at_5"] == 1.0 else "❌"
        lines.append(f"  Chunks top-5    : {ret['retrieved_ids']}")
        lines.append(f"  Gold trouvé @5  : {hit}  MRR={ret['mrr']:.3f}  Hit@3={ret['hit_rate_at_3']:.0f}  Hit@1={ret['hit_rate_at_1']:.0f}")
        lines.append(f"  Similarité moy. : {ret['avg_similarity']:.4f}")
        lines.append(f"  Precision ctx   : {ret['contextual_precision']:.3f}")

        lines.append("")
        lines.append("▶ MÉTRIQUES GÉNÉRATION :")
        lines.append(f"  ROUGE-L F1   : {gen['rouge_l']['f1']:.4f}  (P={gen['rouge_l']['precision']:.3f}  R={gen['rouge_l']['recall']:.3f})")
        bs = gen.get("bert_score", {})
        if bs.get("f1", -1) >= 0:
            lines.append(f"  BERTScore F1 : {bs['f1']:.4f}  (P={bs['precision']:.3f}  R={bs['recall']:.3f})")
        lines.append(f"  Word Overlap : {gen['word_overlap']:.4f}")
        lines.append(f"  Faithfulness : {gen['faithfulness']:.4f}")

        if judge and judge.get("score_moyen", -1) >= 0:
            lines.append("")
            lines.append("▶ LLM-AS-JUDGE (DeepSeek R1) :")
            lines.append(f"  Score moyen  : {judge['score_moyen']:.2f}/5")
            lines.append(f"  Exactitude   : {judge['exactitude']}/5")
            lines.append(f"  Complétude   : {judge['completude']}/5")
            lines.append(f"  Fidélité     : {judge['fidelite']}/5")
            lines.append(f"  Clarté       : {judge['clarte']}/5")
            lines.append(f"  Commentaire  : {judge['commentaire']}")
        elif judge:
            lines.append("")
            lines.append(f"▶ LLM-AS-JUDGE : ERREUR — {judge.get('commentaire', '')}")

        lines.append(f"  ⏱  Retrieval: {d.get('retrieval_time',0):.2f}s  "
                     f"Génération: {d.get('generation_time',0):.1f}s  "
                     f"Évaluation: {d.get('evaluation_time',0):.1f}s")
        lines.append(sep2)

    # Résumé agrégé
    lines.append("")
    lines.append(sep)
    lines.append("  RÉSUMÉ AGRÉGÉ")
    lines.append(sep)
    r = aggregate["retrieval"]
    g = aggregate["generation"]
    lines.append(f"  Hit Rate @5   : {r['hit_rate@5']:.1%}")
    lines.append(f"  Hit Rate @3   : {r['hit_rate@3']:.1%}")
    lines.append(f"  Hit Rate @1   : {r['hit_rate@1']:.1%}")
    lines.append(f"  MRR           : {r['mrr']:.4f}")
    lines.append(f"  Avg Similarity: {r['avg_similarity']:.4f}")
    lines.append("")
    lines.append(f"  ROUGE-L F1    : {g['rouge_l_f1_mean']:.4f}  (médiane: {g['rouge_l_f1_median']:.4f})")
    lines.append(f"  Word Overlap  : {g['word_overlap_mean']:.4f}")
    lines.append(f"  Faithfulness  : {g['faithfulness_mean']:.4f}")
    if "bert_score_f1_mean" in g:
        lines.append(f"  BERTScore F1  : {g['bert_score_f1_mean']:.4f}  (médiane: {g['bert_score_f1_median']:.4f})")
    if "llm_judge" in aggregate:
        j = aggregate["llm_judge"]
        lines.append(f"  LLM Judge     : {j['score_moyen_mean']:.2f}/5  (médiane: {j['score_moyen_median']:.2f}/5, n={j['count']})")
    lines.append(sep)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
