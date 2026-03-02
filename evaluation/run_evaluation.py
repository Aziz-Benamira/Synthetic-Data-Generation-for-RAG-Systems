"""
RAG Evaluation Runner
======================

Script principal d'évaluation du pipeline RAG classique
sur notre Gold Dataset synthétique (MI201).

Architecture :
  1. Charge les 100 chunks sémantiques dans ChromaDB
  2. Charge le Gold Dataset (85 paires QA)
  3. Pour chaque QA :
     a. Retrieve top-k chunks
     b. Génère une réponse RAG avec Qwen2.5-32B
     c. Évalue : retrieval metrics + generation metrics + LLM-as-Judge
  4. Agrège tout et sauvegarde les résultats

Usage :
  python evaluation/run_evaluation.py                          # défaut
  python evaluation/run_evaluation.py --top_k 3                # top-3
  python evaluation/run_evaluation.py --no-judge               # sans LLM judge
  python evaluation/run_evaluation.py --limit 10               # 10 premières QA
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Ajouter le projet au path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.rag_retriever import SemanticRetriever
from evaluation.rag_generator import RAGGenerator
from evaluation.metrics import evaluate_single_qa, compute_aggregate_metrics

# ──────────────────────────────────────────────────────────────────────────────
# Configuration par défaut
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Chemins
    "chunks_path": str(PROJECT_ROOT / "experiments/critic_v2_baseline/data/chunks_mi201.json"),
    "gold_dataset_path": str(PROJECT_ROOT / "réunion/gold_dataset_v4_full.jsonl"),
    "output_dir": str(PROJECT_ROOT / "evaluation/results"),
    
    # Modèles
    "rag_llm_path": os.path.expanduser(
        "~/models/qwen2.5-32b-instruct/Qwen2.5-32B-Instruct-Q4_K_M.gguf"
    ),
    "judge_llm_path": os.path.expanduser(
        "~/models/deepseek-r1-distill-qwen-32b/DeepSeek-R1-Distill-Qwen-32B-IQ3_M.gguf"
    ),
    
    # Embedding
    # BAAI/bge-m3 : SOTA multilingue FR+EN, 570M params, 1024-dim
    # Bien meilleur que all-MiniLM-L6-v2 (EN only, 22M) pour notre polycopié FR
    # Chemin local pour éviter le problème de chargement .bin (torch 2.5 + CVE-2025-32434)
    "embedding_model": os.path.expanduser("~/models/bge-m3"),
    "embedding_device": "cuda",  # GPU L40S disponible → ~5x plus rapide
    
    # Retrieval
    "top_k": 5,
    
    # LLM
    "n_gpu_layers": -1,
    "n_ctx": 4096,
    "temperature": 0.3,
    "max_tokens": 1024,
    
    # Evaluation
    "use_llm_judge": True,
    "bert_device": "cuda",  # GPU L40S dispo, BERTScore ~2x plus rapide
}


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(output_dir: str):
    """Configure le logging avec fichier + console."""
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, "evaluation.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_gold_dataset(path: str) -> list:
    """Charge le Gold Dataset JSONL."""
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    logger.info(f"Loaded {len(entries)} Gold QA pairs from {path}")
    return entries


# ──────────────────────────────────────────────────────────────────────────────
# Main Evaluation Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_evaluation(config: dict):
    """
    Pipeline d'évaluation complet.
    
    Étapes :
    1. Indexation des chunks sémantiques dans ChromaDB
    2. Chargement du LLM RAG (Qwen2.5-32B)
    3. Pour chaque question Gold :
       - Retrieval → top-k chunks
       - Génération → réponse RAG
       - Évaluation → toutes les métriques
    4. Agrégation et sauvegarde
    """
    start_time = time.time()
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("  RAG EVALUATION PIPELINE")
    logger.info("=" * 70)
    logger.info(f"  RAG LLM    : {Path(config['rag_llm_path']).name}")
    logger.info(f"  Judge LLM  : {Path(config['judge_llm_path']).name if config['use_llm_judge'] else 'DISABLED'}")
    logger.info(f"  Embedding  : {config['embedding_model']} (device: {config['embedding_device']})")
    logger.info(f"  Top-k      : {config['top_k']}")
    logger.info(f"  Output     : {output_dir}")
    logger.info("=" * 70)
    
    # ── Étape 1 : Indexation des chunks ──
    logger.info("\n[1/4] Indexing semantic chunks in ChromaDB...")
    retriever = SemanticRetriever(
        embedding_model=config["embedding_model"],
        collection_name="mi201_semantic",
        device=config["embedding_device"]
    )
    num_indexed = retriever.load_chunks(config["chunks_path"])
    logger.info(f"  → {num_indexed} chunks indexed")
    
    # ── Étape 2 : Chargement du LLM RAG ──
    logger.info("\n[2/4] Loading RAG LLM (Qwen2.5-32B-Instruct)...")
    rag_generator = RAGGenerator(
        model_path=config["rag_llm_path"],
        n_gpu_layers=config["n_gpu_layers"],
        n_ctx=config["n_ctx"]
    )
    
    # LLM Judge (optionnel, peut être le même ou différent)
    judge_llm = None
    if config["use_llm_judge"]:
        logger.info("  Loading Judge LLM (DeepSeek-R1-32B)...")
        # Utiliser le même modèle RAG comme juge pour éviter de charger 2 modèles
        # (Le Qwen2.5 est aussi capable de juger)
        judge_llm = rag_generator.llm
        logger.info("  → Using RAG LLM as judge (single model)")
    
    # ── Étape 3 : Évaluation QA par QA ──
    logger.info("\n[3/4] Running evaluation on Gold Dataset...")
    gold_data = load_gold_dataset(config["gold_dataset_path"])
    
    # Limite optionnelle
    limit = config.get("limit")
    if limit and limit < len(gold_data):
        logger.info(f"  ⚠ Limited to {limit} questions (out of {len(gold_data)})")
        gold_data = gold_data[:limit]
    
    all_results = []
    all_detailed = []
    
    for i, qa in enumerate(gold_data, 1):
        question = qa["question"]
        gold_answer = qa["answer"]
        gold_chunk_id = qa["chunk_id"]
        gold_score = qa.get("global_score", 0)
        chapter = qa.get("chapter", "")
        section = qa.get("section", "")
        
        logger.info(f"\n  [{i}/{len(gold_data)}] {question[:80]}...")
        logger.info(f"    Gold chunk: {gold_chunk_id} | Gold score: {gold_score}")
        
        # Récupérer le contenu du chunk gold
        gold_chunk = retriever.get_chunk_by_id(gold_chunk_id)
        gold_chunk_content = gold_chunk["content"] if gold_chunk else ""
        
        # a) Retrieval
        t0 = time.time()
        retrieved = retriever.retrieve(question, top_k=config["top_k"])
        retrieval_time = time.time() - t0
        
        retrieved_ids = [c["chunk_id"] for c in retrieved]
        hit = "✅" if gold_chunk_id in retrieved_ids else "❌"
        logger.info(f"    Retrieval: {hit} top-{config['top_k']} = {retrieved_ids}")
        logger.info(f"    Similarities: {[c['similarity'] for c in retrieved]}")
        
        # b) Generation
        t0 = time.time()
        gen_result = rag_generator.generate_answer(
            question=question,
            retrieved_chunks=retrieved,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"]
        )
        generation_time = time.time() - t0
        
        generated_answer = gen_result["answer"]
        logger.info(f"    Generated ({gen_result['tokens_used']} tokens, {generation_time:.1f}s): {generated_answer[:100]}...")
        
        # c) Evaluation
        t0 = time.time()
        eval_result = evaluate_single_qa(
            question=question,
            gold_answer=gold_answer,
            gold_chunk_id=gold_chunk_id,
            gold_chunk_content=gold_chunk_content,
            generated_answer=generated_answer,
            retrieved_chunks=retrieved,
            context_used=gen_result["context_used"],
            llm_judge=judge_llm,
            top_k=config["top_k"],
            bert_device=config["bert_device"]
        )
        eval_time = time.time() - t0
        
        # Log key metrics
        ret = eval_result["retrieval"]
        gen = eval_result["generation"]
        logger.info(f"    Metrics: Hit@5={ret['hit_rate_at_5']:.0f} MRR={ret['mrr']:.2f} "
                     f"ROUGE-L={gen['rouge_l']['f1']:.3f} Faithful={gen['faithfulness']:.3f}")
        
        if "llm_judge" in eval_result:
            judge = eval_result["llm_judge"]
            logger.info(f"    Judge: {judge['score_moyen']}/5 "
                         f"(exact={judge['exactitude']} compl={judge['completude']} "
                         f"fidel={judge['fidelite']} clair={judge['clarte']})")
        
        all_results.append(eval_result)
        
        # Détail complet pour sauvegarde
        all_detailed.append({
            "index": i,
            "question": question,
            "gold_answer": gold_answer,
            "generated_answer": generated_answer,
            "gold_chunk_id": gold_chunk_id,
            "gold_score": gold_score,
            "chapter": chapter,
            "section": section,
            "retrieved_chunk_ids": retrieved_ids,
            "retrieval_time": round(retrieval_time, 3),
            "generation_time": round(generation_time, 3),
            "evaluation_time": round(eval_time, 3),
            "metrics": eval_result
        })
    
    # ── Étape 4 : Agrégation ──
    logger.info("\n[4/4] Aggregating results...")
    aggregate = compute_aggregate_metrics(all_results)
    
    total_time = time.time() - start_time
    
    # Résumé final
    logger.info("\n" + "=" * 70)
    logger.info("  RÉSULTATS FINAUX")
    logger.info("=" * 70)
    logger.info(f"  Questions évaluées : {aggregate['total_questions']}")
    logger.info(f"  Temps total        : {total_time:.0f}s ({total_time/60:.1f}min)")
    logger.info("")
    logger.info("  RETRIEVAL :")
    logger.info(f"    Hit Rate @5  : {aggregate['retrieval']['hit_rate@5']:.1%}")
    logger.info(f"    Hit Rate @3  : {aggregate['retrieval']['hit_rate@3']:.1%}")
    logger.info(f"    Hit Rate @1  : {aggregate['retrieval']['hit_rate@1']:.1%}")
    logger.info(f"    MRR          : {aggregate['retrieval']['mrr']:.4f}")
    logger.info(f"    Avg Similarity: {aggregate['retrieval']['avg_similarity']:.4f}")
    logger.info("")
    logger.info("  GENERATION :")
    logger.info(f"    ROUGE-L F1   : {aggregate['generation']['rouge_l_f1_mean']:.4f}")
    logger.info(f"    Word Overlap : {aggregate['generation']['word_overlap_mean']:.4f}")
    logger.info(f"    Faithfulness : {aggregate['generation']['faithfulness_mean']:.4f}")
    if "bert_score_f1_mean" in aggregate["generation"]:
        logger.info(f"    BERTScore F1 : {aggregate['generation']['bert_score_f1_mean']:.4f}")
    if "llm_judge" in aggregate:
        logger.info(f"    LLM Judge    : {aggregate['llm_judge']['score_moyen_mean']:.2f}/5")
    logger.info("=" * 70)
    
    # ── Sauvegarde ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Résumé
    summary = {
        "timestamp": timestamp,
        "config": {
            "rag_llm": Path(config["rag_llm_path"]).name,
            "embedding_model": config["embedding_model"],
            "top_k": config["top_k"],
            "chunks_indexed": num_indexed,
            "questions_evaluated": len(gold_data),
        },
        "aggregate_metrics": aggregate,
        "total_time_seconds": round(total_time, 1)
    }
    
    summary_path = os.path.join(output_dir, f"summary_{timestamp}.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSummary saved: {summary_path}")
    
    # Détail par QA
    detailed_path = os.path.join(output_dir, f"detailed_{timestamp}.json")
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump(all_detailed, f, indent=2, ensure_ascii=False)
    logger.info(f"Detailed results saved: {detailed_path}")
    
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="RAG Evaluation Pipeline")
    
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of chunks to retrieve (default: 5)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of QA pairs to evaluate")
    parser.add_argument("--no-judge", action="store_true",
                        help="Disable LLM-as-Judge (faster)")
    parser.add_argument("--rag-model", type=str, default=None,
                        help="Override RAG LLM path")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--n-ctx", type=int, default=4096,
                        help="LLM context size")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = DEFAULT_CONFIG.copy()
    
    if args.top_k:
        config["top_k"] = args.top_k
    if args.limit:
        config["limit"] = args.limit
    if args.no_judge:
        config["use_llm_judge"] = False
    if args.rag_model:
        config["rag_llm_path"] = args.rag_model
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.n_ctx:
        config["n_ctx"] = args.n_ctx
    
    setup_logging(config["output_dir"])
    
    summary = run_evaluation(config)
    
    print("\n✅ Evaluation complete!")
    print(f"   Results in: {config['output_dir']}")
    
    return summary


if __name__ == "__main__":
    main()
