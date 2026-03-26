"""
Projet Pipeline — End-to-End : PDF → Gold Dataset → RAG Evaluation
====================================================================

Ce script orchestre les deux grandes étapes du projet :

  ÉTAPE 1 — Génération du Gold Dataset
  ─────────────────────────────────────
    a) SemanticChunker : extrait N chunks variés du PDF
    b) PipelineV4      : génère les paires QA avec DeepSeek R1 + Critic

  ÉTAPE 2 — Évaluation RAG
  ─────────────────────────
    RAG classique (bge-m3 + ChromaDB + Qwen2.5-32B)
    évalué sur le Gold Dataset avec DeepSeek R1 comme juge

Usage :
  python3 Projet_Pipeline.py --pdf data/pdfs/MI201_2022_poly.pdf
  python3 Projet_Pipeline.py --pdf data/pdfs/Dunod\\ -\\ Physique\\ -\\ tout\\ en\\ un.pdf \\
      --num-chunks 200 --output-base output/physique_dunod
  python3 Projet_Pipeline.py --pdf ... --skip-gold   # sauter la génération
  python3 Projet_Pipeline.py --pdf ... --skip-eval   # sauter l'évaluation
  python3 Projet_Pipeline.py --pdf ... --chunks-only # chunking seulement

Fichiers produits dans --output-base/ :
  chunks.json                    – les chunks extraits du PDF
  gold_dataset.jsonl             – le Gold Dataset (JSONL, une QA par ligne)
  gold_dataset.json              – idem en JSON lisible
  evaluation/
    rapport_lisible_*.txt        – rapport complet d'évaluation
    summary_*.json               – métriques agrégées
    detailed_*.json              – détail par QA
    incremental_results.jsonl    – sauvegarde progressive
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ProjetPipeline")

# ─────────────────────────────────────────────────────────────────────────────
# Defaults modèles
# ─────────────────────────────────────────────────────────────────────────────

DEEPSEEK_PATH = os.path.expanduser(
    "~/models/deepseek-r1-distill-qwen-32b/DeepSeek-R1-Distill-Qwen-32B-IQ3_M.gguf"
)
QWEN_PATH = os.path.expanduser(
    "~/models/qwen2.5-32b-instruct/Qwen2.5-32B-Instruct-Q4_K_M.gguf"
)
BGE_PATH = os.path.expanduser("~/models/bge-m3")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 1a — Chunking PDF
# ─────────────────────────────────────────────────────────────────────────────

def run_chunking(pdf_path: str, output_chunks: str, num_chunks: int) -> str:
    """
    Extrait num_chunks chunks variés et équilibrés par type sémantique.
    Sauvegarde dans output_chunks (JSON).
    Retourne le chemin du fichier créé.
    """
    logger.info("=" * 60)
    logger.info("ÉTAPE 1a — EXTRACTION DE CHUNKS")
    logger.info(f"  PDF         : {pdf_path}")
    logger.info(f"  Chunks cible: {num_chunks}")
    logger.info(f"  Output      : {output_chunks}")
    logger.info("=" * 60)

    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF introuvable : {pdf_path}")

    from src.chunking.semantic_chunker import SemanticChunker

    t0 = time.time()
    chunker = SemanticChunker(
        pdf_path=pdf_path,
        target_chunk_size=1000,
        max_chunk_size=2000,
        chunk_overlap=200,
        min_chunk_size=300,
    )

    logger.info("Chunking en cours (peut prendre quelques minutes sur un gros PDF)...")
    all_chunks = chunker.chunk_document()
    logger.info(f"  → {len(all_chunks)} chunks totaux extraits en {time.time()-t0:.1f}s")

    # Distribution par type sémantique
    by_type: dict = {}
    for c in all_chunks:
        t = c.semantic_type
        by_type.setdefault(t, []).append(c)

    logger.info("  Répartition des types :")
    for t, lst in sorted(by_type.items(), key=lambda x: -len(x[1])):
        logger.info(f"    {t:20s}: {len(lst)}")

    # Sélection équilibrée + variée
    selected = _sample_diverse(by_type, num_chunks, min_len=300)
    logger.info(f"  → {len(selected)} chunks sélectionnés")

    # Log final de la sélection par type
    type_counts: dict = {}
    for c in selected:
        type_counts[c.semantic_type] = type_counts.get(c.semantic_type, 0) + 1
    for t, n in sorted(type_counts.items()):
        logger.info(f"    {t}: {n}")

    # Sérialisation
    chunks_data = []
    for c in selected:
        chunks_data.append({
            "chunk_id": c.chunk_id,
            "content": c.content,
            "chapter": c.chapter_title,
            "section": c.section_title,
            "subsection": c.subsection_title,
            "page_range": list(c.page_range),
            "semantic_type": c.semantic_type,
            "source_file": pdf_path,
            "metadata": c.metadata,
        })

    Path(output_chunks).parent.mkdir(parents=True, exist_ok=True)
    with open(output_chunks, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "source_pdf": pdf_path,
                    "num_chunks": len(chunks_data),
                    "total_chunks_in_pdf": len(all_chunks),
                    "extraction_date": datetime.now().isoformat(),
                    "type_distribution": type_counts,
                },
                "chunks": chunks_data,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(f"  ✅ Chunks sauvegardés : {output_chunks}")
    return output_chunks


def _sample_diverse(by_type: dict, total: int, min_len: int = 300):
    """Échantillonnage équilibré par type sémantique."""
    # Filtrer les trop courts
    filtered = {t: [c for c in lst if len(c.content) >= min_len]
                for t, lst in by_type.items()}
    filtered = {t: lst for t, lst in filtered.items() if lst}

    n_types = len(filtered)
    if n_types == 0:
        return []

    # Quota par type (arrondi vers le bas, puis combler avec les types les plus riches)
    base = total // n_types
    selected = []

    sorted_types = sorted(filtered.keys(), key=lambda t: -len(filtered[t]))
    remaining = total

    for i, t in enumerate(sorted_types):
        quota = base if i < n_types - 1 else remaining
        quota = min(quota, len(filtered[t]))
        sampled = random.sample(filtered[t], quota)
        selected.extend(sampled)
        remaining -= quota
        if remaining <= 0:
            break

    random.shuffle(selected)
    return selected[:total]


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 1b — Génération Gold Dataset (PipelineV4 + DeepSeek R1)
# ─────────────────────────────────────────────────────────────────────────────

def run_gold_generation(chunks_path: str, output_jsonl: str) -> str:
    """
    Lance PipelineV4 sur les chunks → génère le Gold Dataset.
    Retourne le chemin JSONL produit.
    """
    logger.info("=" * 60)
    logger.info("ÉTAPE 1b — GÉNÉRATION GOLD DATASET (DeepSeek R1 + CriticV4)")
    logger.info(f"  Chunks input : {chunks_path}")
    logger.info(f"  Output JSONL : {output_jsonl}")
    logger.info("=" * 60)

    logger.info("Chargement DeepSeek R1-32B...")
    t0 = time.time()
    from src.llm import LLMManager
    llm_manager = LLMManager.from_direct_llamacpp(
        model_path=DEEPSEEK_PATH,
        n_gpu_layers=-1,
        n_ctx=4096,
    )
    llm = llm_manager.provider.llm
    logger.info(f"  ✅ Modèle chargé en {time.time()-t0:.1f}s")

    from src.orchestrator.pipeline_v4 import PipelineV4, PipelineV4Config

    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)

    config = PipelineV4Config(
        chunks_path=chunks_path,
        output_path=output_jsonl,
        max_chunks=None,        # tous les chunks fournis
        min_chunk_length=300,
        max_q_retries=3,
        max_a_retries=2,
        checkpoint_every=10,
    )

    pipeline = PipelineV4(config=config, llm=llm)
    dataset = pipeline.run()

    # JSON lisible en plus du JSONL
    json_out = output_jsonl.replace(".jsonl", ".json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    avg = sum(e["global_score"] for e in dataset) / len(dataset) if dataset else 0
    logger.info(f"  ✅ {len(dataset)} entrées Gold générées (score moyen : {avg:.3f})")
    logger.info(f"     JSONL : {output_jsonl}")
    logger.info(f"     JSON  : {json_out}")

    # Libérer le modèle DeepSeek de la VRAM avant de charger Qwen
    del llm_manager
    del llm
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    logger.info("  → VRAM libérée (DeepSeek déchargé)")

    return output_jsonl


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 2 — Évaluation RAG
# ─────────────────────────────────────────────────────────────────────────────

def run_rag_evaluation(chunks_path: str, gold_dataset_path: str, eval_output_dir: str):
    """
    Lance l'évaluation RAG complète :
      - ChromaDB + bge-m3 (retrieval)
      - Qwen2.5-32B (génération)
      - DeepSeek R1-32B (juge LLM)
    """
    logger.info("=" * 60)
    logger.info("ÉTAPE 2 — ÉVALUATION RAG")
    logger.info(f"  Chunks     : {chunks_path}")
    logger.info(f"  Gold DS    : {gold_dataset_path}")
    logger.info(f"  Output dir : {eval_output_dir}")
    logger.info("=" * 60)

    from evaluation.rag_retriever import SemanticRetriever
    from evaluation.rag_generator import RAGGenerator
    from metrics.metrics import evaluate_single_qa, compute_aggregate_metrics

    # Config évaluation
    config = {
        "chunks_path": chunks_path,
        "gold_dataset_path": gold_dataset_path,
        "output_dir": eval_output_dir,
        "rag_llm_path": QWEN_PATH,
        "judge_llm_path": DEEPSEEK_PATH,
        "embedding_model": BGE_PATH,
        "embedding_device": "cuda",
        "top_k": 5,
        "n_gpu_layers": -1,
        "n_ctx": 8192,
        "temperature": 0.3,
        "max_tokens": 1024,
        "use_llm_judge": True,
        "bert_device": "cuda",
    }

    # Déléguer à run_evaluation.run_evaluation()
    from evaluation.run_evaluation import run_evaluation, setup_logging
    setup_logging(eval_output_dir)
    summary = run_evaluation(config)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Pipeline complet PDF → Gold Dataset → RAG Evaluation"
    )
    p.add_argument("--pdf", required=True, help="Chemin vers le PDF source")
    p.add_argument(
        "--output-base",
        default=None,
        help="Dossier de base pour tous les fichiers produits "
             "(défaut: output/<nom_pdf_sans_extension>/)",
    )
    p.add_argument(
        "--num-chunks",
        type=int,
        default=100,
        help="Nombre de chunks à extraire du PDF (défaut: 100)",
    )

    # Contrôle des étapes
    p.add_argument("--chunks-only", action="store_true",
                   help="Faire uniquement le chunking du PDF")
    p.add_argument("--skip-gold", action="store_true",
                   help="Sauter la génération du Gold Dataset (uses existing)")
    p.add_argument("--skip-eval", action="store_true",
                   help="Sauter l'évaluation RAG")

    # Fichiers préexistants (si on reprend en cours)
    p.add_argument("--chunks-file", default=None,
                   help="Utiliser ce fichier de chunks (saute le chunking)")
    p.add_argument("--gold-file", default=None,
                   help="Utiliser ce fichier Gold Dataset (saute la génération)")

    p.add_argument("--seed", type=int, default=42, help="Seed aléatoire pour le sampling")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    # ── Chemins ──────────────────────────────────────────────────────────────
    pdf_path = str(Path(args.pdf).resolve())
    pdf_stem = Path(pdf_path).stem.replace(" ", "_").replace("-", "_")

    if args.output_base:
        base = Path(args.output_base)
    else:
        base = PROJECT_ROOT / "output" / pdf_stem
    base.mkdir(parents=True, exist_ok=True)

    chunks_file   = args.chunks_file  or str(base / "chunks.json")
    gold_jsonl    = args.gold_file    or str(base / "gold_dataset.jsonl")
    eval_dir      = str(base / "evaluation")

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║           PROJET PIPELINE — END-TO-END                  ║")
    logger.info("╠══════════════════════════════════════════════════════════╣")
    logger.info(f"║  PDF        : {Path(pdf_path).name[:50]:<50} ║")
    logger.info(f"║  Output base: {str(base)[:50]:<50} ║")
    logger.info(f"║  Chunks     : {args.num_chunks:<50} ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    t_total = time.time()

    # ── Étape 1a : Chunking ──────────────────────────────────────────────────
    if args.chunks_file:
        logger.info(f"\n[SKIP] Chunking — utilisation du fichier existant : {chunks_file}")
    elif args.skip_gold and args.skip_eval:
        logger.info("\n[SKIP] Rien à faire (--skip-gold + --skip-eval sans chunking)")
        return
    else:
        run_chunking(pdf_path, chunks_file, args.num_chunks)

    if args.chunks_only:
        logger.info("\n✅ --chunks-only : arrêt après le chunking.")
        return

    # ── Étape 1b : Génération Gold Dataset ───────────────────────────────────
    if args.skip_gold or args.gold_file:
        logger.info(f"\n[SKIP] Gold generation — utilisation de : {gold_jsonl}")
        if not Path(gold_jsonl).exists():
            logger.error(f"Fichier Gold introuvable : {gold_jsonl}")
            sys.exit(1)
    else:
        run_gold_generation(chunks_file, gold_jsonl)

    # ── Étape 2 : Évaluation RAG ─────────────────────────────────────────────
    if args.skip_eval:
        logger.info("\n[SKIP] Évaluation RAG (--skip-eval)")
    else:
        run_rag_evaluation(chunks_file, gold_jsonl, eval_dir)

    # ── Résumé final ─────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    logger.info("\n" + "=" * 60)
    logger.info("  PIPELINE TERMINÉ")
    logger.info("=" * 60)
    logger.info(f"  Temps total : {elapsed:.0f}s ({elapsed/60:.1f} min)")
    logger.info(f"  Chunks      : {chunks_file}")
    logger.info(f"  Gold Dataset: {gold_jsonl}")
    logger.info(f"  Évaluation  : {eval_dir}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
