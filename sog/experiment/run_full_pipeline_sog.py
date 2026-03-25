#!/usr/bin/env python3
"""
run_full_pipeline_sog.py
========================

End-to-end pipeline: QuestionGeneratorV3 → AnswerGeneratorV3SoG → CriticV4
                      with Synthesize-on-Graph (SoG) knowledge graph context.

Usage:
    python run_full_pipeline_sog.py [options]

Options:
    --chunks-path    Path to chunks JSON           [default: data/tipler_chunks.json]
    --graph-path     Path to pre-built graph JSON  [default: data/Tipler_Llewellyn_context_graph.json]
    --output-dir     Output directory              [default: output/full_pipeline_sog]
    --mode           SoG mode: combined|graph_only|disabled  [default: combined]
    --max-chunks     Limit number of chunks processed        [default: 10]
    --model-path     Path to DeepSeek R1 GGUF                [default: auto-detect]
    --n-ctx          LLM context window                      [default: 8192]
    --dry-run        Import check + SoGRetriever test, no LLM load
"""

import sys
import json
import logging
import argparse
import time
from pathlib import Path
from datetime import datetime

# ── sys.path: set before any local imports ────────────────────────────────────
_EXP_DIR    = Path(__file__).parent
_SOG_SRC    = str(_EXP_DIR / "src")
_AGENTIC_AI = "/home/ensta/ensta-ben-amira/projects/Agentic_AI"

sys.path = [p for p in sys.path if p not in ("", ".")]
if _SOG_SRC not in sys.path:
    sys.path.insert(0, _SOG_SRC)
if _AGENTIC_AI not in sys.path:
    sys.path.insert(1, _AGENTIC_AI)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Default paths
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_MODEL = (
    "/home/ensta/ensta-ben-amira/models/deepseek-r1-distill-qwen-32b/"
    "DeepSeek-R1-Distill-Qwen-32B-IQ3_M.gguf"
)
_DEFAULT_CHUNKS = str(_EXP_DIR / "data" / "tipler_chunks.json")
_DEFAULT_GRAPH  = str(_EXP_DIR / "data" / "Tipler_Llewellyn_context_graph.json")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Full SoG pipeline (Q→A→Critic + graph)")
    p.add_argument("--chunks-path",  default=_DEFAULT_CHUNKS)
    p.add_argument("--graph-path",   default=_DEFAULT_GRAPH)
    p.add_argument("--output-dir",   default=str(_EXP_DIR / "output" / "full_pipeline_sog"))
    p.add_argument("--mode",         default="combined",
                   choices=["combined", "graph_only", "disabled"])
    p.add_argument("--max-chunks",   type=int, default=10)
    p.add_argument("--model-path",   default=_DEFAULT_MODEL)
    p.add_argument("--n-ctx",        type=int, default=16384)
    p.add_argument("--multihop",     action="store_true", default=True,
                   help="Use BFS multi-hop retrieval (Section 3.2 SoG) — default True")
    p.add_argument("--no-multihop",  dest="multihop", action="store_false",
                   help="Disable multi-hop, use flat 1-hop expansion")
    p.add_argument("--sog-depth",    type=int, default=2,
                   help="BFS depth for multi-hop traversal (default 2)")
    p.add_argument("--sog-top-k",    type=int, default=3,
                   help="Number of seed paragraphs (default 3)")
    p.add_argument("--sog-top-w",    type=int, default=3,
                   help="Width at each BFS hop — neighbours kept (default 3)")
    p.add_argument("--dry-run",      action="store_true",
                   help="Import check + SoGRetriever test only — no LLM load")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Dry-run
# ─────────────────────────────────────────────────────────────────────────────

def dry_run(args):
    """Verify imports and SoGRetriever without loading the LLM."""
    logger.info("═" * 55)
    logger.info("  DRY-RUN — full pipeline import check")
    logger.info("═" * 55)

    # 1. Core pipeline imports
    logger.info("Vérification des imports pipeline …")
    from src.orchestrator.pipeline_v4 import PipelineV4, PipelineV4Config, GoldEntry
    logger.info("  src.orchestrator.pipeline_v4 ✓")
    from src.agents import QuestionGeneratorV3, AnswerGeneratorV3
    logger.info("  src.agents ✓")
    from src.critic_v4 import QuestionEvaluator
    from src.critic_v4.metrics import AnswerCompleteness, AnswerAnchoring
    logger.info("  src.critic_v4 ✓")
    from src.utils.scoped_memory import ScopedMemory
    logger.info("  src.utils.scoped_memory ✓")

    # 2. SoG imports
    from answer_generator_v3_sog import AnswerGeneratorV3SoG
    logger.info("  answer_generator_v3_sog ✓")
    from pipeline_v4_sog import PipelineV4SoG, PipelineV4SoGConfig, GoldEntrySoG
    logger.info("  pipeline_v4_sog ✓")
    from sog_retriever import SoGRetriever
    from context_graph import ContextGraph
    logger.info("  sog_retriever + context_graph ✓")

    # 3. Graph + embedder
    logger.info(f"\nChargement graphe: {Path(args.graph_path).name}")
    import json
    with open(args.graph_path) as f:
        g_data = json.load(f)
    graph = ContextGraph.from_dict(g_data)
    logger.info(f"  → {len(graph.nodes)} entités, {len(graph.edges)} arêtes ✓")

    logger.info("Chargement embedder: BAAI/bge-small-en-v1.5")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    def embed_fn(text: str):
        """Accept a single string, return a 1-D numpy float32 array."""
        return model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0].astype("float32")

    def batch_embed_fn(texts):
        """Batched encode — used at precompute time only."""
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    retriever = SoGRetriever(graph, embed_fn, precompute=True, batch_embed_fn=batch_embed_fn)
    logger.info("  SoGRetriever prêt ✓")

    # 4. Test retrieval
    test_q = "Qu'est-ce que l'énergie de masse au repos d'un proton ?"
    result = retriever.retrieve(test_q, top_k=5)
    logger.info(f"\nTest retrieval: «{test_q}»")
    logger.info(f"  Entités    : {result['entities'][:5]}")
    logger.info(f"  Relations  : {result['relations'][:3]}")
    logger.info(f"  Passages   : {len(result['passages'])}")
    logger.info(f"  Formatted  : {result['formatted'][:120]}…")

    # 5. Config check
    logger.info(f"\nConfig pipeline:")
    logger.info(f"  chunks_path  = {args.chunks_path}")
    logger.info(f"  graph_path   = {args.graph_path}")
    logger.info(f"  sog_mode     = {args.mode}")
    logger.info(f"  max_chunks   = {args.max_chunks}")
    logger.info(f"  output_dir   = {args.output_dir}")

    logger.info("\nDry run terminé ✓  (prêt pour sbatch)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.dry_run:
        dry_run(args)
        return

    # ── 1. Load LLM ──────────────────────────────────────────────────────────
    logger.info("═" * 55)
    logger.info("  FULL PIPELINE SoG — QuestionGen → AnswerGen → Critic")
    logger.info("═" * 55)

    logger.info(f"Chargement DeepSeek R1-32B …")
    from llama_cpp import Llama
    llm = Llama(
        model_path=args.model_path,
        n_gpu_layers=-1,
        n_ctx=args.n_ctx,
        verbose=False,
    )
    logger.info("  LLM chargé ✓")

    # Wrap to strip <think>...</think> from all responses.
    # DeepSeek R1 emits chain-of-thought in think blocks before the final JSON.
    # CriticV4 metrics call json.loads() directly and crash on <think> prefixes.
    from deepseek_wrapper import DeepSeekR1Wrapper
    llm = DeepSeekR1Wrapper(llm)
    logger.info("  DeepSeekR1Wrapper appliqué (think tags stripped) ✓")

    # ── 2. Build embed_fn ─────────────────────────────────────────────────────
    logger.info("Chargement embedder: BAAI/bge-small-en-v1.5 …")
    from sentence_transformers import SentenceTransformer
    _st_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    def embed_fn(text: str):
        """Accept a single string, return a 1-D numpy float32 array."""
        return _st_model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0].astype("float32")

    def batch_embed_fn(texts):
        """Batched encode — used at precompute time only."""
        return _st_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    logger.info("  Embedder prêt ✓")

    # ── 3. Pipeline config ────────────────────────────────────────────────────
    from pipeline_v4_sog import PipelineV4SoG, PipelineV4SoGConfig

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_jsonl = str(output_dir / f"gold_sog_{args.mode}_{ts}.jsonl")

    config = PipelineV4SoGConfig(
        chunks_path      = args.chunks_path,
        output_path      = output_jsonl,
        max_chunks       = args.max_chunks,
        min_chunk_length = 300,
        max_q_retries    = 3,
        max_a_retries    = 2,
        q_temperature    = 0.7,
        a_temperature    = 0.3,
        checkpoint_every = 5,
        sog_graph_path   = args.graph_path,
        sog_mode         = args.mode,
        sog_top_k        = args.sog_top_k,
        sog_depth        = args.sog_depth,
        sog_top_w        = args.sog_top_w,
        sog_multihop     = args.multihop,
    )

    logger.info(f"\nConfiguration:")
    logger.info(f"  chunks_path  = {args.chunks_path}")
    logger.info(f"  graph_path   = {args.graph_path}")
    logger.info(f"  sog_mode     = {args.mode}")
    logger.info(f"  multihop     = {args.multihop}  (depth={args.sog_depth}, top_k={args.sog_top_k}, top_w={args.sog_top_w})")
    logger.info(f"  max_chunks   = {args.max_chunks}")
    logger.info(f"  output       = {output_jsonl}")

    # ── 4. Run pipeline ───────────────────────────────────────────────────────
    t_start = time.time()
    pipeline = PipelineV4SoG(
        config=config, llm=llm, embed_fn=embed_fn, batch_embed_fn=batch_embed_fn
    )
    dataset  = pipeline.run()
    elapsed  = time.time() - t_start

    # ── 5. Summary ────────────────────────────────────────────────────────────
    logger.info("\n" + "═" * 55)
    logger.info("  RÉSULTATS")
    logger.info("═" * 55)
    logger.info(f"  Mode          : {args.mode}")
    logger.info(f"  Entrées gold  : {len(dataset)}")
    logger.info(f"  Durée totale  : {elapsed:.0f}s")
    logger.info(f"  Fichier       : {output_jsonl}")

    if dataset:
        scores = [e["global_score"] for e in dataset if e.get("global_score") is not None]
        if scores:
            logger.info(f"  Score moyen   : {sum(scores)/len(scores):.3f}")
            logger.info(f"  Score max     : {max(scores):.3f}")
            logger.info(f"  Score min     : {min(scores):.3f}")

        # Quick sample
        sample = dataset[0]
        logger.info(f"\n  Exemple [0]:")
        logger.info(f"    Q: {sample['question'][:120]}")
        logger.info(f"    A: {sample['answer'][:150]}…")
        if sample.get("graph_entities"):
            logger.info(f"    Graph entities: {sample['graph_entities'][:5]}")

    # Save human-readable summary
    summary_path = output_dir / f"summary_sog_{args.mode}_{ts}.json"
    summary = {
        "mode":       args.mode,
        "timestamp":  ts,
        "n_entries":  len(dataset),
        "elapsed_s":  round(elapsed, 1),
        "model":      Path(args.model_path).name,
        "scores": {
            "mean":  round(sum(e["global_score"] for e in dataset) / len(dataset), 3)
                     if dataset else None,
            "max":   max((e["global_score"] for e in dataset), default=None),
            "min":   min((e["global_score"] for e in dataset), default=None),
        },
        "entries": dataset,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"  Résumé JSON   : {summary_path}")
    logger.info("═" * 55)


if __name__ == "__main__":
    main()
