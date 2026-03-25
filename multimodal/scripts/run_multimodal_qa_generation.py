"""
run_multimodal_qa_generation.py — Gold Dataset generation from figure chunks

Runs PipelineV4 on multimodal figure chunks (produced by the VL captioning step)
to generate validated QA pairs in the same Gold Dataset format as text chunks.

Each figure chunk's 'content' = VL description + caption is treated as the
source text by DeepSeek R1, which generates questions about what the figure
shows (architecture, mechanism, relationships, etc.).

Difficulty grading (Bloom taxonomy) is enabled by default.

Usage (GPU required — run via SLURM):
    sbatch run_multimodal_qa_generation.sbatch

Or directly:
    source ~/envs/agentic_ai/bin/activate
    cd "/home/ensta/ensta-ghozzi/AI RAG"
    python3 scripts/run_multimodal_qa_generation.py \
        --chunks output/multimodal_attention/figure_chunks.jsonl \
        --output output/multimodal_attention/gold_dataset_figures.jsonl
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Add project root to Python path
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

MODEL_PATH = (
    "/home/ensta/ensta-ghozzi/models/deepseek-r1-distill-qwen-32b/"
    "DeepSeek-R1-Distill-Qwen-32B-IQ3_M.gguf"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Gold QA Dataset from multimodal figure chunks"
    )
    parser.add_argument(
        "--chunks",
        default="output/multimodal_attention/figure_chunks.json",
        help="Path to figure chunks JSON (from run_multimodal_demo.py)",
    )
    parser.add_argument(
        "--output",
        default="output/multimodal_attention/gold_dataset_figures.jsonl",
        help="Output Gold Dataset path",
    )
    parser.add_argument(
        "--model",
        default=MODEL_PATH,
        help="Path to DeepSeek R1 GGUF model",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=4096,
        help="LLM context size (default: 4096)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Process only first N chunks (for testing)",
    )
    parser.add_argument(
        "--no-difficulty",
        action="store_true",
        help="Disable Bloom difficulty grading (faster)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("MULTIMODAL QA GENERATION — PipelineV4 on figure chunks")
    logger.info("=" * 60)
    logger.info("  Chunks : %s", args.chunks)
    logger.info("  Output : %s", args.output)
    logger.info("  Model  : %s", args.model)

    # ── Load LLM ──────────────────────────────────────────────────────────────
    logger.info("\nLoading DeepSeek R1-32B …")
    t0 = time.time()
    from src.llm import LLMManager

    llm_manager = LLMManager.from_direct_llamacpp(
        model_path=args.model,
        n_gpu_layers=-1,
        n_ctx=args.n_ctx,
    )
    llm = llm_manager.provider.llm
    logger.info("Model loaded in %.1fs", time.time() - t0)

    # ── Run PipelineV4 ────────────────────────────────────────────────────────
    from src.orchestrator.pipeline_v4 import PipelineV4, PipelineV4Config

    config = PipelineV4Config(
        chunks_path=args.chunks,
        output_path=args.output,
        max_chunks=args.max_chunks,
        min_chunk_length=200,          # lower threshold: VL descriptions can be shorter
        semantic_types=["figure"],     # only process figure chunks
        max_q_retries=3,
        max_a_retries=2,
        checkpoint_every=5,
        enable_difficulty_grading=not args.no_difficulty,
    )

    pipeline = PipelineV4(config=config, llm=llm)
    dataset = pipeline.run()

    # ── Save readable JSON ────────────────────────────────────────────────────
    json_out = Path(args.output).with_suffix(".json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info("  Gold entries : %d", len(dataset))
    logger.info("  JSONL        : %s", args.output)
    logger.info("  JSON         : %s", json_out)

    if dataset:
        avg_score = sum(e["global_score"] for e in dataset) / len(dataset)
        logger.info("  Avg global score : %.3f", avg_score)

        logger.info("\n  QA pairs generated:")
        for e in dataset:
            logger.info(
                "    [%s] score=%.2f  diff=%s  Q: %s",
                e["chunk_id"],
                e["global_score"],
                e.get("difficulty_label", "n/a"),
                e["question"][:80],
            )

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
