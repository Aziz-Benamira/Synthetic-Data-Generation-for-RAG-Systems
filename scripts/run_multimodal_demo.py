"""
run_multimodal_demo.py — Multimodal pipeline demo on academic PDFs

Extracts figures from a PDF, describes them with Qwen2-VL-7B-Instruct,
and saves the resulting figure chunks as JSON/JSONL ready for RAG ingestion.

Usage (GPU required):
    source ~/envs/agentic_ai/bin/activate
    cd "/home/ensta/ensta-ghozzi/AI RAG"

    python3 scripts/run_multimodal_demo.py \
        --pdf data/pdfs/Attention_Is_All_You_Need.pdf \
        --output output/multimodal_attention

    # Test extraction only (no GPU needed)
    python3 scripts/run_multimodal_demo.py \
        --pdf data/pdfs/Attention_Is_All_You_Need.pdf \
        --output output/multimodal_attention \
        --extract-only
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Add project root to Python path
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multimodal pipeline demo: PDF → figures → VL descriptions → chunks"
    )
    parser.add_argument(
        "--pdf",
        default="data/pdfs/Attention_Is_All_You_Need.pdf",
        help="Path to input PDF (default: Attention_Is_All_You_Need.pdf)",
    )
    parser.add_argument(
        "--output",
        default="output/multimodal_attention",
        help="Output directory (default: output/multimodal_attention)",
    )
    parser.add_argument(
        "--vl-model",
        default="/home/ensta/data/Qwen2-VL-7B-Instruct",
        help="Path to Qwen2-VL model directory",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=100,
        help="Minimum image width in pixels to extract (default: 100)",
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=100,
        help="Minimum image height in pixels to extract (default: 100)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max tokens for VL description (default: 512)",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract images (skip VL captioning, no GPU needed)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N figures (default: all)",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        logger.error("PDF not found: %s", pdf_path)
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # ── Step 1: Extract images ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 1: Extracting figures from %s", pdf_path.name)
    logger.info("=" * 60)

    from src.multimodal import ImageExtractor

    extractor = ImageExtractor(
        str(pdf_path),
        min_width=args.min_width,
        min_height=args.min_height,
    )
    t0 = time.time()
    figures = extractor.extract_all()
    logger.info("Extracted %d figures in %.1fs", len(figures), time.time() - t0)

    # Apply limit if set
    if args.limit and len(figures) > args.limit:
        logger.info("--limit %d: keeping first %d figures out of %d", args.limit, args.limit, len(figures))
        figures = figures[: args.limit]

    if not figures:
        logger.warning("No figures found. Exiting.")
        sys.exit(0)

    # Save raw images to disk
    for fig in figures:
        img_path = figures_dir / f"{fig.figure_id}.png"
        img_path.write_bytes(fig.image_bytes)
        logger.info("  Saved %s (%dx%d)  caption=%r", img_path.name, fig.width, fig.height, fig.caption[:60])

    # Print extraction summary
    logger.info("\n--- Extraction Summary ---")
    for fig in figures:
        logger.info(
            "  [%s] page=%d  %dx%d  section=%r  caption=%r",
            fig.figure_id,
            fig.page + 1,
            fig.width,
            fig.height,
            fig.section[:50] if fig.section else "",
            fig.caption[:60] if fig.caption else "",
        )

    if args.extract_only:
        logger.info("\n--extract-only flag set. Skipping VL captioning.")
        _save_chunks(figures, out_dir, step="extraction_only")
        return

    # ── Step 2: VL description ──────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Generating VL descriptions with Qwen2-VL-7B")
    logger.info("=" * 60)

    from src.multimodal import VLCaptioner

    captioner = VLCaptioner(
        model_path=args.vl_model,
        max_new_tokens=args.max_new_tokens,
    )
    captioner.load()

    t1 = time.time()
    for i, fig in enumerate(figures):
        logger.info("[%d/%d] Describing %s …", i + 1, len(figures), fig.figure_id)
        t_fig = time.time()
        desc = captioner.describe(fig)
        logger.info("  Done in %.1fs | %d chars", time.time() - t_fig, len(desc))
        logger.info("  Preview: %s", desc[:120].replace("\n", " "))

    captioner.unload()
    logger.info(
        "\nAll descriptions generated in %.1fs  (%.1fs/figure)",
        time.time() - t1,
        (time.time() - t1) / max(len(figures), 1),
    )

    # ── Step 3: Save figure chunks ─────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Saving figure chunks")
    logger.info("=" * 60)

    _save_chunks(figures, out_dir, step="full")

    # ── Final report ────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("MULTIMODAL PIPELINE REPORT")
    logger.info("=" * 60)
    logger.info("  PDF         : %s", pdf_path.name)
    logger.info("  Figures     : %d", len(figures))
    logger.info("  Output dir  : %s", out_dir)
    logger.info("  Images      : %s/figures/", out_dir)
    logger.info("  Chunks      : %s/figure_chunks.jsonl", out_dir)
    logger.info("")
    logger.info("  Figure details:")
    for fig in figures:
        vl = fig.metadata.get("vl_description", "")
        logger.info(
            "    [%s] page=%d  caption=%r",
            fig.figure_id, fig.page + 1, (fig.caption or "(none)")[:60],
        )
        if vl:
            logger.info("       VL preview: %s", vl[:120].replace("\n", " "))
    logger.info("=" * 60)


def _save_chunks(figures, out_dir: Path, step: str = "full"):
    """Save figure chunks as JSONL + readable JSON."""
    chunks = [fig.to_chunk_dict() for fig in figures]

    # Remove image_bytes from output (too large)
    for chunk in chunks:
        chunk.get("metadata", {}).pop("image_bytes", None)

    jsonl_path = out_dir / "figure_chunks.jsonl"
    json_path = out_dir / "figure_chunks.json"

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d figure chunks → %s", len(chunks), jsonl_path)
    logger.info("Saved readable JSON  → %s", json_path)


if __name__ == "__main__":
    main()
