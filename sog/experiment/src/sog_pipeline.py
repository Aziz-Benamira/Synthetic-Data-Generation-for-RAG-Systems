"""
sog_pipeline.py
===============
Synthesize-on-Graph (SoG) — End-to-End Pipeline

Usage example:
    python sog_pipeline.py \
        --input_dir  data/raw_papers/ \
        --output_dir data/synthetic/ \
        --model      gpt-4o-mini \
        --depth      1 \
        --top_w      3 \
        --target_coverage 1.0 \
        --num_subsets 1

The generated QA pairs are saved as JSONL, compatible with your existing
HuggingFace upload scripts in the repo.

Drop into: generation/graph-based/sog_pipeline.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# --- local imports (files in the same directory) ---
from context_graph import ContextGraph
try:
    from context_graph import build_context_graph
    from cross_document_sampling import build_all_paths, secondary_sampling
    from generation_strategies import generate_from_subset
except ImportError:
    pass  # Not needed when only using make_embed_fn + ContextGraph.from_dict


# ---------------------------------------------------------------------------
# Embedding function (swap with your preferred model)
# ---------------------------------------------------------------------------

def make_embed_fn(model_name: str = "BAAI/bge-small-en-v1.5"):
    """
    Returns a callable: text -> np.ndarray.
    Requires: pip install sentence-transformers
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        st_model = SentenceTransformer(model_name)
        def embed(text: str):
            return st_model.encode(text, normalize_embeddings=True)
        return embed
    except ImportError:
        # Fallback: random embeddings (for testing without GPU / heavy deps)
        import numpy as np
        import hashlib
        print("[WARNING] sentence-transformers not installed. Using random embeddings.")
        def embed(text: str):
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(seed)
            return rng.random(384).astype(np.float32)
        return embed


# ---------------------------------------------------------------------------
# Document loaders
# ---------------------------------------------------------------------------

def load_documents_from_dir(input_dir: str, extensions: tuple = (".txt", ".md")) -> dict[str, str]:
    """Load plain-text or markdown documents from a directory."""
    docs = {}
    for path in Path(input_dir).rglob("*"):
        if path.suffix in extensions:
            doc_id = path.stem
            docs[doc_id] = path.read_text(encoding="utf-8", errors="ignore")
    return docs


def load_documents_from_pdf_dir(input_dir: str) -> dict[str, str]:
    """
    Load PDFs using pypdf (or pdfminer). 
    Requires: pip install pypdf
    Math in PDFs is typically rendered as unicode/text — the math-aware splitter
    will still detect LaTeX if the PDF was generated from LaTeX source.
    For scanned PDFs, pair with an OCR step first.
    """
    try:
        import pypdf
    except ImportError:
        raise ImportError("Install pypdf: pip install pypdf")

    docs = {}
    for path in Path(input_dir).rglob("*.pdf"):
        reader = pypdf.PdfReader(str(path))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        docs[path.stem] = text
    return docs


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_sog_pipeline(
    documents: dict[str, str],
    output_path: str,
    model: str = "gpt-4o-mini",
    depth: int = 1,
    top_w: int = 3,
    max_starts_per_entity: int = 5,
    target_coverage: float = 1.0,
    num_subsets: int = 1,
    embed_model: str = "BAAI/bge-small-en-v1.5",
    graph_cache_path: str | None = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Full SoG pipeline:
      1. Build context graph
      2. Traverse graph to collect cross-document paths
      3. Secondary sampling (balanced CoT + CC)
      4. Generate synthetic QA pairs
      5. Save to JSONL

    Args:
        documents:         {doc_id: raw_text}
        output_path:       path to write JSONL output
        model:             OpenAI-compatible model name
        depth:             BFS traversal depth (paper uses 1 or 2)
        top_w:             top-W similar paragraphs per hop
        target_coverage:   fraction of corpus chunks to cover
        num_subsets:       how many balanced subsets to generate from
        embed_model:       sentence-transformer model name for similarity
        graph_cache_path:  optional path to cache/reload the built graph as JSON

    Returns:
        list of QA sample dicts
    """
    client = OpenAI()  # reads OPENAI_API_KEY from environment

    # Step 1: Build (or load) context graph
    if graph_cache_path and Path(graph_cache_path).exists():
        if verbose:
            print(f"[pipeline] Loading cached graph from {graph_cache_path}")
        with open(graph_cache_path) as f:
            graph = ContextGraph.from_dict(json.load(f))
    else:
        if verbose:
            print(f"[pipeline] Building context graph over {len(documents)} documents...")
        graph = build_context_graph(documents, client, model=model, verbose=verbose)
        if graph_cache_path:
            Path(graph_cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(graph_cache_path, "w") as f:
                json.dump(graph.to_dict(), f, indent=2)
            if verbose:
                print(f"[pipeline] Graph cached to {graph_cache_path}")

    if verbose:
        print(f"[pipeline] Graph: {len(graph.nodes)} entities, {len(graph.edges)} edges")

    # Step 2: Build embedding index + traverse graph
    from cross_document_sampling import EmbeddingIndex
    embed_fn = make_embed_fn(embed_model)
    index = EmbeddingIndex(embed_fn)

    if verbose:
        print(f"[pipeline] Collecting cross-document paths (depth={depth}, top_w={top_w})...")
    all_paths = build_all_paths(
        graph=graph,
        index=index,
        depth=depth,
        top_w=top_w,
        max_starts_per_entity=max_starts_per_entity,
    )
    if verbose:
        print(f"[pipeline] Total paths collected: {len(all_paths)}")

    # Step 3: Secondary sampling
    if verbose:
        print("[pipeline] Running secondary balanced sampling...")
    subsets = secondary_sampling(
        all_paths=all_paths,
        graph=graph,
        target_coverage_rate=target_coverage,
        hop=depth,
    )
    if verbose:
        print(f"[pipeline] Produced {len(subsets)} balanced subsets")

    # Step 4: Generate synthetic QA
    all_samples: list[dict] = []
    for i, subset in enumerate(subsets[:num_subsets]):
        if verbose:
            print(f"[pipeline] Generating from subset {i+1}/{min(num_subsets, len(subsets))}...")
        samples = generate_from_subset(subset, client, model=model, verbose=verbose)
        all_samples.extend(samples)

    # Step 5: Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    if verbose:
        print(f"[pipeline] Saved {len(all_samples)} samples to {output_path}")

    return all_samples


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Synthesize-on-Graph (SoG) pipeline")
    parser.add_argument("--input_dir",  required=True, help="Directory with input documents")
    parser.add_argument("--output_dir", default="outputs", help="Output directory for JSONL files (default: outputs/)")
    parser.add_argument("--file_type",  default="txt", choices=["txt", "pdf"],
                        help="Input file type")
    parser.add_argument("--model",      default="gpt-4o-mini")
    parser.add_argument("--embed_model",default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--depth",      type=int, default=1,
                        help="BFS traversal depth (1 or 2 recommended)")
    parser.add_argument("--top_w",      type=int, default=3,
                        help="Top-W similar paragraphs per hop")
    parser.add_argument("--max_starts", type=int, default=5,
                        help="Max starting paragraphs per entity (S)")
    parser.add_argument("--coverage",   type=float, default=1.0,
                        help="Target corpus coverage rate (r)")
    parser.add_argument("--num_subsets",type=int, default=1,
                        help="Number of balanced subsets to generate from")
    parser.add_argument("--graph_cache",default=None,
                        help="Path to cache the built context graph as JSON")
    args = parser.parse_args()

    # Load docs
    if args.file_type == "pdf":
        docs = load_documents_from_pdf_dir(args.input_dir)
    else:
        docs = load_documents_from_dir(args.input_dir)

    if not docs:
        print(f"[ERROR] No documents found in {args.input_dir}")
        return

    output_path = str(Path(args.output_dir) / "sog_synthetic_qa.jsonl")

    run_sog_pipeline(
        documents=docs,
        output_path=output_path,
        model=args.model,
        depth=args.depth,
        top_w=args.top_w,
        max_starts_per_entity=args.max_starts,
        target_coverage=args.coverage,
        num_subsets=args.num_subsets,
        embed_model=args.embed_model,
        graph_cache_path=args.graph_cache,
    )


if __name__ == "__main__":
    main()
