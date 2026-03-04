"""
run_on_pdf.py
=============
Minimal end-to-end test of the SoG pipeline on a single PDF.

Setup (once):
    pip install openai sentence-transformers pypdf numpy

Provider options (choose one — all work with the same code):

  1. Ollama  [FREE, local, no account needed]
       a) Install Ollama:  https://ollama.com/download
       b) Pull a model:    ollama pull llama3.2
       c) Run:             python run_on_pdf.py --pdf input.pdf --provider ollama --model llama3.2

  2. Groq    [FREE cloud tier, ~14 k req/day, fast]
       a) Sign up at https://console.groq.com  (no credit card)
       b) Create a free API key
       c) Set variable:    set GROQ_API_KEY=gsk_...
       d) Run:             python run_on_pdf.py --pdf input.pdf --provider groq --model llama-3.1-8b-instant

  3. OpenAI  [requires paid credits]
       set OPENAI_API_KEY=sk-...
       python run_on_pdf.py --pdf input.pdf --provider openai --model gpt-4o-mini

Outputs (written to the outputs/ folder next to this script):
    outputs/<stem>_context_graph.json   — the built entity graph (cached for re-use)
    outputs/<stem>_sog_qa.jsonl         — generated QA pairs, one JSON object per line
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ── 1. Dependency check ────────────────────────────────────────────────────
REQUIRED = ["openai", "sentence_transformers", "pypdf", "numpy"]
missing = []
for pkg in REQUIRED:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg.replace("_", "-"))

if missing:
    print("Missing dependencies. Run:\n")
    print(f"    pip install {' '.join(missing)}\n")
    sys.exit(1)

# ── 2. Imports ─────────────────────────────────────────────────────────────
import pypdf
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Local SoG modules (must be in the same directory)
from context_graph import build_context_graph, ContextGraph
from cross_document_sampling import (
    build_all_paths, secondary_sampling, EmbeddingIndex
)
from generation_strategies import generate_from_subset


# ── 3. PDF loader ──────────────────────────────────────────────────────────

def load_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    reader = pypdf.PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    full_text = "\n\n".join(pages)
    print(f"  Loaded {len(reader.pages)} pages, "
          f"{len(full_text):,} characters from '{pdf_path}'")
    return full_text


# ── 4. Main runner ─────────────────────────────────────────────────────────

# Default model names per provider
_PROVIDER_DEFAULTS = {
    "ollama": "llama3.2",
    "groq":   "llama-3.1-8b-instant",
    "openai": "gpt-4o-mini",
}

# OpenAI-compatible base URLs
_PROVIDER_URLS = {
    "ollama": "http://localhost:11434/v1",
    "groq":   "https://api.groq.com/openai/v1",
    "openai": None,   # use the default OpenAI URL
}


def run(
    pdf_path: str,
    provider: str = "ollama",
    model: str | None = None,
    embed_model: str = "BAAI/bge-small-en-v1.5",
    depth: int = 1,       # 1-hop paths  (use 2 for richer multi-hop, costs more)
    top_w: int = 3,       # top-W similar paragraphs per hop
    max_starts: int = 5,  # max starting paragraphs per entity  (paper's S)
    coverage: float = 1.0,
    num_subsets: int = 1, # how many balanced subsets to generate from
    max_paras: int = 50,  # cap paragraphs to keep cost low during testing
                          # set to None to process the full document
):
    stem = Path(pdf_path).stem
    outputs_dir = Path(__file__).parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    graph_cache = str(outputs_dir / f"{stem}_context_graph.json")
    output_path = str(outputs_dir / f"{stem}_sog_qa.jsonl")

    provider = provider.lower()
    if provider not in _PROVIDER_DEFAULTS:
        print(f"ERROR: Unknown provider '{provider}'. Choose from: {list(_PROVIDER_DEFAULTS)}")
        sys.exit(1)

    # Resolve model name
    if model is None:
        model = _PROVIDER_DEFAULTS[provider]

    # ── Validate provider credentials / connectivity ───────────────────────
    if provider == "ollama":
        print(f"[provider] Ollama (local)  model={model}")
        print("  Make sure Ollama is running: ollama serve")
        print(f"  Make sure the model is pulled: ollama pull {model}")
        client = OpenAI(
            base_url=_PROVIDER_URLS["ollama"],
            api_key="ollama",           # required by the client lib, ignored by Ollama
        )
    elif provider == "groq":
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("ERROR: GROQ_API_KEY environment variable not set.")
            print("  Sign up free at https://console.groq.com and create an API key, then:")
            print("  set GROQ_API_KEY=gsk_...")
            sys.exit(1)
        print(f"[provider] Groq (cloud free tier)  model={model}")
        client = OpenAI(
            base_url=_PROVIDER_URLS["groq"],
            api_key=api_key,
        )
    else:  # openai
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY environment variable not set.")
            print("  set OPENAI_API_KEY=sk-...")
            sys.exit(1)
        print(f"[provider] OpenAI  model={model}")
        client = OpenAI(api_key=api_key)

    # ── Load PDF ───────────────────────────────────────────────────────────
    print("\n[1/5] Loading PDF...")
    raw_text = load_pdf(pdf_path)
    documents = {"input_doc": raw_text}

    # ── Build (or reload) context graph ───────────────────────────────────
    _rebuild = True
    if Path(graph_cache).exists():
        print(f"\n[2/5] Loading cached context graph from '{graph_cache}'...")
        with open(graph_cache) as f:
            graph = ContextGraph.from_dict(json.load(f))
        print(f"  {len(graph.nodes)} entities, {len(graph.edges)} edges")
        if len(graph.nodes) == 0:
            print(f"  WARNING: Cached graph is empty (likely from a previously failed run).")
            print(f"  Deleting '{graph_cache}' and rebuilding...")
            Path(graph_cache).unlink()
        else:
            _rebuild = False

    if _rebuild:
        print(f"\n[2/5] Building context graph (max {max_paras} paragraphs)...")
        print("  This calls the LLM once per paragraph — may take a minute.")
        graph = build_context_graph(
            documents,
            client,
            model=model,
            max_paras_per_doc=max_paras,
            verbose=True,
        )
        with open(graph_cache, "w") as f:
            json.dump(graph.to_dict(), f, indent=2)
        print(f"  Graph saved to '{graph_cache}'")
        print(f"  {len(graph.nodes)} entities, {len(graph.edges)} edges")

    if len(graph.nodes) == 0:
        print("\nERROR: No entities extracted. Check your PDF has extractable text.")
        sys.exit(1)

    # ── Build embedding index ──────────────────────────────────────────────
    print(f"\n[3/5] Loading sentence-transformer '{embed_model}'...")
    st_model = SentenceTransformer(embed_model)
    embed_fn = lambda text: st_model.encode(text, normalize_embeddings=True)
    index = EmbeddingIndex(embed_fn)

    # ── Traverse graph for cross-document paths ────────────────────────────
    print(f"\n[4/5] Traversing context graph (depth={depth}, top_w={top_w})...")
    all_paths = build_all_paths(
        graph=graph,
        index=index,
        depth=depth,
        top_w=top_w,
        max_starts_per_entity=max_starts,
    )
    print(f"  {len(all_paths)} paths collected")

    if not all_paths:
        print("\nWARNING: No paths found. The graph may have too few edges.")
        print("  Try increasing max_paras or check that the PDF has rich entity overlap.")
        sys.exit(1)

    # ── Secondary sampling ─────────────────────────────────────────────────
    subsets = secondary_sampling(
        all_paths=all_paths,
        graph=graph,
        target_coverage_rate=coverage,
        hop=depth,
    )
    total_cot = sum(len(s.cot) for s in subsets[:num_subsets])
    total_cc  = sum(len(s.cc)  for s in subsets[:num_subsets])
    print(f"  {len(subsets)} subsets → using {num_subsets}, "
          f"~{total_cot} CoT paths + {total_cc} CC pairs to generate from")

    # ── Generate QA pairs ──────────────────────────────────────────────────
    print(f"\n[5/5] Generating synthetic QA pairs...")
    print("  Each path = 1 LLM call. This is the most expensive step.")
    all_samples = []
    for i, subset in enumerate(subsets[:num_subsets]):
        print(f"  Subset {i+1}/{num_subsets}...")
        samples = generate_from_subset(subset, client, model=model, verbose=True)
        all_samples.extend(samples)

    # ── Save output ────────────────────────────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n✓ Done! {len(all_samples)} QA pairs written to '{output_path}'")
    print("\nSample output:")
    print("-" * 60)
    if all_samples:
        sample = all_samples[0]
        print(f"  Source  : {sample['source']}")
        print(f"  Entities: {sample['entities']}")
        print(f"  Question: {sample['question'][:120]}...")
        print(f"  Answer  : {sample['answer'][:120]}...")
    print("-" * 60)
    print(f"\nFull results: {output_path}")
    print(f"Graph cache:  {graph_cache}  (delete to force rebuild)")


# ── 5. CLI ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SoG synthetic QA generation on a single PDF"
    )
    parser.add_argument("--pdf",         required=True,  help="Path to input PDF")
    parser.add_argument("--provider",    default="ollama",
                        choices=["ollama", "groq", "openai"],
                        help="LLM provider. 'ollama' = free local, 'groq' = free cloud, 'openai' = paid")
    parser.add_argument("--model",       default=None,
                        help="Model name (default: llama3.2 for ollama, llama-3.1-8b-instant for groq, gpt-4o-mini for openai)")
    parser.add_argument("--embed_model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--depth",       type=int,   default=1,
                        help="BFS depth: 1=single-hop, 2=multi-hop")
    parser.add_argument("--top_w",       type=int,   default=3,
                        help="Top-W similar neighbours per hop")
    parser.add_argument("--max_starts",  type=int,   default=5,
                        help="Max starting paragraphs per entity")
    parser.add_argument("--coverage",    type=float, default=1.0)
    parser.add_argument("--num_subsets", type=int,   default=1)
    parser.add_argument("--max_paras",   type=int,   default=50,
                        help="Max paragraphs to process")
    args = parser.parse_args()

    run(
        pdf_path=args.pdf,
        provider=args.provider,
        model=args.model,
        embed_model=args.embed_model,
        depth=args.depth,
        top_w=args.top_w,
        max_starts=args.max_starts,
        coverage=args.coverage,
        num_subsets=args.num_subsets,
        max_paras=args.max_paras,
    )
