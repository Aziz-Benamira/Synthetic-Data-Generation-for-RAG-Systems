
# Synthesize-on-Graph (SoG) — Implementation

A Python implementation of the **Synthesize-on-Graph (SoG)** framework from the paper:
> *"Synthesize-on-Graph: Knowledgeable Synthetic Data Generation for Continued Pre-training of Large Language Models"* (Ma et al., 2025)

Implemented here as a **graph-based QA generation module** for the [Synthetic Data Generation for RAG Systems](https://github.com/Aziz-Benamira/Synthetic-Data-Generation-for-RAG-Systems) project.

---

## What SoG does differently from semantic chunking

Classic semantic chunking (used in the multi-agent branch) splits a document into flat, self-contained chunks and generates questions from each chunk in isolation. SoG instead:

1. Extracts **entities** from every paragraph across all input documents
2. Builds a **context graph** where entities that co-occur in the same paragraph are connected by an edge
3. **Traverses that graph** with a BFS walk, collecting multi-hop paths that cross document boundaries
4. Generates QA pairs from those cross-document paths — producing questions that require integrating information from multiple paragraphs/documents to answer

This directly addresses the core weakness of flat chunking for multi-hop reasoning tasks.

---

## File overview

```
sog/
├── context_graph.py            # Step 1 & 2 — entity extraction + graph construction
├── cross_document_sampling.py  # Step 3   — BFS traversal + balanced sampling
├── generation_strategies.py    # Step 4   — CoT and CC QA generation
├── sog_pipeline.py             # Full pipeline CLI (batch, multi-doc)
├── run_on_pdf.py               # Single-PDF test runner (start here)
└── test_sog.py                 # Smoke tests (no API key required)
```

---

## File details

### `context_graph.py` — Context Graph Construction
*Implements paper Section 3.1*

- **Math-aware paragraph splitting**: display math blocks (`$$...$$`, `\[...\]`, `\begin{equation}`) are extracted before LLM processing and replaced with `__MATH_0__` placeholder tokens. This treats equations as atomic entities rather than letting them be split across boundaries.
- **Entity extraction via LLM**: prompts `gpt-4o-mini` to extract up to 10 key entities per paragraph, including math placeholders as graph nodes.
- **`ContextGraph` class**: stores nodes (entities), edges (co-occurrence pairs), and the entity→paragraph mapping `M`. Supports JSON serialisation for caching.
- **`build_context_graph()`**: top-level function that runs the full extraction loop over a `{doc_id: raw_text}` dict.

### `cross_document_sampling.py` — Cross-Document Sampling
*Implements paper Section 3.2 + Algorithms 1 & 2*

- **`EmbeddingIndex`**: thin wrapper around any sentence-transformer embedding function, with per-paragraph caching to avoid redundant encode calls.
- **`build_all_paths()`**: BFS traversal from every entity as root. At each hop, selects the top-W most semantically similar neighbour paragraphs (dot-product similarity). Respects the `max_starts_per_entity` cap (paper's hyperparameter *S*).
- **`balanced_sampling()`**: implements the secondary sampling algorithm — sorts paths by ascending entity utilisation count so least-seen entities are prioritised. When a coverage gap remains after `standard_length` paths, triggers **Contrastive Clarifying (CC)** pair construction for the sparsest entities.
- **`secondary_sampling()`**: iteratively allocates paths into subsets (each subset = one generation batch), returning a list of `SampledPathSet` objects.

### `generation_strategies.py` — Generation Strategies
*Implements paper Section 3.3, Figures 5 & 6*

- **CoT generation**: prompts the LLM to build a causal narrative across path fragments (Initiation → Development → Turning Point → Conclusion), then formulate a question whose answer requires understanding the full chain. Output is a step-by-step CoT answer.
- **CC generation**: prompts the LLM to produce a contrastive comparative analysis of two sparse-entity fragments, highlighting differences and similarities without forcing artificial connections.
- **`generate_from_subset()`**: batch-processes a `SampledPathSet`, runs both CoT and CC generation, and returns a list of dicts ready for JSONL serialisation or HuggingFace upload.

### `sog_pipeline.py` — Full Pipeline CLI
*End-to-end runner for batch / multi-document use*

Ties all three modules together with a command-line interface. Handles document loading from a directory (`.txt`, `.md`, or `.pdf`), graph caching, and output writing.

```bash
python sog_pipeline.py \
    --input_dir  data/papers/ \
    --output_dir data/synthetic/ \
    --file_type  pdf \
    --depth      1 \
    --max_paras  100
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--depth` | `1` | BFS hop depth. `1` = single-hop, `2` = multi-hop (richer but ~2-3x more LLM calls) |
| `--top_w` | `3` | Top-W most similar neighbour paragraphs per hop |
| `--max_starts` | `5` | Max starting paragraphs per entity (paper's *S*) |
| `--coverage` | `1.0` | Target corpus coverage rate (paper's *r*) |
| `--num_subsets` | `1` | How many balanced subsets to generate from |
| `--graph_cache` | `None` | Path to cache the built graph as JSON (skips rebuild on re-run) |

### `run_on_pdf.py` — Single-PDF Test Runner
*The best place to start*

A self-contained script that runs the full pipeline on a single PDF with sensible defaults. Includes a dependency check, helpful error messages, and a capped `--max_paras` flag to keep cost low during testing.

```bash
export OPENAI_API_KEY=sk-...
python run_on_pdf.py --pdf input.pdf
```

Produces two output files:
- `input_context_graph.json` — cached entity graph (delete to force rebuild)
- `input_sog_qa.jsonl` — generated QA pairs, one JSON object per line

### `test_sog.py` — Smoke Tests
*No API key or GPU required*

Four offline unit tests that validate the pipeline without any LLM or embedding calls:
1. Math-aware paragraph splitting and equation extraction
2. Graph construction with injected fake entities
3. BFS path traversal with random-vector embeddings
4. Graph JSON serialisation round-trip

Run with: `python test_sog.py`

---

## Quick start

```bash
# 1. Install dependencies
pip install openai sentence-transformers pypdf numpy

# 2. Set API key
export OPENAI_API_KEY=sk-...

# 3. Smoke test (no API key needed)
python test_sog.py

# 4. Run on your PDF
python run_on_pdf.py --pdf input.pdf --max_paras 50
```

---

## Output format

Each line of the output JSONL file is a QA sample:

```json
{
  "source":   "cot",
  "entities": ["attention mechanism", "multi-head attention"],
  "path_ids": ["paper_attention_p0", "paper_transformer_p2"],
  "question": "How does multi-head attention extend the scaled dot-product formulation?",
  "answer":   "Step 1: ... Step 2: ... Final answer: ...",
  "context":  "The narrative linking both fragments..."
}
```

`source` is either `"cot"` (Chain-of-Thought, cross-document path) or `"cc"` (Contrastive Clarifying, sparse-entity pair).

---

## How this fits with the multi-agent branch

SoG and the existing multi-agent pipeline are **complementary, not competing**:

| | the existing multi-agent branch | SoG (this module) |
|---|---|---|
| **Strength** | Sophisticated QA refinement (Reflexion loop, Constitutional AI critique, RAGAS evaluation) | Sophisticated information sampling (cross-document entity graph, multi-hop paths) |
| **Weakness** | Flat semantic chunking as ingestion | Naive CoT/CC prompts with no iterative refinement |
| **Question types** | High-quality single-document QA | Multi-hop cross-document QA |

The natural integration is to use **SoG's context graph paths as the context input** to the existing agent loop, replacing the semantic chunking + ChromaDB retrieval step while keeping all downstream agents intact. This gives you both richer context sampling *and* high-quality iterative refinement — and produces a benchmark dataset that spans both easy single-document questions and hard multi-hop ones.

---

## Dependencies

| Package | Purpose |
|---|---|
| `openai` | LLM calls for entity extraction and QA generation |
| `sentence-transformers` | Paragraph embeddings for similarity-guided graph traversal |
| `pypdf` | PDF text extraction |
| `numpy` | Cosine similarity computation |
