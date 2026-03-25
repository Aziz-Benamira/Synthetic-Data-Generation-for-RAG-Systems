"""
evaluate_sog.py
===============
Evaluates a Synthesize-on-Graph (SoG) pipeline output using:

Retrieval metrics (always computed — requires the context graph JSON):
  - Hit@1  — at least 1 gold paragraph found in top-1 retrieved results
  - Hit@5  — at least 1 gold paragraph found in top-5 retrieved results
  - MRR    — Mean Reciprocal Rank of the first relevant paragraph

Generation metrics (add --generate flag):
  - ROUGE-L F1   — lexical overlap between generated and reference answer
  - BERTScore F1 — semantic similarity  (pip install bert-score)
  - LLM Judge    — LLM-rated answer quality on a 1–5 scale

Any OpenAI-compatible provider is supported via --base_url / --api_key.
Convenience shortcuts:
  --groq          uses Groq (https://api.groq.com/openai/v1) + GROQ_API_KEY env var
  --base_url URL  any custom endpoint (Ollama, Together, OpenRouter, …)
  --api_key KEY   explicit API key (otherwise read from env)

Usage
-----
# Retrieval only (fast, no API key needed):
    python evaluate_sog.py \\
        --graph outputs/Attention_Is_All_You_Need_context_graph.json \\
        --qa    outputs/Attention_Is_All_You_Need_sog_qa.jsonl

# Groq (llama-3.3-70b-versatile is fast and free-tier friendly):
    python evaluate_sog.py \\
        --graph    outputs/Attention_Is_All_You_Need_context_graph.json \\
        --qa       outputs/Attention_Is_All_You_Need_sog_qa.jsonl \\
        --generate --groq --model llama-3.3-70b-versatile

# OpenAI:
    python evaluate_sog.py \\
        --graph    outputs/Attention_Is_All_You_Need_context_graph.json \\
        --qa       outputs/Attention_Is_All_You_Need_sog_qa.jsonl \\
        --generate --model gpt-4o-mini

# Ollama (local):
    python evaluate_sog.py \\
        --graph    outputs/Attention_Is_All_You_Need_context_graph.json \\
        --qa       outputs/Attention_Is_All_You_Need_sog_qa.jsonl \\
        --generate --base_url http://localhost:11434/v1 --api_key ollama --model llama3

# Evaluate all JSONL/graph pairs in the outputs/ directory at once:
    python evaluate_sog.py --output_dir outputs/ [--generate] [--groq]

Optional pip packages:
    pip install rouge-score    # ROUGE-L
    pip install bert-score     # BERTScore
    pip install sentence-transformers  # dense retrieval
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Local path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from context_graph import ContextGraph, Paragraph
from sog_pipeline import make_embed_fn


# ---------------------------------------------------------------------------
# Optional heavy dependencies — graceful degradation
# ---------------------------------------------------------------------------

try:
    from rouge_score import rouge_scorer as _rs_mod  # type: ignore
    _rouge = _rs_mod.RougeScorer(["rougeL"], use_stemmer=True)
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False

try:
    import bert_score as _bs_mod  # type: ignore
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_qa_pairs(path: str) -> list[dict]:
    """Load QA pairs from a JSONL file, skipping blank lines."""
    pairs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def _parse_reference_answer(answer_field) -> str:
    """
    Extract plain-text answer from the JSONL 'answer' field.
    The field can be:
      - a plain string
      - a JSON-encoded string with 'cot_answer' or 'answer' keys
      - an already-parsed dict
    """
    if isinstance(answer_field, dict):
        return answer_field.get("cot_answer") or answer_field.get("answer") or ""
    if isinstance(answer_field, str):
        stripped = re.sub(r"```(?:json)?|```", "", answer_field).strip()
        try:
            d = json.loads(stripped)
            if isinstance(d, dict):
                return d.get("cot_answer") or d.get("answer") or answer_field
        except Exception:
            pass
    return str(answer_field)


def load_corpus(graph_json_path: str) -> tuple[ContextGraph, list[Paragraph]]:
    """
    Load a context graph JSON and return:
      - the ContextGraph object
      - a deduplicated, stable-order list of all paragraphs
    """
    with open(graph_json_path, encoding="utf-8") as fh:
        data = json.load(fh)
    graph = ContextGraph.from_dict(data)

    seen: set[str] = set()
    paragraphs: list[Paragraph] = []
    for paras in graph.mapping.values():
        for p in paras:
            if p.para_id not in seen:
                seen.add(p.para_id)
                paragraphs.append(p)
    return graph, paragraphs


# ---------------------------------------------------------------------------
# Dense retrieval index
# ---------------------------------------------------------------------------

class RetrievalIndex:
    """
    Normalised dense embedding index for cosine-similarity retrieval.
    Paragraphs are embedded once at build() time; queries are cheap.
    """

    def __init__(self, paragraphs: list[Paragraph], embed_fn) -> None:
        self._paragraphs = paragraphs
        self._embed_fn = embed_fn
        self._ids: list[str] = [p.para_id for p in paragraphs]
        self._matrix: np.ndarray | None = None  # shape (N, D), L2-normalised

    def build(self, verbose: bool = True) -> None:
        n = len(self._paragraphs)
        if verbose:
            print(f"[index] Embedding {n} paragraphs…")
        vecs = []
        for i, para in enumerate(self._paragraphs):
            vecs.append(self._embed_fn(para.text))
            if verbose and (i + 1) % 50 == 0:
                print(f"  {i + 1}/{n}")
        mat = np.stack(vecs).astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._matrix = mat / norms

    def retrieve(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Return up to k (para_id, cosine_score) pairs, sorted descending."""
        q = self._embed_fn(query).astype(np.float32)
        norm = float(np.linalg.norm(q))
        if norm > 0:
            q = q / norm
        scores: np.ndarray = self._matrix @ q
        k_clamped = min(k, len(self._ids))
        top = np.argpartition(-scores, k_clamped - 1)[:k_clamped]
        top = top[np.argsort(-scores[top])]
        return [(self._ids[i], float(scores[i])) for i in top]


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(
    qa_pairs: list[dict],
    index: RetrievalIndex,
    k_values: tuple[int, ...] = (1, 5),
) -> dict[str, float]:
    """
    Compute Hit@K and MRR using 'path_ids' as ground-truth relevance.

    Hit@K  = fraction of questions where ≥1 gold para is in the top-K results.
    MRR    = mean of 1/rank(first gold para) across all questions.
    """
    max_k = max(k_values)
    hit: dict[int, int] = {k: 0 for k in k_values}
    rrs: list[float] = []
    skipped = 0

    for qa in qa_pairs:
        gold: set[str] = set(qa.get("path_ids") or [])
        if not gold:
            skipped += 1
            continue

        retrieved = [pid for pid, _ in index.retrieve(qa["question"], k=max_k)]

        for k in k_values:
            if gold & set(retrieved[:k]):
                hit[k] += 1

        rr = 0.0
        for rank, pid in enumerate(retrieved, start=1):
            if pid in gold:
                rr = 1.0 / rank
                break
        rrs.append(rr)

    n = len(rrs)

    if skipped:
        print(f"[WARNING] {skipped} QA pair(s) had no path_ids and were skipped.")

    if n == 0:
        print(
            "[WARNING] No QA pairs had non-empty path_ids — retrieval metrics "
            "cannot be computed.\n"
            "  → Make sure you are pairing the correct graph JSON with its JSONL."
        )
        nan = float("nan")
        return {f"Hit@{k}": nan for k in k_values} | {"MRR": nan, "n_evaluated": 0.0}

    results: dict[str, float] = {f"Hit@{k}": hit[k] / n for k in k_values}
    results["MRR"] = float(np.mean(rrs))
    results["n_evaluated"] = float(n)
    return results


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

_GEN_SYSTEM = (
    "You are a helpful assistant. Answer the question as accurately as possible "
    "using only the provided context passages."
)
_GEN_USER = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

_JUDGE_SYSTEM = (
    "You are an expert evaluator for question-answering systems.\n"
    "Rate the Generated Answer against the Reference Answer on a 1–5 scale:\n"
    "  1 = completely wrong / irrelevant\n"
    "  2 = mostly wrong\n"
    "  3 = partially correct\n"
    "  4 = mostly correct, minor gaps\n"
    "  5 = comprehensive and accurate\n"
    'Respond with ONLY valid JSON: {"score": <integer 1-5>, "reason": "<brief reason>"}'
)
_JUDGE_USER = (
    "Question: {question}\n\n"
    "Reference Answer:\n{reference}\n\n"
    "Generated Answer:\n{generated}"
)


def _generate_answer(question: str, passages: list[str], client, model: str) -> str:
    context = "\n\n---\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _GEN_SYSTEM},
            {"role": "user", "content": _GEN_USER.format(context=context, question=question)},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    return resp.choices[0].message.content or ""


def _llm_judge(question: str, reference: str, generated: str, client, model: str) -> float:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": _JUDGE_USER.format(
                question=question, reference=reference, generated=generated
            )},
        ],
        temperature=0.0,
        max_tokens=150,
    )
    raw = re.sub(r"```(?:json)?|```", "", resp.choices[0].message.content or "").strip()
    try:
        d = json.loads(raw)
        return max(1.0, min(5.0, float(d["score"])))
    except Exception:
        m = re.search(r"[1-5]", raw)
        return float(m.group()) if m else float("nan")


def _rouge_l(hyp: str, ref: str) -> float:
    if not HAS_ROUGE:
        return float("nan")
    return float(_rouge.score(ref, hyp)["rougeL"].fmeasure)


def _semantic_sim_batch(hyps: list[str], refs: list[str], embed_fn) -> list[float]:
    """Cosine similarity between hypothesis and reference using the pipeline's embedder."""
    scores = []
    for h, r in zip(hyps, refs):
        vh = embed_fn(h).astype(np.float32)
        vr = embed_fn(r).astype(np.float32)
        nh, nr = float(np.linalg.norm(vh)), float(np.linalg.norm(vr))
        if nh > 0 and nr > 0:
            scores.append(float(np.dot(vh / nh, vr / nr)))
        else:
            scores.append(float("nan"))
    return scores


def _bertscore_batch(
    hyps: list[str], refs: list[str], model_type: str, embed_fn=None
) -> tuple[list[float], str]:
    """
    Try BERTScore; fall back to sentence-transformer cosine similarity
    if bert-score is unavailable or crashes (e.g. Python 3.14 / tokenizers wheel issue).
    Returns (scores, metric_label).
    """
    if HAS_BERTSCORE:
        try:
            _, _, F1 = _bs_mod.score(
                hyps, refs, model_type=model_type, verbose=False, batch_size=16
            )
            return [float(x) for x in F1.tolist()], "BERTScore F1"
        except Exception as exc:
            print(
                f"[WARNING] BERTScore failed ({exc}).\n"
                "  Falling back to sentence-transformer cosine similarity."
            )
    if embed_fn is not None:
        print("[eval] Computing semantic similarity with sentence-transformers…")
        return _semantic_sim_batch(hyps, refs, embed_fn), "SemanticSim F1"
    return [float("nan")] * len(hyps), "BERTScore F1"


def _mean_valid(vals: list[float]) -> float:
    v = [x for x in vals if not math.isnan(x)]
    return float(np.mean(v)) if v else float("nan")


# ---------------------------------------------------------------------------
# Generation metrics
# ---------------------------------------------------------------------------

def compute_generation_metrics(
    qa_pairs: list[dict],
    index: RetrievalIndex,
    corpus_map: dict[str, str],
    client,
    model: str = "gpt-4o-mini",
    top_k: int = 5,
    bertscore_model: str = "distilbert-base-uncased",
    cache_path: str | None = None,
    verbose: bool = True,
) -> dict[str, float]:
    """
    For each QA pair:
      1. Retrieve top_k context paragraphs with the question.
      2. Generate an answer via LLM.
      3. Compare to the reference answer: ROUGE-L, BERTScore (or SemanticSim fallback), LLM Judge.

    cache_path: if provided, generated answers and judge scores are saved here
    after every item so a crashed run can be resumed without repeating LLM calls.
    On re-run, already-cached items are loaded and skipped automatically.
    """
    if not HAS_ROUGE:
        print(
            "[WARNING] rouge-score not installed — ROUGE-L will be NaN.\n"
            "  Install: pip install rouge-score"
        )
    if not HAS_BERTSCORE:
        print(
            "[WARNING] bert-score not available — will use sentence-transformer "
            "cosine similarity as SemanticSim F1 instead."
        )

    # --- Load existing cache ---
    cache: list[dict] = []
    if cache_path and Path(cache_path).exists():
        with open(cache_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    cache.append(json.loads(line))
        if verbose and cache:
            print(f"[eval] Resuming from cache: {len(cache)}/{len(qa_pairs)} items already done.")

    cached_questions = {c["question"] for c in cache}

    hyps: list[str] = [c["generated"] for c in cache]
    refs: list[str] = [c["reference"] for c in cache]
    judge_scores: list[float] = [c["judge_score"] for c in cache]

    # --- Open cache file for appending ---
    cache_fh = open(cache_path, "a", encoding="utf-8") if cache_path else None

    try:
        for i, qa in enumerate(qa_pairs, start=1):
            question = qa["question"]
            if question in cached_questions:
                if verbose:
                    print(f"  [{i}/{len(qa_pairs)}] (cached) {question[:72]}…")
                continue

            if verbose:
                print(f"  [{i}/{len(qa_pairs)}] {question[:72]}…")

            reference = _parse_reference_answer(qa.get("answer", ""))

            passages = [
                corpus_map[pid]
                for pid, _ in index.retrieve(question, k=top_k)
                if pid in corpus_map
            ]
            if not passages:
                stored = qa.get("context", "")
                passages = [stored] if stored else ["No context available."]

            generated = _generate_answer(question, passages, client, model)
            judge = _llm_judge(question, reference, generated, client, model)

            hyps.append(generated)
            refs.append(reference)
            judge_scores.append(judge)

            if cache_fh:
                cache_fh.write(json.dumps({
                    "question": question,
                    "generated": generated,
                    "reference": reference,
                    "judge_score": judge,
                }, ensure_ascii=False) + "\n")
                cache_fh.flush()
    finally:
        if cache_fh:
            cache_fh.close()

    rouge_scores = [_rouge_l(h, r) for h, r in zip(hyps, refs)]
    bert_scores, bert_label = _bertscore_batch(
        hyps, refs, bertscore_model, embed_fn=index._embed_fn
    )

    return {
        "ROUGE-L F1": _mean_valid(rouge_scores),
        bert_label: _mean_valid(bert_scores),
        "LLM Judge (/5)": _mean_valid(judge_scores),
        "n_evaluated": float(len(qa_pairs)),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    if isinstance(v, float):
        return "N/A" if math.isnan(v) else f"{v:.3f}"
    return str(int(v)) if isinstance(v, (int, float)) else str(v)


def print_report(
    label: str,
    retrieval: dict[str, float],
    generation: dict[str, float] | None,
    output_path: str | None,
) -> None:
    print(f"\n{'=' * 54}")
    print(f"  SoG Evaluation — {label}")
    print(f"{'=' * 54}")

    print("\nRetrieval Metrics")
    print("-" * 42)
    for k, v in retrieval.items():
        if k == "n_evaluated":
            continue
        print(f"  {k:<14}  {_fmt(v)}")
    print(f"  (n = {int(retrieval.get('n_evaluated', 0))} QA pairs)")

    if generation:
        print("\nGeneration Metrics")
        print("-" * 42)
        for k, v in generation.items():
            if k == "n_evaluated":
                continue
            print(f"  {k:<22}  {_fmt(v)}")
        print(f"  (n = {int(generation.get('n_evaluated', 0))} QA pairs)")

    print(f"{'=' * 54}\n")

    if output_path:
        out: dict = {"label": label, "retrieval": {}, "generation": {}}
        for k, v in retrieval.items():
            out["retrieval"][k] = None if math.isnan(v) else v
        if generation:
            for k, v in generation.items():
                out["generation"][k] = None if isinstance(v, float) and math.isnan(v) else v
        Path(output_path).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[eval] Results saved to {output_path}")


# ---------------------------------------------------------------------------
# Core evaluation runner (also usable as a library function)
# ---------------------------------------------------------------------------

def evaluate(
    graph_json: str,
    qa_jsonl: str,
    embed_model: str = "BAAI/bge-small-en-v1.5",
    generate: bool = False,
    model: str = "gpt-4o-mini",
    base_url: str | None = None,
    api_key: str | None = None,
    top_k: int = 5,
    bertscore_model: str = "distilbert-base-uncased",
    limit: int | None = None,
    output_json: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run the full evaluation suite and return a dict with 'retrieval' (and
    optionally 'generation') result sub-dicts.

    base_url / api_key controls which LLM provider is used for generation:
      - None (default) → reads OPENAI_API_KEY from environment (OpenAI)
      - Groq           → base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY
      - Ollama         → base_url="http://localhost:11434/v1", api_key="ollama"
      - OpenRouter     → base_url="https://openrouter.ai/api/v1", api_key=...

    Can be called programmatically:
        from evaluate_sog import evaluate
        results = evaluate(
            graph_json="outputs/foo_context_graph.json",
            qa_jsonl="outputs/foo_sog_qa.jsonl",
            base_url="https://api.groq.com/openai/v1",
            api_key="gsk_…",
            model="llama-3.3-70b-versatile",
        )
    """
    label = Path(qa_jsonl).stem

    # --- Load ---
    if verbose:
        print(f"[eval] Loading QA pairs from {qa_jsonl}…")
    qa_pairs = load_qa_pairs(qa_jsonl)
    if limit:
        qa_pairs = qa_pairs[:limit]
    if verbose:
        print(f"[eval] {len(qa_pairs)} QA pairs.")

    if verbose:
        print(f"[eval] Loading corpus from {graph_json}…")
    _, paragraphs = load_corpus(graph_json)
    if verbose:
        print(f"[eval] {len(paragraphs)} unique paragraphs in corpus.")

    # Warn early if path_ids don't overlap with the corpus
    all_path_ids = {pid for qa in qa_pairs for pid in (qa.get("path_ids") or [])}
    corpus_ids = {p.para_id for p in paragraphs}
    if all_path_ids and not (all_path_ids & corpus_ids):
        print(
            f"[WARNING] None of the {len(all_path_ids)} unique path_ids in "
            f"'{Path(qa_jsonl).name}' match any para_id in '{Path(graph_json).name}'.\n"
            f"  → Retrieval metrics will be 0. Are you pairing the right files?\n"
            f"  Example path_ids : {sorted(all_path_ids)[:3]}\n"
            f"  Example para_ids : {sorted(corpus_ids)[:3]}"
        )

    # --- Build index ---
    embed_fn = make_embed_fn(embed_model)
    index = RetrievalIndex(paragraphs, embed_fn)
    index.build(verbose=verbose)

    # --- Retrieval ---
    if verbose:
        print("\n[eval] Computing retrieval metrics…")
    retrieval_results = compute_retrieval_metrics(qa_pairs, index)

    # --- Generation (optional) ---
    gen_results: dict[str, float] | None = None
    if generate:
        try:
            from openai import OpenAI
            client_kwargs: dict = {}
            if base_url:
                client_kwargs["base_url"] = base_url
            if api_key:
                client_kwargs["api_key"] = api_key
            client = OpenAI(**client_kwargs)
            provider = base_url or "OpenAI"
            if verbose:
                print(f"[eval] LLM provider: {provider}  model: {model}")
        except Exception as exc:
            print(f"[eval] Cannot initialise LLM client ({exc}). Skipping generation metrics.")
        else:
            corpus_map = {p.para_id: p.text for p in paragraphs}
            if verbose:
                print("\n[eval] Computing generation metrics…")
            # Cache lives next to the JSONL, e.g. foo_sog_qa_gen_cache.jsonl
            auto_cache = str(Path(qa_jsonl).with_suffix("")) + "_gen_cache.jsonl"
            gen_results = compute_generation_metrics(
                qa_pairs=qa_pairs,
                index=index,
                corpus_map=corpus_map,
                client=client,
                model=model,
                top_k=top_k,
                bertscore_model=bertscore_model,
                cache_path=auto_cache,
                verbose=verbose,
            )

    print_report(label, retrieval_results, gen_results, output_json)

    result = {"retrieval": retrieval_results}
    if gen_results:
        result["generation"] = gen_results
    return result


# ---------------------------------------------------------------------------
# Batch mode: evaluate all graph+JSONL pairs in a directory
# ---------------------------------------------------------------------------

def _find_pairs(output_dir: str) -> list[tuple[str, str]]:
    """
    Auto-discover matching (graph_json, qa_jsonl) pairs in a directory.
    Matches files that share the same stem prefix, e.g.
      Attention_Is_All_You_Need_context_graph.json  <->
      Attention_Is_All_You_Need_sog_qa.jsonl
    """
    d = Path(output_dir)
    graphs = {f.stem.replace("_context_graph", ""): f for f in d.glob("*_context_graph.json")}
    jsonls = {f.stem.replace("_sog_qa", ""): f for f in d.glob("*_sog_qa.jsonl")}
    pairs = []
    for key in sorted(graphs):
        if key in jsonls:
            pairs.append((str(graphs[key]), str(jsonls[key])))
    return pairs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SoG pipeline outputs.\n"
            "Retrieval: Hit@1, Hit@5, MRR\n"
            "Generation (--generate): ROUGE-L, BERTScore, LLM Judge"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Explicit pair
    parser.add_argument("--graph", default=None,
                        help="Context graph JSON (e.g. outputs/foo_context_graph.json)")
    parser.add_argument("--qa", default=None,
                        help="QA JSONL (e.g. outputs/foo_sog_qa.jsonl)")

    # Batch mode
    parser.add_argument("--output_dir", default=None,
                        help="Directory to scan for all graph+JSONL pairs (batch mode)")

    # Shared options
    parser.add_argument("--embed_model", default="BAAI/bge-small-en-v1.5",
                        help="Sentence-transformer model for embeddings (default: BAAI/bge-small-en-v1.5)")
    parser.add_argument("--generate", action="store_true",
                        help="Also compute generation metrics (requires an API key)")
    parser.add_argument("--groq", action="store_true",
                        help="Use Groq API (sets base_url automatically, reads GROQ_API_KEY)")
    parser.add_argument("--base_url", default=None,
                        help="Custom OpenAI-compatible base URL "
                             "(e.g. https://api.groq.com/openai/v1, "
                             "http://localhost:11434/v1)")
    parser.add_argument("--api_key", default=None,
                        help="Explicit API key (overrides environment variable)")
    parser.add_argument("--model", default="llama-3.1-8b-instant",
                        help="Model name for answer generation and LLM judge "
                             "(e.g. llama-3.3-70b-versatile for Groq)")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Passages retrieved per question (default: 5)")
    parser.add_argument("--bertscore_model", default="distilbert-base-uncased",
                        help="HuggingFace model for BERTScore (default: distilbert-base-uncased)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N QA pairs per file")
    parser.add_argument("--output", default=None,
                        help="Save results as JSON (single-pair mode only)")

    args = parser.parse_args()

    # Determine pairs to evaluate
    if args.output_dir:
        pairs = _find_pairs(args.output_dir)
        if not pairs:
            print(f"[eval] No matching graph+JSONL pairs found in {args.output_dir}.")
            sys.exit(1)
        print(f"[eval] Found {len(pairs)} pair(s) in {args.output_dir}.")
    elif args.graph and args.qa:
        pairs = [(args.graph, args.qa)]
    else:
        parser.error("Provide either --graph + --qa, or --output_dir.")

    # Resolve provider settings
    import os
    base_url = args.base_url
    api_key = args.api_key
    if args.groq:
        base_url = base_url or "https://api.groq.com/openai/v1"
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            parser.error(
                "--groq requires a Groq API key. "
                "Set the GROQ_API_KEY environment variable or pass --api_key."
            )

    kwargs = dict(
        embed_model=args.embed_model,
        generate=args.generate,
        model=args.model,
        base_url=base_url,
        api_key=api_key,
        top_k=args.top_k,
        bertscore_model=args.bertscore_model,
        limit=args.limit,
    )

    all_results = {}
    for graph_json, qa_jsonl in pairs:
        out_path = args.output if (args.output and len(pairs) == 1) else None
        result = evaluate(graph_json=graph_json, qa_jsonl=qa_jsonl,
                          output_json=out_path, **kwargs)
        all_results[Path(qa_jsonl).stem] = result

    # If batch mode with multiple pairs, summarise retrieval across docs
    if len(pairs) > 1:
        print("\n" + "=" * 54)
        print("  Summary Across All Documents")
        print("=" * 54)
        for key, k_val in [("Hit@1", 1), ("Hit@5", 5), ("MRR", None)]:
            vals = [
                r["retrieval"].get(key, float("nan"))
                for r in all_results.values()
                if not math.isnan(r["retrieval"].get(key, float("nan")))
            ]
            label = key
            avg = float(np.mean(vals)) if vals else float("nan")
            print(f"  {label:<14}  {_fmt(avg)}  (avg over {len(vals)} docs)")
        print("=" * 54 + "\n")


if __name__ == "__main__":
    main()
