"""
context_graph.py
================
Synthesize-on-Graph (SoG) — Context Graph Construction
Implements Section 3.1 of the paper:
  - Entity extraction (via LLM)
  - Entity-context mapping
  - Co-occurrence-based context graph
  - Math-aware chunking (equations preserved as atomic entities)

Drop this file into:  generation/graph-based/src/context_graph.py
"""

from __future__ import annotations

import re
import json
import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI


# ---------------------------------------------------------------------------
# Quota / rate-limit helpers
# ---------------------------------------------------------------------------

class QuotaExhaustedError(RuntimeError):
    """Raised when the OpenAI account has no remaining quota."""


def _is_insufficient_quota(exc: Exception) -> bool:
    """Return True if *exc* is an OpenAI insufficient_quota error."""
    msg = str(exc)
    return "insufficient_quota" in msg or "exceeded your current quota" in msg


def _is_rate_limit(exc: Exception) -> bool:
    """Return True if *exc* is a transient rate-limit (429) error."""
    msg = str(exc)
    return "rate_limit" in msg or ("429" in msg and not _is_insufficient_quota(exc))


def _llm_call_with_retry(fn, *, max_retries: int = 5):
    """
    Call *fn()* (a zero-argument callable that calls the OpenAI API).
    - Re-raises QuotaExhaustedError immediately on insufficient_quota.
    - Retries with exponential back-off on transient rate-limit errors.
    - Lets all other exceptions propagate unchanged.
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            if _is_insufficient_quota(exc):
                raise QuotaExhaustedError(
                    "OpenAI quota exhausted. Add credits at "
                    "https://platform.openai.com/account/billing/overview"
                ) from exc
            if _is_rate_limit(exc):
                wait = 2 ** attempt  # 1 s, 2 s, 4 s, 8 s, 16 s
                print(f"[rate limit] Retrying in {wait}s (attempt {attempt + 1}/{max_retries})…")
                time.sleep(wait)
                continue
            raise  # unknown error – propagate
    raise RuntimeError("Max retries exceeded due to rate limiting.")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Paragraph:
    doc_id: str
    para_id: str          # "{doc_id}_p{index}"
    text: str
    math_blocks: list[str] = field(default_factory=list)   # LaTeX/display math preserved
    entities: list[str] = field(default_factory=list)


@dataclass
class ContextGraph:
    """
    G = (E, edges)
    nodes  : entity strings
    edges  : set of (entity_a, entity_b) pairs that co-occur in the same paragraph
    mapping: entity -> list[Paragraph]  (entity-context map M)
    """
    nodes: set[str] = field(default_factory=set)
    edges: set[tuple[str, str]] = field(default_factory=set)
    mapping: dict[str, list[Paragraph]] = field(default_factory=lambda: defaultdict(list))

    def add_paragraph(self, para: Paragraph) -> None:
        ents = para.entities
        for e in ents:
            self.nodes.add(e)
            self.mapping[e].append(para)
        # Add co-occurrence edges
        for i, ex in enumerate(ents):
            for ey in ents[i + 1:]:
                edge = tuple(sorted([ex, ey]))
                self.edges.add(edge)  # type: ignore[arg-type]

    def neighbors(self, entity: str) -> list[str]:
        nbrs = []
        for (a, b) in self.edges:
            if a == entity:
                nbrs.append(b)
            elif b == entity:
                nbrs.append(a)
        return nbrs

    def to_dict(self) -> dict:
        return {
            "nodes": list(self.nodes),
            "edges": [list(e) for e in self.edges],
            "mapping": {
                k: [{"doc_id": p.doc_id, "para_id": p.para_id, "text": p.text,
                      "math_blocks": p.math_blocks, "entities": p.entities}
                    for p in v]
                for k, v in self.mapping.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ContextGraph":
        g = cls()
        g.nodes = set(d["nodes"])
        g.edges = {tuple(e) for e in d["edges"]}  # type: ignore[misc]
        for k, paras in d["mapping"].items():
            for p in paras:
                g.mapping[k].append(Paragraph(**p))
        return g


# ---------------------------------------------------------------------------
# Math-aware paragraph splitting
# ---------------------------------------------------------------------------

# Regex patterns for LaTeX math environments
_DISPLAY_MATH_RE = re.compile(
    r"(\$\$.*?\$\$"            # $$...$$
    r"|\\\[.*?\\\]"            # \[...\]
    r"|\\begin\{equation\*?\}.*?\\end\{equation\*?\}"
    r"|\\begin\{align\*?\}.*?\\end\{align\*?\}"
    r"|\\begin\{gather\*?\}.*?\\end\{gather\*?\})",
    re.DOTALL,
)

_INLINE_MATH_RE = re.compile(r"\$(?!\$).+?(?<!\$)\$")


def extract_math_and_clean(text: str) -> tuple[str, list[str]]:
    """
    Replace display math blocks with placeholder tokens and return:
      - cleaned text (placeholders in place of equations)
      - list of original math strings

    This ensures equations are treated as ATOMIC units during entity extraction
    rather than being split across token boundaries.

    Inline math ($...$) is kept in-place but also catalogued so the LLM
    receives the full formula as a single token-like string.
    """
    math_blocks: list[str] = []

    def _replace_display(m: re.Match) -> str:
        idx = len(math_blocks)
        math_blocks.append(m.group(0))
        return f"__MATH_{idx}__"

    cleaned = _DISPLAY_MATH_RE.sub(_replace_display, text)

    # Inline math: catalogue but keep visible (they're short; no placeholder needed)
    for m in _INLINE_MATH_RE.finditer(cleaned):
        math_blocks.append(m.group(0))

    return cleaned, math_blocks


def split_into_paragraphs(doc_id: str, raw_text: str, min_length: int = 80) -> list[Paragraph]:
    """
    Split a document into paragraphs, extracting and preserving math.
    Paragraphs shorter than `min_length` chars are merged with the next one.
    """
    raw_paras = [p.strip() for p in re.split(r"\n{2,}", raw_text) if p.strip()]
    paragraphs: list[Paragraph] = []
    buffer = ""

    for i, raw in enumerate(raw_paras):
        cleaned, math_blocks = extract_math_and_clean(raw)
        buffer_cleaned = buffer + (" " if buffer else "") + cleaned
        buffer_math = math_blocks  # simplified: track from last segment

        if len(buffer_cleaned) >= min_length or i == len(raw_paras) - 1:
            para_id = f"{doc_id}_p{len(paragraphs)}"
            paragraphs.append(Paragraph(
                doc_id=doc_id,
                para_id=para_id,
                text=buffer_cleaned,
                math_blocks=buffer_math,
            ))
            buffer = ""
        else:
            buffer = buffer_cleaned

    return paragraphs


# ---------------------------------------------------------------------------
# Entity extraction via LLM
# ---------------------------------------------------------------------------

ENTITY_EXTRACTION_PROMPT = """\
You are an expert at extracting key entities and concepts from academic text.

Given a paragraph of text (which may contain mathematical placeholders like __MATH_0__
representing equations), extract the most important named entities, concepts, and
mathematical objects. Mathematical placeholders like __MATH_0__ should be treated as
atomic entities — include them as-is in your output.

Return a JSON object with a single key "entities" containing a list of strings.
Keep entities concise (1-5 words). Extract at most 10 entities per paragraph.
Do not explain — only return valid JSON.

Paragraph:
{text}
"""


def extract_entities_llm(
    para: Paragraph,
    client: OpenAI,
    model: str = "gpt-4o-mini",
) -> list[str]:
    """Call the LLM to extract entities from a paragraph."""
    prompt = ENTITY_EXTRACTION_PROMPT.format(text=para.text[:2000])

    def _call():
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )

    try:
        response = _llm_call_with_retry(_call)
        raw = response.choices[0].message.content or ""
        # Strip markdown fences if present
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        data = json.loads(raw)
        entities = data.get("entities", [])
        # Append math blocks as additional entities so they become graph nodes
        for mb in para.math_blocks:
            short = mb[:60].replace("\n", " ")  # fingerprint long formulas
            entities.append(short)
        return [str(e).strip() for e in entities if e]
    except QuotaExhaustedError:
        raise  # propagate immediately so build_context_graph can stop early
    except Exception as exc:
        print(f"[entity extraction] {para.para_id}: {exc}")
        return []


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_context_graph(
    documents: dict[str, str],       # {doc_id: raw_text}
    client: OpenAI,
    model: str = "gpt-4o-mini",
    max_paras_per_doc: int | None = None,
    verbose: bool = True,
) -> ContextGraph:
    """
    Full pipeline: raw docs -> paragraphs -> entity extraction -> ContextGraph.

    Args:
        documents: mapping of document ID to raw text content
        client:    OpenAI-compatible client
        model:     generation model name
        max_paras_per_doc: cap on paragraphs processed per doc (useful for testing)

    Returns:
        A populated ContextGraph instance.
    """
    graph = ContextGraph()

    for doc_id, raw_text in documents.items():
        if verbose:
            print(f"[build_context_graph] Processing doc: {doc_id}")

        paragraphs = split_into_paragraphs(doc_id, raw_text)
        if max_paras_per_doc:
            paragraphs = paragraphs[:max_paras_per_doc]

        for para in paragraphs:
            try:
                entities = extract_entities_llm(para, client, model)
            except QuotaExhaustedError as exc:
                print(f"\n[build_context_graph] FATAL: {exc}")
                print("  Stopping early. Partial graph will be returned.")
                return graph
            para.entities = entities
            graph.add_paragraph(para)

        if verbose:
            print(f"  -> {len(paragraphs)} paragraphs, "
                  f"{len(graph.nodes)} nodes so far")

    return graph
