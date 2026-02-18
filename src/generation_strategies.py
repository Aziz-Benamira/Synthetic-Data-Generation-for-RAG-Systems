"""
generation_strategies.py
========================
Synthesize-on-Graph (SoG) — Generation Strategies (Section 3.3)

Implements:
  - Chain-of-Thought (CoT) generation from cross-document paths
  - Contrastive Clarifying (CC) generation for sparse entities
  - Final QA dataset assembly

Drop into: generation/graph-based/src/generation_strategies.py
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from cross_document_sampling import Path, PathNode, SampledPathSet


# ---------------------------------------------------------------------------
# Prompt templates (mirrors Figures 5 & 6 from the paper)
# ---------------------------------------------------------------------------

COT_SYSTEM_PROMPT = """\
You are tasked with constructing a coherent narrative that builds a causal \
relationship among several text fragments.
Your role involves generating an information chain that fulfills the following criteria:

1. **Causal Narrative Development**: Use the information from each text fragment \
to build a step-by-step narrative that establishes a causal relationship. Develop \
a storyline where each fragment logically leads to the next, creating a clear flow \
of cause and effect.
2. **Use Provided Information Fully**: Ensure that the generated narrative makes \
full use of the key information from each text fragment. The causal relationships \
should be based directly on the details provided.
3. **Logical Structure with Transitions**: Structure the narrative to include \
distinct phases — Initiation, Development, Turning Point, and Conclusion — with \
natural transitions that preserve the logical flow.
4. **Chain-of-Thought Question and Answer**: Based on the causal narrative, \
formulate a question that requires understanding the entire information chain to \
answer. Provide a detailed answer in Chain-of-Thought style, breaking down the \
reasoning step by step.

Output format (JSON):
{
  "narrative": "...",
  "question": "...",
  "cot_answer": "step 1: ... step 2: ... Final answer: ..."
}
Only return valid JSON. No markdown fences.
"""

CC_SYSTEM_PROMPT = """\
You are tasked with generating a comparative analysis based on several text fragments. \
In this scenario, the text fragments may be unrelated to each other in certain aspects, \
and your role involves generating a thoughtful contrastive narrative that fulfills:

1. **Entity-Focused Comparative Analysis**: Focus on comparing and contrasting the \
given text fragments. Do not force a connection between unrelated fragments.
2. **Maximize Use of Provided Information**: Make full use of key information in \
each fragment, drawing on the distinct points presented.
3. **Highlight Differences and Similarities**: Identify and highlight differences \
and any possible similarities between key entities. If no direct similarity exists, \
focus on how each entity contributes its unique perspective or domain.
4. **Objective and Analytical Tone**: Maintain an objective, analytical tone \
throughout the narrative.
5. **Structured and Cohesive Presentation**: Examine each entity in separate sections, \
then provide a comparative summary.

Output format (JSON):
{
  "comparative_narrative": "...",
  "question": "...",
  "answer": "..."
}
Only return valid JSON. No markdown fences.
"""


def _format_fragments(nodes: list[PathNode]) -> str:
    parts = []
    for i, node in enumerate(nodes, 1):
        text = node.paragraph.text[:800]  # truncate long paragraphs
        parts.append(f"**Fragment {i} [{node.entity}]:**\n{text}")
    return "\n\n".join(parts)


def _safe_parse_json(raw: str) -> dict:
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: return raw text in a wrapper
        return {"raw_output": raw}


# ---------------------------------------------------------------------------
# CoT generation
# ---------------------------------------------------------------------------

@dataclass
class CoTSample:
    path_ids: list[str]          # para_ids of the path nodes
    entities: list[str]
    narrative: str
    question: str
    cot_answer: str
    source: str = "cot"


def generate_cot_sample(
    path: Path,
    client: OpenAI,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
) -> CoTSample | None:
    """Generate one CoT QA sample from a multi-hop path."""
    if len(path) < 2:
        return None  # need at least 2 nodes for a meaningful path

    fragments = _format_fragments(path)
    user_msg = f"### INPUT:\n{fragments}"

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": COT_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=temperature,
            max_tokens=1200,
        )
        raw = resp.choices[0].message.content or ""
        data = _safe_parse_json(raw)

        return CoTSample(
            path_ids=[n.paragraph.para_id for n in path],
            entities=[n.entity for n in path],
            narrative=data.get("narrative", ""),
            question=data.get("question", ""),
            cot_answer=data.get("cot_answer", data.get("raw_output", "")),
        )
    except Exception as exc:
        print(f"[CoT generation] Error: {exc}")
        return None


# ---------------------------------------------------------------------------
# CC generation
# ---------------------------------------------------------------------------

@dataclass
class CCSample:
    entity_a: str
    entity_b: str
    para_id_a: str
    para_id_b: str
    comparative_narrative: str
    question: str
    answer: str
    source: str = "cc"


def generate_cc_sample(
    node_a: PathNode,
    node_b: PathNode,
    client: OpenAI,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
) -> CCSample | None:
    """Generate one CC contrastive QA sample from a pair of sparse-entity nodes."""
    fragments = _format_fragments([node_a, node_b])
    user_msg = f"### INPUT:\n{fragments}"

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CC_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=temperature,
            max_tokens=900,
        )
        raw = resp.choices[0].message.content or ""
        data = _safe_parse_json(raw)

        return CCSample(
            entity_a=node_a.entity,
            entity_b=node_b.entity,
            para_id_a=node_a.paragraph.para_id,
            para_id_b=node_b.paragraph.para_id,
            comparative_narrative=data.get("comparative_narrative", ""),
            question=data.get("question", ""),
            answer=data.get("answer", data.get("raw_output", "")),
        )
    except Exception as exc:
        print(f"[CC generation] Error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Batch generation over a SampledPathSet
# ---------------------------------------------------------------------------

def generate_from_subset(
    subset: SampledPathSet,
    client: OpenAI,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    verbose: bool = True,
) -> list[dict]:
    """
    Run both CoT and CC generation over one SampledPathSet.
    Returns a list of dicts ready for JSON serialisation / HuggingFace upload.
    """
    samples: list[dict] = []

    if verbose:
        print(f"[generate] CoT paths: {len(subset.cot)}, CC pairs: {len(subset.cc)}")

    for i, path in enumerate(subset.cot):
        sample = generate_cot_sample(path, client, model, temperature)
        if sample:
            samples.append({
                "source": sample.source,
                "entities": sample.entities,
                "path_ids": sample.path_ids,
                "question": sample.question,
                "answer": sample.cot_answer,
                "context": sample.narrative,
            })
        if verbose and i % 20 == 0:
            print(f"  CoT {i}/{len(subset.cot)} done")

    for i, (node_a, node_b) in enumerate(subset.cc):
        sample = generate_cc_sample(node_a, node_b, client, model, temperature)
        if sample:
            samples.append({
                "source": sample.source,
                "entities": [sample.entity_a, sample.entity_b],
                "path_ids": [sample.para_id_a, sample.para_id_b],
                "question": sample.question,
                "answer": sample.answer,
                "context": sample.comparative_narrative,
            })
        if verbose and i % 20 == 0:
            print(f"  CC {i}/{len(subset.cc)} done")

    return samples
