"""
cross_document_sampling.py
==========================
Synthesize-on-Graph (SoG) — Cross-Document Sampling (Section 3.2)

Implements:
  - BFS-based context-graph traversal with embedding similarity selection
  - Secondary sampling with entity utilization balancing
  - Controlled allocation into path subsets (CoT paths + CC pairs)

Drop into: generation/graph-based/src/cross_document_sampling.py
"""

from __future__ import annotations

import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from context_graph import ContextGraph, Paragraph


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PathNode:
    entity: str
    paragraph: Paragraph


# A path is a list of PathNode objects: [(root_entity, root_para), (e1, c1), ...]
Path = list[PathNode]


@dataclass
class SampledPathSet:
    """Output of secondary sampling: CoT paths + CC pairs."""
    cot: list[Path] = field(default_factory=list)
    cc: list[tuple[PathNode, PathNode]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Embedding similarity helper (plug in your own embedder)
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


class EmbeddingIndex:
    """
    Thin wrapper around any sentence-transformer-style embedder.
    Cache embeddings by para_id to avoid redundant API calls.
    """

    def __init__(self, embed_fn):
        """
        embed_fn: callable(text: str) -> np.ndarray
        e.g. from sentence_transformers import SentenceTransformer
             model = SentenceTransformer('BAAI/bge-small-en-v1.5')
             embed_fn = lambda t: model.encode(t, normalize_embeddings=True)
        """
        self._fn = embed_fn
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, para: Paragraph) -> np.ndarray:
        if para.para_id not in self._cache:
            self._cache[para.para_id] = self._fn(para.text)
        return self._cache[para.para_id]

    def similarity(self, root_para: Paragraph, candidate: Paragraph) -> float:
        return cosine_similarity(self.embed(root_para), self.embed(candidate))


# ---------------------------------------------------------------------------
# BFS traversal (Section 3.2.1 + 3.2.2)
# ---------------------------------------------------------------------------

def _traverse_from_root(
    root_entity: str,
    root_para: Paragraph,
    graph: ContextGraph,
    index: EmbeddingIndex,
    depth: int = 2,
    top_w: int = 3,
) -> list[Path]:
    """
    BFS from (root_entity, root_para) up to `depth` hops.
    At each hop, select the top-W most similar paragraphs among neighbours.
    Returns all completed paths (sequences of PathNodes).
    """
    # Each frontier item: current partial path
    frontier: deque[Path] = deque()
    frontier.append([PathNode(entity=root_entity, paragraph=root_para)])
    completed: list[Path] = []

    for _ in range(depth):
        next_frontier: deque[Path] = deque()
        for path in frontier:
            current_entity = path[-1].entity
            neighbours = graph.neighbors(current_entity)
            if not neighbours:
                completed.append(path)
                continue

            # Collect candidate (entity, para) pairs from neighbours
            candidates: list[tuple[str, Paragraph, float]] = []
            for nb_entity in neighbours:
                for nb_para in graph.mapping.get(nb_entity, []):
                    sim = index.similarity(root_para, nb_para)
                    candidates.append((nb_entity, nb_para, sim))

            # Pick top-W by similarity
            candidates.sort(key=lambda x: x[2], reverse=True)
            for nb_entity, nb_para, _ in candidates[:top_w]:
                new_path = path + [PathNode(entity=nb_entity, paragraph=nb_para)]
                next_frontier.append(new_path)

        if not next_frontier:
            break
        frontier = next_frontier

    completed.extend(list(frontier))  # flush remaining partial paths
    return completed


def build_all_paths(
    graph: ContextGraph,
    index: EmbeddingIndex,
    depth: int = 1,
    top_w: int = 3,
    max_starts_per_entity: int = 5,
) -> list[Path]:
    """
    Traverse from every entity in the graph as root (Section 3.2.1).
    If an entity maps to many paragraphs, randomly sample up to
    `max_starts_per_entity` starting paragraphs (hyperparameter S in the paper).
    """
    all_paths: list[Path] = []
    for entity in graph.nodes:
        paras = graph.mapping.get(entity, [])
        if not paras:
            continue
        starts = random.sample(paras, min(len(paras), max_starts_per_entity))
        for root_para in starts:
            paths = _traverse_from_root(
                root_entity=entity,
                root_para=root_para,
                graph=graph,
                index=index,
                depth=depth,
                top_w=top_w,
            )
            all_paths.extend(paths)
    return all_paths


# ---------------------------------------------------------------------------
# Secondary sampling + controlled allocation (Section 3.2.3)
# ---------------------------------------------------------------------------

def _path_utilization_count(
    path: Path,
    utilization: dict[str, int],
) -> int:
    return sum(utilization.get(node.entity, 0) for node in path)


def _update_utilization(
    path: Path,
    utilization: dict[str, int],
    entity_to_chunks: dict[str, set[str]],
    coverage_tracker: set[str],
) -> None:
    for node in path:
        utilization[node.entity] = utilization.get(node.entity, 0) + 1
        coverage_tracker.add(node.paragraph.para_id)
        entity_to_chunks[node.entity].add(node.paragraph.para_id)


def balanced_sampling(
    paths: list[Path],
    target_coverage_rate: float,
    total_chunks: int,
    standard_length: int,
    entity_to_chunks: dict[str, set[str]],
    cc_trigger_ratio: float = 0.9,
) -> SampledPathSet:
    """
    Algorithm 1+2 from the paper.

    - Sorts paths by ascending utilization (least-used entities first).
    - Samples CoT paths until `target_coverage_rate` is achieved or
      `standard_length` paths are sampled.
    - When coverage gap remains after standard_length, triggers CC for
      the least-sampled sparse entities.

    Args:
        paths:                remaining path pool
        target_coverage_rate: fraction of corpus chunks to cover (r)
        total_chunks:         total number of unique paragraphs in corpus
        standard_length:      reference subset size (l) — typically total_chunks / (hop+1)
        entity_to_chunks:     mutable map entity -> set of chunk IDs seen so far
        cc_trigger_ratio:     fraction of standard_length after which CC kicks in

    Returns:
        SampledPathSet with .cot and .cc lists
    """
    result = SampledPathSet()
    utilization: dict[str, int] = defaultdict(int)
    coverage: set[str] = set()
    remaining = list(paths)

    def current_rate() -> float:
        return len(coverage) / total_chunks if total_chunks else 0.0

    while remaining and current_rate() < target_coverage_rate:
        # Sort ascending by sum of entity utilization counts
        remaining.sort(key=lambda p: _path_utilization_count(p, utilization))
        path = remaining.pop(0)
        result.cot.append(path)
        _update_utilization(path, utilization, entity_to_chunks, coverage)

        # CC trigger: if we've reached standard_length but coverage gap remains
        if len(result.cot) >= standard_length and current_rate() < target_coverage_rate:
            delta_r = (target_coverage_rate - current_rate()) / target_coverage_rate
            cut = int((1 - delta_r) * standard_length)
            # Return excess CoT paths back to pool
            returned = result.cot[cut:]
            remaining.extend(returned)
            result.cot = result.cot[:cut]
            # Reverse utilization for returned paths
            for p in returned:
                for node in p:
                    utilization[node.entity] = max(0, utilization[node.entity] - 1)
                    coverage.discard(node.paragraph.para_id)

            # Build CC pairs from least-sampled entities
            k = int(delta_r * standard_length)
            sparse_entities = sorted(utilization, key=lambda e: utilization[e])[:k]
            random.shuffle(sparse_entities)
            for i in range(0, len(sparse_entities) - 1, 2):
                ex, ey = sparse_entities[i], sparse_entities[i + 1]
                paras_x = graph_mapping_lookup(entity_to_chunks, ex)
                paras_y = graph_mapping_lookup(entity_to_chunks, ey)
                if paras_x and paras_y:
                    result.cc.append((
                        PathNode(entity=ex, paragraph=random.choice(paras_x)),
                        PathNode(entity=ey, paragraph=random.choice(paras_y)),
                    ))
            break

    return result


def graph_mapping_lookup(
    entity_to_chunks: dict[str, set[str]],
    entity: str,
    # We need access to the full mapping for actual Paragraph objects
    # Pass the graph mapping instead in practice
    graph_mapping: dict[str, list[Paragraph]] | None = None,
) -> list[Paragraph]:
    """Helper: return Paragraph objects for an entity."""
    if graph_mapping:
        return graph_mapping.get(entity, [])
    return []


def secondary_sampling(
    all_paths: list[Path],
    graph: ContextGraph,
    target_coverage_rate: float = 1.0,
    hop: int = 1,
) -> list[SampledPathSet]:
    """
    Iteratively allocate paths into subsets (each subset = one generation batch).
    Returns a list of SampledPathSet objects. The first has highest coverage;
    subsequent ones progressively lower.
    """
    total_chunks = sum(len(paras) for paras in graph.mapping.values())
    standard_length = max(1, total_chunks // (hop + 1))
    entity_to_chunks: dict[str, set[str]] = defaultdict(set)

    remaining = list(all_paths)
    subsets: list[SampledPathSet] = []

    while remaining:
        subset = balanced_sampling(
            paths=remaining,
            target_coverage_rate=target_coverage_rate,
            total_chunks=total_chunks,
            standard_length=standard_length,
            entity_to_chunks=entity_to_chunks,
        )
        # Supplement CC lookup with graph mapping
        subsets.append(subset)
        # Remove used paths from pool (approximation: remove CoT paths used)
        used_ids = {id(p) for p in subset.cot}
        remaining = [p for p in remaining if id(p) not in used_ids]
        if not subset.cot and not subset.cc:
            break  # safety: avoid infinite loop

    return subsets
