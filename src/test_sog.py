"""
test_sog.py
===========
Quick smoke test for the SoG pipeline using two toy academic paragraphs.
Run from the generation/graph-based/ directory:

    OPENAI_API_KEY=sk-... python test_sog.py

No real documents or GPU required — uses the random-embedding fallback
if sentence-transformers is absent.
"""

import json
import os
import sys

# Ensure local imports work
sys.path.insert(0, os.path.dirname(__file__))

from context_graph import split_into_paragraphs, extract_math_and_clean, ContextGraph, Paragraph

# ---- Toy documents --------------------------------------------------------
DOCS = {
    "paper_attention": """
Attention mechanisms allow models to focus on relevant parts of the input sequence.
The core formula is the scaled dot-product attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where $Q$, $K$, $V$ are query, key, and value matrices respectively, and $d_k$ is
the dimension of the key vectors. This formulation enables the model to attend
to different positions of the input simultaneously.

Multi-head attention extends this by running $h$ attention heads in parallel:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

Each head uses a separate set of learned projections $W_i^Q, W_i^K, W_i^V$.
""",

    "paper_transformer": """
The Transformer architecture, introduced by Vaswani et al. (2017), relies entirely
on attention mechanisms, dispensing with recurrence and convolutions.

The encoder stack consists of N identical layers, each containing a multi-head
self-attention sublayer and a position-wise feed-forward network.
Residual connections and layer normalisation are applied around each sublayer.

The positional encoding adds information about the relative or absolute position
of tokens in the sequence using sine and cosine functions of different frequencies:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

This allows the model to leverage order information without recurrence.
"""
}

# ---- Test 1: Math-aware splitting ----------------------------------------
def test_math_extraction():
    print("=== Test 1: Math-aware paragraph splitting ===")
    paras = split_into_paragraphs("paper_attention", DOCS["paper_attention"])
    print(f"  {len(paras)} paragraphs extracted")
    for p in paras:
        print(f"  [{p.para_id}] math_blocks={len(p.math_blocks)}")
        for mb in p.math_blocks:
            print(f"    MATH: {mb[:60]}...")
    print()


# ---- Test 2: Graph construction (without LLM — inject fake entities) ------
def test_graph_construction():
    print("=== Test 2: Context graph construction (fake entities) ===")
    graph = ContextGraph()

    paras_a = split_into_paragraphs("paper_attention", DOCS["paper_attention"])
    paras_b = split_into_paragraphs("paper_transformer", DOCS["paper_transformer"])

    # Inject synthetic entities (simulating LLM extraction)
    fake_entities = {
        paras_a[0].para_id: ["attention mechanism", "scaled dot-product", "__MATH_0__"],
        paras_a[1].para_id: ["multi-head attention", "__MATH_0__", "learned projections"],
        paras_b[0].para_id: ["Transformer architecture", "attention mechanism", "Vaswani et al."],
        paras_b[1].para_id: ["encoder stack", "residual connections", "positional encoding", "__MATH_0__"],
    }

    all_paras = paras_a + paras_b
    for para in all_paras:
        para.entities = fake_entities.get(para.para_id, [])
        graph.add_paragraph(para)

    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Edges: {len(graph.edges)}")
    print(f"  Neighbours of 'attention mechanism': {graph.neighbors('attention mechanism')}")
    print()
    return graph


# ---- Test 3: Path traversal (no embedding — random similarity) -----------
def test_traversal(graph: ContextGraph):
    print("=== Test 3: Cross-document path traversal ===")
    from cross_document_sampling import build_all_paths, EmbeddingIndex
    import numpy as np

    # Random embeddings fallback
    def random_embed(text: str) -> np.ndarray:
        import hashlib
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.random(128).astype(np.float32)

    index = EmbeddingIndex(random_embed)
    paths = build_all_paths(graph, index, depth=1, top_w=2, max_starts_per_entity=2)
    print(f"  Total paths: {len(paths)}")
    if paths:
        sample = paths[0]
        print(f"  Sample path length: {len(sample)}")
        for node in sample:
            print(f"    entity='{node.entity}' para='{node.paragraph.para_id}'")
    print()
    return paths


# ---- Test 4: Serialisation -----------------------------------------------
def test_serialisation(graph: ContextGraph):
    print("=== Test 4: Graph serialisation round-trip ===")
    d = graph.to_dict()
    graph2 = ContextGraph.from_dict(d)
    assert len(graph2.nodes) == len(graph.nodes), "Node count mismatch after round-trip"
    assert len(graph2.edges) == len(graph.edges), "Edge count mismatch after round-trip"
    print("  Round-trip serialisation: OK")
    print()


# --------------------------------------------------------------------------
if __name__ == "__main__":
    test_math_extraction()
    graph = test_graph_construction()
    paths = test_traversal(graph)
    test_serialisation(graph)
    print("All tests passed ✓")
    print()
    print("Next step: set OPENAI_API_KEY and run the full pipeline:")
    print("  python sog_pipeline.py --input_dir data/papers/ --output_dir data/synthetic/ --file_type pdf")
