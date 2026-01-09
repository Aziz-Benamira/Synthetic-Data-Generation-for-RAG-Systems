# Synthesize-on-Graph (SoG) for RAG Systems

This report examines graph-based synthetic data generation methods for improving and evaluating Retrieval-Augmented Generation (RAG) systems, with a focus on the Synthesize-on-Graph (SoG) framework.

## Overview

RAG systems face significant evaluation challenges due to limited labeled data in specialized domains, high annotation costs, and the need for diverse question types. This report explores how graph-based approaches can address these challenges by generating high-quality synthetic datasets.

## Key Content

### The Problem with Traditional Approaches

Early methods like EntiGraph focused on intra-document relationships, decomposing text into entity lists and generating descriptions within single documents. This approach had critical limitations:
- Inability to capture cross-document knowledge associations
- Limited content diversity and knowledge depth
- Poor performance on multi-hop reasoning tasks

### Synthesize-on-Graph (SoG) Framework

The report presents SoG (Jiang et al., 2025), a context-graph-enhanced framework that overcomes these limitations through:

**1. Context Graph Construction**
- Builds a graph where nodes represent entities and edges represent cross-document knowledge associations
- Enables sophisticated two-stage sampling with BFS traversal and similarity-based selection
- Addresses long-tail entity distribution through secondary sampling

**2. Dual Generation Strategies**
- **Chain-of-Thought (CoT)**: Creates step-by-step narratives connecting fragments across documents with logical flow
- **Contrastive Clarifying (CC)**: Generates comparative analyses for sparse entities with limited connections

**3. Key Innovations**
- Entity-context mapping for efficient cross-document relationship discovery
- Multi-hop path construction capturing contextually connected knowledge
- Adaptive strategy selection optimizing for both common and rare knowledge elements

### Experimental Results

SoG demonstrated substantial improvements on the MultiHop-RAG benchmark, outperforming EntiGraph with performance gains up to 9× synthetic data volume. The framework successfully transforms long-tail entity distributions into balanced, near-normal distributions.

## Author

Yassine Zanned
