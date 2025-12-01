# Multi-Agent RAG System for Synthetic Data Generation

A sophisticated multi-agent system for generating high-quality Question-Answer-Context triplets from academic textbooks using state-of-the-art research techniques.

## Overview

This project implements a RAG (Retrieval-Augmented Generation) benchmarking system that combines five cutting-edge research papers:
- **Reflexion** (NeurIPS 2023) - Multi-agent cooperative system with self-improvement
- **HyDE** (ACL 2023) - Query expansion via hypothetical document generation
- **Constitutional AI** (Anthropic 2022) - Academic validation principles
- **RAGAS** (arXiv 2023) - Automated evaluation metrics
- **Self-RAG** (arXiv 2023) - Self-assessment and pre-filtering

## Project Structure

```
.
├── src/                    # Source code
│   ├── agents/            # Agent implementations
│   ├── mcp/               # Model Context Protocol server
│   ├── evaluation/        # RAGAS integration
│   ├── utils/             # Utility functions
│   ├── config.py          # Configuration management
│   ├── pdf_processor.py   # PDF text extraction
│   └── vector_store.py    # ChromaDB integration
├── tests/                  # Test suite
├── scripts/                # Utility scripts
├── notebooks/              # Jupyter experiments
├── data/                   # PDF textbooks (gitignored)
├── outputs/                # Generated datasets
├── docs/                   # Documentation
│   ├── architecture/      # Architecture documents
│   ├── tutorials/         # Implementation guides
│   └── a2a_learning/      # A2A protocol resources
├── IMPLEMENTATION_PLAN.md  # 16-week development roadmap
├── GETTING_STARTED.md      # Quick start guide
├── CURRENT_STATUS.md       # Project progress tracker
└── requirements.txt        # Python dependencies
```

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# Required: OPENAI_API_KEY
```

### 2. Add a PDF Textbook

Download a free academic textbook and place it in the data directory:
```bash
# Recommended: OpenStax textbooks (https://openstax.org/)
# Save as: data/sample_textbook.pdf
```

### 3. Test PDF Processing

```bash
python scripts/test_pdf_processor.py
```

### 4. Follow the Implementation Plan

See `IMPLEMENTATION_PLAN.md` for the complete 16-week roadmap.

Start with `GETTING_STARTED.md` for Phase 1 instructions.

## Current Status

**Phase:** 1 - Foundation  
**Task:** PDF Processing  
**Branch:** Aziz_branch

See `CURRENT_STATUS.md` for detailed progress tracking.

## Development Approach

This project is built incrementally with focus on understanding:
1. Each component is built and tested individually
2. No moving forward until current component is mastered
3. Comprehensive testing at every step
4. Clear documentation for every decision

## Documentation

- **IMPLEMENTATION_PLAN.md** - Complete 16-week development roadmap
- **GETTING_STARTED.md** - Phase 1 step-by-step guide
- **CURRENT_STATUS.md** - Progress tracking and next steps
- **TECHNICAL_REPORT.md** - Complete technical architecture
- **docs/** - Additional documentation and learning resources

## Key Features (Planned)

### Phase 1: Foundation
- PDF text extraction with structure preservation
- Semantic chunking strategy
- Vector store with ChromaDB
- MCP server for data access

### Phase 2-4: Agents
- Question Generator with diversity management
- Answer Generator with HyDE and Self-RAG
- Critic Agent with RAGAS and Constitutional AI

### Phase 5: Reflexion Loop
- Iterative improvement through feedback
- Multi-turn refinement
- Quality convergence tracking

### Phase 6: Orchestration
- LangGraph state machine
- SQLite checkpointing for crash recovery
- Batch processing with progress tracking

### Phase 7: Advanced Features
- Multi-dimensional diversity tracking
- Hard negative generation
- Advanced semantic chunking

### Phase 8: Evaluation
- Baseline comparisons
- Performance benchmarking
- Final dataset generation

## Technologies

- **LLM**: OpenAI GPT-4 Turbo
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector Store**: ChromaDB
- **Orchestration**: LangGraph
- **Evaluation**: RAGAS
- **PDF Processing**: PyMuPDF
- **Data Layer**: Model Context Protocol (MCP)

## License

TBD

## Contact

Research project - Academic collaboration
