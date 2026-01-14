# Synthesize-on-Graph (SoG) Implementation & Research Report

This workspace contains a comprehensive implementation of the **Synthesize-on-Graph (SoG)** framework for generating synthetic data to improve and evaluate Retrieval-Augmented Generation (RAG) systems, along with a bibliographical research report on the topic.

## 📁 Workspace Structure

```
ProjetIALatexContainer/
├── 📄 LaTeX Report Files
│   ├── main.tex                    # LaTeX source for the bibliographical report
│   ├── references.bib              # Bibliography entries (SoG paper, EntiGraph)
│   └── README.md (this file)       # Complete project documentation
│
├── 🐍 SoG Framework Implementation
│   ├── context_graph.py            # Context graph construction module
│   ├── sampling_strategy.py        # Two-stage sampling strategy
│   ├── generation_strategies.py    # CoT & CC generation strategies
│   ├── sog_pipeline.py            # Main pipeline orchestrator
│   ├── entity_extraction.py        # Entity extraction utilities
│   ├── text_processing.py         # Text preprocessing utilities
│   └── example_usage.py           # Usage examples & demonstrations
│
├── 📚 Documentation & Config
│   ├── README_SoG.md              # Detailed SoG implementation guide
│   ├── requirements.txt            # Python dependencies
│   └── LICENSE                     # MIT License
│
└── 📊 Output (generated when running)
    └── output/                     # Generated synthetic data files
        ├── synthetic_data_basic.jsonl
        ├── synthetic_data_custom.jsonl
        └── ...
```

---

## 📖 Project Overview

### The Research Report

**File**: `main.tex`

A LaTeX document that examines graph-based synthetic data generation methods for improving and evaluating RAG systems, with a focus on the Synthesize-on-Graph (SoG) framework by Jiang et al. (2025).

**Key Topics Covered:**
- Evolution from intra-document (EntiGraph) to cross-document methods
- Context graph construction with cross-document knowledge associations
- Two-stage sampling strategy (BFS traversal + secondary sampling)
- Dual generation strategies (Chain-of-Thought & Contrastive Clarifying)
- Experimental validation on MultiHop-RAG benchmark
- Implications for RAG system evaluation

**Compile the report:**
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use your preferred LaTeX editor/compiler.
