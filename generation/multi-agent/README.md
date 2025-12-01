# Multi-Agent Generation Approach

**Research Lead:** Aziz Benamira

## 🎯 Objective

Develop a sophisticated multi-agent system for generating high-quality Question-Answer-Context triplets from specialized domain documents using:
- **Reflexion** (NeurIPS 2023) - Multi-agent cooperative system
- **HyDE** (ACL 2023) - Query expansion via hypothetical documents
- **Constitutional AI** (Anthropic 2022) - Academic validation principles
- **RAGAS** (arXiv 2023) - Automated evaluation metrics
- **Self-RAG** (arXiv 2023) - Self-assessment and pre-filtering

## 📁 Structure

```
multi-agent/
├── src/                    # Source code
│   ├── agents/            # Question Generator, Answer Generator, Critic
│   ├── mcp/               # Model Context Protocol server
│   ├── evaluation/        # RAGAS integration
│   └── utils/             # Utilities
├── tests/                  # Test suite
├── notebooks/              # Jupyter experiments
├── scripts/                # Utility scripts
└── outputs/                # Generated outputs
```

## 📚 Documentation

- [Technical Report](../../docs/architecture/TECHNICAL_REPORT.md)
- [Implementation Roadmap](../../docs/architecture/PROJECT_ROADMAP.md)
- [System Architecture](../../docs/architecture/ARCHITECTURE.md)

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r ../../requirements.txt

# Set up environment
cp ../../.env.example .env
# Edit .env with your API keys

# Run generation pipeline (coming soon)
python scripts/run_pipeline.py
```

## 📊 Current Status

- ✅ Architecture designed (5 research papers integrated)
- ✅ MCP server implemented
- ✅ PDF processor and vector store ready
- ⏳ Agent implementations (following 16-week roadmap)

## 🤝 Integration Points

This approach will be compared with:
- Other generation methods (graph-based, evolutionary)
- Evaluation metrics from `evaluation/` team
- Question taxonomy from `taxonomy/` team
- Multimodal extensions from `multimodal/` team
