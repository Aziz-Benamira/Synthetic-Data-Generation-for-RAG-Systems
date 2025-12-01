# Synthetic RAG Evaluation

A research project focused on generating and evaluating synthetic datasets for Retrieval-Augmented Generation (RAG) systems in specialized domains.

## 📋 Overview

This project explores methods for creating synthetic question-answer pairs to evaluate and improve RAG systems, particularly for domain-specific use cases where labeled data is scarce. The generated datasets will be published on HuggingFace.

## 🎯 Objectives

- Review state-of-the-art synthetic data generation methods
- Implement and test various generation approaches
- Evaluate quality of synthetic data (coverage, relevance, diversity)
- Develop novel methods combining different techniques (graph-based, agentic, RL, active learning)
- Assess alignment between synthetic and real-world RAG performance

## 📁 Project Structure

```
Synthetic-Data-Generation-for-RAG-Systems/
├── generation/              # Data generation methods
│   └── multi-agent/        # Multi-agent approach (Aziz)
│       ├── src/            # Source code
│       ├── tests/          # Tests
│       ├── notebooks/      # Experiments
│       └── scripts/        # Utilities
├── evaluation/              # Evaluation metrics research
│   ├── metrics/            # Metric implementations
│   └── benchmarks/         # Benchmark results
├── taxonomy/                # Question taxonomy research
│   ├── question-types/     # Type definitions
│   └── analysis/           # Analysis tools
├── multimodal/              # Multimodal RAG research
│   ├── vision/             # Visual processing
│   └── document-processing/ # Multimodal documents
├── shared/                  # Shared resources
│   ├── utils/              # Shared utilities
│   ├── data/               # Shared datasets
│   └── configs/            # Shared configurations
├── docs/                    # Documentation
│   ├── architecture/       # Architecture docs
│   ├── tutorials/          # Tutorials
│   └── a2a_learning/       # A2A materials
└── .github/                 # CI/CD workflows
```

## 🔬 Research Tracks

### 1. **Data Generation** (`generation/`)
Multiple approaches for generating synthetic QA pairs:
- **Multi-Agent** (Aziz) - LangGraph orchestration with Reflexion, HyDE, Constitutional AI
- **Graph-Based** - Knowledge graph approaches
- **Evolutionary** - Genetic algorithms

### 2. **Evaluation Metrics** (`evaluation/`)
Developing and testing RAG evaluation metrics:
- RAGAS extensions
- LLM-as-judge metrics
- Multimodal metrics
- Synthetic data quality metrics

### 3. **Question Taxonomy** (`taxonomy/`)
Classification and analysis of question types:
- Question type definitions
- Automatic classifiers
- Distribution analysis
- Bloom's taxonomy alignment

### 4. **Multimodal RAG** (`multimodal/`)
Extending to text + visual documents:
- Image extraction from PDFs
- Vision-Language Models
- Multimodal question generation
- Visual grounding evaluation

## 🚀 Getting Started

### For All Team Members

```bash
# Clone the repository
git clone https://github.com/Aziz-Benamira/Synthetic-Data-Generation-for-RAG-Systems.git
cd Synthetic-Data-Generation-for-RAG-Systems

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

### Navigate to Your Research Track

```bash
# For generation research
cd generation/multi-agent/

# For evaluation research
cd evaluation/

# For taxonomy research
cd taxonomy/

# For multimodal research
cd multimodal/
```

Each directory has its own README with specific instructions.

## 📚 Documentation

### Architecture & Planning
- [Technical Report](docs/architecture/TECHNICAL_REPORT.md) - Complete system design
- [Project Roadmap](docs/architecture/PROJECT_ROADMAP.md) - 16-week implementation plan
- [System Architecture](docs/architecture/ARCHITECTURE.md) - Architecture details

### Collaboration
- [Collaboration Guide](docs/COLLABORATION_GUIDE.md) - **READ THIS FIRST!**
- [Contributing Guidelines](CONTRIBUTING.md) - Code standards
- [Research Track READMEs](generation/) - Track-specific guides

## 🤝 Team Collaboration

This is a **multi-track research project**. Each team member focuses on a specific area:

- **Folder-based organization** - Each track has its own directory
- **Feature branches** - Create branches for each feature
- **Shared resources** - Common utilities in `shared/`
- **Regular integration** - Combine approaches in final phase

**See [COLLABORATION_GUIDE.md](docs/COLLABORATION_GUIDE.md) for detailed workflow!**

## 📄 License

TBD

## 👥 Team

Research team - Academic Project

## 📧 Contact

For questions and collaboration: 
aziz.ben-amira@ensta-paris.fr
ameni.hidouri@ensta-paris.fr
maloe.aymonier@ensta-paris.fr
yassine.zanned@ensta-paris.fr
seifeddine.ghozzi@ensta-paris.fr
