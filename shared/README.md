# Shared Resources

This directory contains utilities, data, and configurations shared across all research tracks.

## 📁 Structure

```
shared/
├── utils/                  # Shared utility functions
│   ├── llm/               # LLM client wrappers
│   ├── embeddings/        # Embedding utilities
│   ├── evaluation/        # Shared evaluation metrics
│   └── preprocessing/     # Data preprocessing
├── data/                   # Shared datasets
│   ├── raw/               # Raw documents (PDFs, etc.)
│   ├── processed/         # Processed documents
│   └── datasets/          # Final generated datasets
└── configs/                # Shared configurations
    ├── models/            # LLM model configs
    ├── generation/        # Generation configs
    └── evaluation/        # Evaluation configs
```

## 🎯 Purpose

- **Avoid Code Duplication** - Write once, use everywhere
- **Consistency** - Same preprocessing, embeddings, API calls
- **Easy Integration** - All tracks use same utilities
- **Version Control** - Single source of truth

## 🛠️ Shared Utilities

### LLM Clients (`utils/llm/`)
```python
from shared.utils.llm import OpenAIClient, AnthropicClient

client = OpenAIClient(model="gpt-4-turbo")
response = client.generate(prompt="...", temperature=0.7)
```

### Embeddings (`utils/embeddings/`)
```python
from shared.utils.embeddings import get_embeddings

embeddings = get_embeddings(texts, model="text-embedding-3-large")
```

### Evaluation Metrics (`utils/evaluation/`)
```python
from shared.utils.evaluation import calculate_ragas_metrics

scores = calculate_ragas_metrics(
    questions=questions,
    answers=answers,
    contexts=contexts
)
```

## 📊 Shared Data

### Raw Documents (`data/raw/`)
- Store original PDFs, documents here
- Accessible to all tracks
- Not tracked in git (too large)

### Processed Data (`data/processed/`)
- Chunked documents
- Extracted metadata
- Vector embeddings

### Final Datasets (`data/datasets/`)
- Generated QA pairs
- Evaluation results
- Benchmark datasets
- **Will be published to HuggingFace**

## ⚙️ Shared Configs

### Model Configurations (`configs/models/`)
```yaml
# gpt-4-turbo.yaml
model: gpt-4-turbo
temperature: 0.7
max_tokens: 2000
top_p: 0.95
```

### Generation Configurations (`configs/generation/`)
- Prompt templates
- Generation parameters
- Quality thresholds

## 🤝 Contribution Guidelines

### Adding Utilities
1. Create module in appropriate `utils/` subfolder
2. Add docstrings and type hints
3. Write unit tests
4. Update this README

### Using Shared Resources
```python
# Add shared to path
import sys
sys.path.append('../../shared')

# Import utilities
from utils.llm import OpenAIClient
from utils.evaluation import calculate_metrics
```

## 📦 Installation

```bash
# Install shared dependencies
pip install -r ../../requirements.txt

# For development
pip install -e ../../  # Install as editable package
```

## 📝 Best Practices

✅ **DO:**
- Put reusable code here
- Document all functions
- Write tests
- Use type hints

❌ **DON'T:**
- Put track-specific code here
- Hardcode paths or API keys
- Commit large data files
- Break existing APIs without notice
