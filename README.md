# Synthesize-on-Graph (SoG) Implementation

A Python implementation of the **Synthesize-on-Graph (SoG)** framework for generating synthetic data to improve and evaluate Retrieval-Augmented Generation (RAG) systems.

Based on the paper: *Synthesize-on-Graph: Knowledgeable Synthetic Data Generation for Continue Pre-training of Large Language Models* by Jiang et al. (2025).

## Overview

The SoG framework addresses the challenge of generating high-quality synthetic datasets for RAG system evaluation by:

- **Context Graph Construction**: Building a graph where nodes represent entities and edges represent cross-document knowledge associations
- **Two-Stage Sampling**: Combining BFS traversal with similarity-based selection and secondary sampling for long-tail entity mitigation
- **Dual Generation Strategies**:
  - **Chain-of-Thought (CoT)**: Creates step-by-step narratives connecting fragments across documents
  - **Contrastive Clarifying (CC)**: Generates comparative analyses for sparse entities with limited connections

## Features

✅ Context graph construction with cross-document entity relationships  
✅ BFS-based path sampling with similarity scoring  
✅ Adaptive strategy selection (CoT vs CC)  
✅ Long-tail entity distribution balancing  
✅ Multi-hop reasoning question generation  
✅ Comprehensive statistics and analysis  
✅ Flexible configuration system  
✅ Extensible architecture for custom LLM integration

## Installation

### Requirements

```bash
pip install networkx numpy tqdm
```

### Optional Dependencies

For advanced entity extraction:
```bash
# spaCy-based extraction
pip install spacy
python -m spacy download en_core_web_sm

# For actual LLM integration (examples)
pip install openai anthropic  # Choose your LLM provider
```

## Quick Start

### Basic Usage

```python
from sog_pipeline import SynthesizeOnGraph, SoGConfig, Document

# Create documents
documents = [
    Document(
        doc_id=0,
        title="Introduction to ML",
        paragraphs=["Machine learning is...", "Deep learning uses..."],
        entities_per_paragraph=[["Machine Learning", "AI"], ["Deep Learning", "Neural Networks"]]
    ),
    # Add more documents...
]

# Configure and run pipeline
config = SoGConfig(
    max_depth=3,
    num_samples=100,
    output_path="synthetic_data.jsonl"
)

sog = SynthesizeOnGraph(config=config)
samples = sog.run_pipeline(documents)

# Each sample contains: question, answer, context, reasoning_type, strategy_used
```

### Running Examples

```bash
python example_usage.py
```

This will run 5 different examples demonstrating various features of the framework.

## Architecture

### Core Modules

1. **`context_graph.py`**: Context graph construction and entity-paragraph mapping
2. **`sampling_strategy.py`**: Two-stage sampling with BFS traversal and secondary sampling
3. **`generation_strategies.py`**: CoT and CC generation strategies with adaptive selection
4. **`sog_pipeline.py`**: Main orchestrator integrating all components
5. **`entity_extraction.py`**: Entity extraction utilities (simple, spaCy, or LLM-based)
6. **`text_processing.py`**: Text preprocessing and document handling utilities

### Pipeline Flow

```
Documents → Entity Extraction → Context Graph Construction
                                        ↓
                              BFS Traversal Sampling
                                        ↓
                              Secondary Sampling (Long-tail boost)
                                        ↓
                          Adaptive Strategy Selection (CoT/CC)
                                        ↓
                            Synthetic Sample Generation
                                        ↓
                              Output: Q&A Pairs + Context
```

## Configuration Options

```python
SoGConfig(
    # Sampling parameters
    max_depth=3,              # Maximum BFS depth (1-3 hops)
    top_k_neighbors=5,        # Top-k neighbors to consider at each step
    long_tail_boost=1.5,      # Boost factor for long-tail entities
    
    # Generation parameters
    sparse_threshold=3,       # Threshold for sparse entity detection
    utilization_threshold=0.1, # Threshold for low utilization rate
    
    # LLM parameters
    max_tokens=1500,          # Max tokens for generation
    temperature=0.7,          # Sampling temperature
    
    # Output parameters
    num_samples=100,          # Number of samples to generate
    random_seed=42,           # Random seed for reproducibility
    output_path="output.jsonl", # Output file path
    
    # Processing
    batch_size=10,            # Batch size for processing
    verbose=True              # Enable progress logging
)
```

## Extending with Real LLMs

The implementation includes mock LLM interfaces for demonstration. To integrate with real LLMs:

### OpenAI Integration

```python
from generation_strategies import LLMInterface
import openai

class OpenAILLM(LLMInterface):
    def __init__(self, api_key, model="gpt-4"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def generate(self, prompt, max_tokens=1000, temperature=0.7):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content

# Use in pipeline
llm = OpenAILLM(api_key="your-key")
sog = SynthesizeOnGraph(llm=llm)
```

### Anthropic (Claude) Integration

```python
import anthropic

class ClaudeLLM(LLMInterface):
    def __init__(self, api_key, model="claude-3-sonnet-20240229"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
    
    def generate(self, prompt, max_tokens=1000, temperature=0.7):
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

llm = ClaudeLLM(api_key="your-key")
sog = SynthesizeOnGraph(llm=llm)
```

## Input Data Format

### JSON Format

```json
[
  {
    "doc_id": 0,
    "title": "Document Title",
    "paragraphs": ["Paragraph 1 text...", "Paragraph 2 text..."],
    "entities_per_paragraph": [["Entity1", "Entity2"], ["Entity3", "Entity4"]]
  }
]
```

### JSONL Format (one document per line)

```jsonl
{"doc_id": 0, "title": "Doc 1", "paragraphs": [...], "entities_per_paragraph": [...]}
{"doc_id": 1, "title": "Doc 2", "paragraphs": [...], "entities_per_paragraph": [...]}
```

### Loading Documents

```python
from sog_pipeline import load_documents_from_json, load_documents_from_jsonl

# From JSON
documents = load_documents_from_json("corpus.json")

# From JSONL
documents = load_documents_from_jsonl("corpus.jsonl")
```

## Output Format

Generated samples are saved in JSONL format:

```json
{
  "question": "How do the concepts of machine learning and deep learning relate?",
  "answer": "Let me explain step by step: 1. First, machine learning...",
  "context": ["Para 1 text", "Para 2 text"],
  "reasoning_type": "multi-hop",
  "strategy": "cot"
}
```

## Entity Extraction

### Simple Pattern-Based

```python
from entity_extraction import SimpleEntityExtractor

extractor = SimpleEntityExtractor()
entities = extractor.extract_entities("Machine learning is a subset of AI.")
# Returns: ["Machine", "AI"]
```

### spaCy-Based

```python
from entity_extraction import SpaCyEntityExtractor

extractor = SpaCyEntityExtractor(model_name="en_core_web_sm")
entities = extractor.extract_entities("Apple Inc. is based in Cupertino.")
# Returns: ["Apple Inc.", "Cupertino"]
```

### LLM-Based

```python
from entity_extraction import LLMEntityExtractor

extractor = LLMEntityExtractor(llm_interface=your_llm)
entities = extractor.extract_entities(text)
```

## Examples

The `example_usage.py` script demonstrates:

1. **Basic Usage**: Default configuration with sample documents
2. **Custom Configuration**: Adjusting parameters for different scenarios
3. **Text File Processing**: Loading and processing raw text files
4. **Incremental Processing**: Adding documents in batches
5. **Analysis Only**: Analyzing corpus without generation

## Performance Considerations

- **Graph Construction**: O(N×E) where N is number of paragraphs, E is entities per paragraph
- **BFS Traversal**: O(V + E) per root node where V is entities, E is edges
- **Generation**: Depends on LLM latency; can be parallelized

### Optimization Tips

1. Use efficient embedding models for similarity computation
2. Cache embeddings to avoid recomputation
3. Adjust `max_depth` and `top_k_neighbors` to balance quality vs. speed
4. Process documents in batches for large corpora
5. Use GPU-accelerated models for entity extraction and embeddings

## Validation & Testing

```python
# Get statistics
stats = sog.get_statistics()
print(f"Total samples: {stats['generation']['total_samples_generated']}")
print(f"CoT ratio: {stats['generation']['cot_samples'] / stats['generation']['total_samples_generated']}")

# Analyze graph
graph_stats = stats['graph']
print(f"Graph density: {graph_stats['density']}")
print(f"Connected components: {graph_stats['num_connected_components']}")
```

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{jiang2025synthesize,
  title={Synthesize-on-Graph: Knowledgeable Synthetic Data Generation for Continue Pre-training of Large Language Models},
  author={Jiang, Xuhui and Ma, Shengjie and Xu, Chengjin and Yang, Cehao and Zhang, Liyu and Guo, Jian},
  journal={arXiv preprint arXiv:2505.00979},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Areas for improvement:

- Integration with additional LLM providers
- Advanced entity extraction methods
- Embedding model optimizations
- Multi-modal support (images, tables)
- Evaluation metrics for synthetic data quality
- Support for additional languages

## Support

For issues, questions, or suggestions, please open an issue on GitHub.

## Acknowledgments

Implementation based on the SoG framework by Jiang et al. (2025). Thanks to the authors for their innovative approach to synthetic data generation for RAG systems.
