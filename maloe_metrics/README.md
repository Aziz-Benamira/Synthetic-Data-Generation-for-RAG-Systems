# Metrics Module

Implementation of RAG evaluation metrics covering retriever performance, text generation quality, risk awareness, and system efficiency.

A more in-depth explaination of the metrics is in [bibliography_survey.md](bibliography_survey.md)

## Table of Contents

1. [Evaluator Framework](#evaluator-framework)
2. [Retriever Metrics - Non-Rank Based](#retriever-metrics---non-rank-based)
3. [Retriever Metrics - Rank Based](#retriever-metrics---rank-based)
4. [Generation Metrics - Traditional](#generation-metrics---traditional)
5. [Risk-Aware Metrics](#risk-aware-metrics)
6. [Efficiency Metrics](#efficiency-metrics)
7. [Helper Functions](#helper-functions)

---

## Evaluator Framework

Flexible, extensible architecture for RAG system evaluation with modular, composable evaluators.

### Architecture

The framework consists of specialized evaluators, each handling a specific evaluation dimension:

- **`RetrieverEvaluator`** - Ranking metrics (MRR, NDCG@k, MAP@k)
- **`GenerationEvaluator`** - Text quality (Exact Match, METEOR)
- **`LLMEvaluator`** - Semantic evaluation (context support, relevance, coherence)
- **`PerplexityEvaluator`** - Model confidence and abstention
- **`RiskAwareEvaluator`** - Risk metrics (risk, prudence, alignment, coverage)
- **`EfficiencyEvaluator`** - Performance metrics (latency, cost, ROI)
- **`ComprehensiveEvaluator`** - Orchestrates all evaluators for full pipeline evaluation

### Quick Start

Simple usage with the comprehensive evaluator:

```python
from evaluator import ComprehensiveEvaluator

# Initialize evaluator
evaluator = ComprehensiveEvaluator(llm_model="gpt-4")

# Evaluate a single response
report = evaluator.evaluate_full_pipeline(
    query="What is machine learning?",
    response="Machine learning is a field of AI that enables systems to learn...",
    context="Machine learning involves algorithms that improve through experience...",
    references=["Machine learning is a subset of artificial intelligence..."],
    ttft_ms=150.0,
    total_latency_ms=1200.0
)

# Get results
print(evaluator.to_json(report, pretty=True))
```

### Individual Evaluators

For focused evaluation on specific dimensions:

```python
# Retriever evaluation
from evaluator import RetrieverEvaluator

retriever = RetrieverEvaluator()
result = retriever.evaluate(
    predictions_list=[[1, 2, 3, 5, 8]],  # Ranked doc IDs
    ground_truth_list=[[1, 2, 3]],        # Relevant docs
    k_values=[5, 10]
)
print(result.metrics)  # {'rank_based': {'mrr': 1.0, 'ndcg@5': 0.92, ...}}


# Generation evaluation
from evaluator import GenerationEvaluator

generation = GenerationEvaluator()
result = generation.evaluate(
    predictions=["The capital of France is Paris"],
    references=["Paris is the capital of France"]
)
print(result.metrics)  # {'exact_match': 0.0, 'meteor': 0.85}


# LLM-as-judge evaluation
from evaluator import LLMEvaluator

llm = LLMEvaluator(model="gpt-4")
result = llm.evaluate(
    query="What is ML?",
    response="Machine learning enables...",
    context="ML is a field of AI..."
)
print(result.metrics)  # {'context_support': 0.95, 'relevance': 0.92, ...}
```

### Configuration

Customize evaluators via `EvaluatorConfig`:

```python
from evaluator import ComprehensiveEvaluator, EvaluatorConfig

# Custom configurations
configs = {
    "retriever": EvaluatorConfig(
        name="RetrieverEvaluator",
        additional_params={"k_values": [3, 5, 10]}
    ),
    "perplexity": EvaluatorConfig(
        name="PerplexityEvaluator",
        additional_params={"confidence_threshold": 1.5}
    )
}

evaluator = ComprehensiveEvaluator(config=configs, llm_model="claude-3")
```

### Result Format

All evaluations return `EvaluationResult` objects with standardized structure:

```python
result.evaluator_name      # "RetrieverEvaluator"
result.metrics             # Dict of computed metrics
result.success             # True if evaluation succeeded
result.error_message       # Error details if failed
result.metadata            # Additional info (num_samples, k_values, etc.)
```

---


## Retriever Metrics - Non-Rank Based

These metrics evaluate document retrieval without considering ranking order.

### `unrankedMetrics(predictions, ground_truth)`

**Description:** Computes basic retrieval quality metrics using confusion matrix approach.

**Returns:** Dictionary with:
- `accuracy` - Proportion of correct predictions
- `precision` - Relevant docs retrieved / total docs retrieved
- `recall` - Relevant docs retrieved / total relevant docs
- `f1` - Harmonic mean of precision and recall

**Useful for:** Quick assessment of binary retrieval decisions (relevant/not relevant) without ranking considerations. Good baseline metric.

**Example:**
```python
predictions = [1, 2, 4, 5]
ground_truth = [1, 2, 3]
result = unrankedMetrics(predictions, ground_truth)
# Returns: {'accuracy': 0.67, 'precision': 0.5, 'recall': 0.67, 'f1': 0.57}
```

---

## Retriever Metrics - Rank Based

These metrics consider the ordering of retrieved documents, crucial since users rarely look beyond top results.

### `mean_reciprocal_rank(predictions_list, ground_truth_list)`

**Description:** Measures the average position of the first relevant document across multiple queries.

**Formula:** $\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$

**Returns:** Float between 0-1

**Useful for:** Evaluating answer-seeking scenarios where only the first correct result matters (e.g., fact-checking). Higher scores indicate relevant results appear earlier.

**Example:**
```python
predictions = [[5, 1, 2], [4, 3, 1]]  # 2 queries
ground_truth = [[1, 2], [3, 4]]
mrr = mean_reciprocal_rank(predictions, ground_truth)
# Returns: 0.75 (first result at position 2, second at position 1)
```

### `dcg_at_k(predictions, ground_truth, k=None)`

**Description:** Computes Discounted Cumulative Gain, which rewards relevant documents appearing at higher ranks.

**Formula:** $\text{DCG} = \sum_{i=1}^{k} \frac{\text{relevance}_i}{\log_2(i+1)}$

**Returns:** Float representing cumulative gain (logarithmically discounted by position)

**Useful for:** Building block for NDCG. Quantifies how well relevant items are ranked.

### `idcg_at_k(ground_truth, k=None)`

**Description:** Computes the ideal DCG score - the maximum possible DCG if all relevant items ranked first.

**Returns:** Float representing maximum possible gain for given ground truth

**Useful for:** Normalization factor for NDCG calculation. Enables fair comparison across queries with different numbers of relevant documents.

### `ndcg_at_k(predictions_list, ground_truth_list, k=None)`

**Description:** Normalized DCG - measures ranking quality on a 0-1 scale, heavily rewarding relevant documents at top positions.

**Formula:** $\text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}$

**Returns:** Float between 0-1 (higher is better)

**Useful for:** Primary ranking quality metric. Captures both presence and position of relevant items. Supports @k evaluation (NDCG@5, NDCG@10, etc.).

**Example:**
```python
predictions = [[1, 2, 5], [3, 1, 4]]  # 2 queries
ground_truth = [[1, 2], [3, 4]]
ndcg = ndcg_at_k(predictions, ground_truth, k=10)
# Returns: 1.0 (perfect ranking)
```

### `average_precision_at_k(predictions, ground_truth, k=None)`

**Description:** Computes precision at each relevant position and averages them for a single query.

**Formula:** $\text{AP@k} = \frac{1}{|\text{relevant}|} \sum_{i \in \text{relevant}} \text{precision@i}$

**Returns:** Float between 0-1

**Useful for:** Evaluating how well relevant documents are distributed throughout the ranking.

### `mean_average_precision(predictions_list, ground_truth_list, k=None)`

**Description:** Average of AP scores across multiple queries. Accounts for order of all relevant documents.

**Formula:** $\text{MAP@k} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \text{AP@k}_i$

**Returns:** Float between 0-1

**Useful for:** Standard ranking evaluation metric in information retrieval. More holistic than MRR as it considers all relevant items, not just the first.

**Example:**
```python
predictions = [[1, 2, 5], [3, 1, 4]]
ground_truth = [[1, 2], [3, 4]]
map_score = mean_average_precision(predictions, ground_truth, k=5)
```

---

## Generation Metrics - Traditional

Evaluate text generation quality based on similarity to reference answers.

### `exact_match(predictions, references)`

**Description:** Binary metric - checks if predicted answer exactly matches reference (case-insensitive, whitespace-trimmed).

**Returns:** Float between 0-1 (proportion of exact matches)

**Useful for:** Structured answers (dates, names, numbers). Poor for free-form text due to strict binary nature. Fast to compute, no ML required.

**Example:**
```python
predictions = ["Paris", "London", "Berlin"]
references = ["PARIS", "Paris", "berlin"]
em = exact_match(predictions, references)
# Returns: 1.0 (all match ignoring case)
```

### `meteor(prediction, reference)`

**Description:** Metric for Evaluation of Translation with Explicit Ordering. Blends precision/recall with word order penalty.

**Formula:** $\text{METEOR} = (1-p) \times \frac{(\alpha^2+1) \times \text{Precision} \times \text{Recall}}{\text{Recall} + \alpha \times \text{Precision}}$

**Returns:** Float between 0-1

**Useful for:** Better correlation with human judgments than BLEU/EM for fluency and semantic similarity. Handles synonyms and word order (simplified implementation).

**Example:**
```python
prediction = "The quick brown fox"
reference = "A fast brown fox"
score = meteor(prediction, reference)
# Returns: ~0.7 (partial match with word order difference)
```

### `meteor_batch(predictions, references)`

**Description:** Computes average METEOR score across multiple prediction-reference pairs.

**Returns:** Float between 0-1

**Useful for:** Batch evaluation of multiple answers. Convenient wrapper for multiple queries.

---

## Risk-Aware Metrics

For systems that can abstain from answering to avoid hallucinations.

### `risk_aware_metrics(answerable_kept, unanswerable_kept, unanswerable_discarded, answerable_discarded)`

**Description:** Evaluates quality of abstention decisions. Classifies outcomes into 4 categories:
- **AK (Answerable, Kept):** Correct answer provided (success)
- **UK (Unanswerable, Kept):** Hallucination - wrong answer provided (failure)
- **UD (Unanswerable, Discarded):** Correctly refused (safe)
- **AD (Answerable, Discarded):** Missed opportunity (opportunity cost)

**Returns:** Dictionary with four metrics:

- **`risk`** - Hallucination rate: $\frac{UK}{AK + UK}$ (lower is better)
- **`prudence`** - Detection capability: $\frac{UD}{UK + UD}$ (higher is better)
- **`alignment`** - Decision accuracy: $\frac{AK + UD}{\text{Total}}$ (higher is better)
- **`coverage`** - Answer rate: $\frac{AK + UK}{\text{Total}}$ (depends on use case)

**Useful for:** High-stakes applications where accuracy matters more than coverage. Safety-critical systems that need to control hallucination rates.

**Example:**
```python
metrics = risk_aware_metrics(
    answerable_kept=80,
    unanswerable_kept=10,      # hallucinations
    unanswerable_discarded=70,  # correctly rejected
    answerable_discarded=40     # missed opportunities
)
# risk=0.11, prudence=0.875, alignment=0.833, coverage=0.75
```

---

## Efficiency Metrics

Measure system performance in terms of speed and cost.

### `latency_metrics(ttft_milliseconds, total_latency_milliseconds)`

**Description:** Captures response time characteristics.

**Returns:** Dictionary with:
- `ttft_ms`, `ttft_s` - Time to First Token (user perceives this as response start)
- `total_latency_ms`, `total_latency_s` - Complete generation time

**Useful for:** User experience optimization. TTFT critical for interactive systems. Total latency affects throughput.

**Example:**
```python
metrics = latency_metrics(ttft_milliseconds=150, total_latency_milliseconds=2500)
# Returns: {'ttft_ms': 150, 'ttft_s': 0.15, 'total_latency_ms': 2500, 'total_latency_s': 2.5}
```

### `cost_metrics(input_tokens, output_tokens, cost_per_input_token, cost_per_output_token)`

**Description:** Breaks down financial costs for API-based LLM usage.

**Returns:** Dictionary with:
- `input_cost`, `output_cost` - Itemized costs (output usually more expensive)
- `total_cost` - Sum of input and output costs
- `total_tokens` - Combined token count
- `cost_per_token` - Average cost per token

**Useful for:** Budget tracking, cost optimization, ROI analysis. Essential for production systems using paid APIs.

**Example:**
```python
metrics = cost_metrics(
    input_tokens=500,
    output_tokens=200,
    cost_per_input_token=0.0000015,
    cost_per_output_token=0.000006
)
# Returns: detailed cost breakdown
```

### `retriever_roi(retriever_cost, context_tokens_per_query, llm_cost_per_input_token, num_queries)`

**Description:** Analyzes cost-benefit of using a retriever. Better retrieval = denser context = fewer LLM input tokens.

**Returns:** Dictionary with:
- `retriever_cost` - Total cost of all retrievals
- `llm_context_cost` - Cost of passing context to LLM
- `total_cost` - Sum of both
- `cost_per_query` - Average cost

**Useful for:** Justifying investment in better retrievers. Shows how improving retrieval quality reduces overall system cost through fewer input tokens.

---

## Helper Functions

### `evaluate_retriever(predictions_list, ground_truth_list, k_values=None)`

**Description:** Comprehensive retriever evaluation in one call.

**Returns:** Dictionary containing:
- `non_rank_based` - Accuracy, precision, recall, F1
- `rank_based` - MRR, NDCG@k, MAP@k (for each k in k_values)

**Useful for:** Full pipeline evaluation. Default k_values: [5, 10, 20].

**Example:**
```python
results = evaluate_retriever(predictions, ground_truth)
# Returns all non-rank and rank-based metrics at once
```

### `evaluate_generation(predictions, references)`

**Description:** Comprehensive generation evaluation.

**Returns:** Dictionary containing:
- `exact_match` - EM ratio
- `meteor` - Average METEOR score

**Useful for:** Quick assessment of answer quality across multiple samples.