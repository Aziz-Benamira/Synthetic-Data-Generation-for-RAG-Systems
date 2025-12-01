# Question Taxonomy Research

**Research Focus:** Classifying and analyzing question types for RAG evaluation

## 🎯 Objectives

1. **Literature Review** - Survey existing question taxonomies
2. **Taxonomy Development** - Create comprehensive classification system
3. **Analysis Tools** - Build tools to analyze question distributions
4. **Guidelines** - Provide generation guidelines for balanced datasets

## 📁 Structure

```
taxonomy/
├── question-types/       # Question type definitions and examples
│   ├── factoid/         # Simple fact retrieval
│   ├── definition/      # Concept definitions
│   ├── comparison/      # Comparing entities/concepts
│   ├── reasoning/       # Multi-step reasoning
│   ├── calculation/     # Mathematical problems
│   ├── application/     # Applying concepts
│   └── analysis/        # Critical analysis
└── analysis/            # Analysis tools and results
    ├── classifiers/     # Automatic classification
    ├── statistics/      # Distribution analysis
    └── visualizations/  # Question type visualizations
```

## 📚 Taxonomy Dimensions

### 1. **Question Complexity**
- **Single-hop** - Answer in one document chunk
- **Multi-hop** - Requires multiple chunks
- **Reasoning** - Requires inference/deduction

### 2. **Question Type**
- Factoid (Who, What, When, Where)
- Definition (What is...)
- Comparison (Difference between...)
- Reasoning (Why, How)
- Calculation (Numerical problems)
- Application (Apply concept to scenario)
- Analysis (Evaluate, critique)

### 3. **Cognitive Level (Bloom's Taxonomy)**
- Remember (recall facts)
- Understand (explain concepts)
- Apply (use in new situations)
- Analyze (break down, compare)
- Evaluate (judge, critique)
- Create (synthesize, design)

### 4. **Domain Specificity**
- Generic (transferable across domains)
- Domain-specific (requires specialized knowledge)

## 🔬 Research Questions

1. What question types are most common in academic domains?
2. How to ensure balanced distribution of question types?
3. What types best evaluate different RAG capabilities?
4. How do question types differ across domains (physics vs law)?

## 🤝 Integration Points

- Provide taxonomy to `generation/` for balanced generation
- Use datasets from `generation/` for distribution analysis
- Coordinate with `evaluation/` on type-specific metrics
- Extend taxonomy for `multimodal/` questions

## 📊 Deliverables

- [ ] Comprehensive taxonomy document
- [ ] Question type classifier
- [ ] Distribution analysis tool
- [ ] Balanced generation guidelines
- [ ] Type-specific evaluation rubrics

## 🚀 Getting Started

```bash
# Install dependencies
pip install -r ../../requirements.txt
pip install scikit-learn matplotlib seaborn

# Analyze dataset
python analysis/analyze_dataset.py --dataset path/to/dataset.jsonl

# Classify questions
python question-types/classifier.py --input questions.txt
```

## 📚 Key Papers to Review

- Question taxonomies in QA systems
- Bloom's taxonomy applications
- SQuAD, Natural Questions taxonomies
- Educational question classification
