# Data Generation Methods

This directory contains different approaches for generating synthetic question-answer pairs for RAG evaluation.

## 🔬 Research Tracks

### Multi-Agent Approach (`multi-agent/`)
**Lead:** Aziz Benamira  
**Status:** Active Development  
**Approach:** LangGraph orchestration with 3 specialized agents (Generator, Answerer, Critic) using Reflexion loop and Constitutional AI

**Key Techniques:**
- Reflexion (NeurIPS 2023)
- HyDE (ACL 2023)
- Constitutional AI (Anthropic 2022)
- RAGAS (arXiv 2023)
- Self-RAG (arXiv 2023)

### Other Approaches (Optional Future Work)

#### Graph-Based Generation
- Knowledge graph extraction from documents
- Graph traversal for question generation
- Entity-relationship based QA pairs

#### Evolutionary Methods
- Genetic algorithms for question optimization
- Population-based diversity maintenance
- Fitness function based on quality metrics

#### Prompt Engineering
- Zero-shot and few-shot prompting strategies
- Chain-of-thought for complex reasoning
- Instruction tuning techniques

## 🤝 Collaboration

Each approach should:
1. Document methodology in its own README
2. Use shared utilities from `shared/utils/`
3. Save datasets to `shared/data/datasets/`
4. Use evaluation metrics from `evaluation/metrics/`
5. Follow question taxonomy from `taxonomy/`

## 📊 Comparison Framework

All approaches will be evaluated on:
- Quality (faithfulness, relevance, accuracy)
- Diversity (content, type, difficulty)
- Coverage (document comprehensiveness)
- Cost (API calls, time)
- Scalability

Results will be compared in `docs/benchmarks/generation_comparison.md`
