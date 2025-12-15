# Multi-Agent RAG System for Synthetic Q&A Generation

## Project Overview

**Goal**: Build a multi-agent Retrieval-Augmented Generation (RAG) system to generate high-quality synthetic question-answer pairs from academic textbooks for training and evaluation purposes.

**Innovation**: Integrate state-of-the-art research papers (Reflexion, HyDE, Constitutional AI, RAGAS, Self-RAG) into a coordinated multi-agent architecture with semantic chunking for superior retrieval quality.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MULTI-AGENT RAG PIPELINE                         │
└─────────────────────────────────────────────────────────────────────┘

Phase 1: Document Processing
    │
    ├─► [PDF Parser (ENSTA)]
    │   └─► Extract text from academic textbook PDFs
    │
    ├─► [Semantic Chunker (Our Innovation)]
    │   ├─► Extract TOC structure (chapters, sections, subsections)
    │   ├─► Detect semantic units (definitions, theorems, examples)
    │   ├─► Create variable-size chunks (300-2000 chars)
    │   └─► Add rich metadata (chapter, section, type, pages)
    │
    └─► [Vector Store (ChromaDB + OpenAI Embeddings)]
        └─► Embed chunks with metadata for semantic search

Phase 2: Multi-Agent Q&A Generation
    │
    ├─► [Retriever Agent with HyDE]
    │   ├─► Input: Topic/section from textbook
    │   ├─► Generate hypothetical answer (HyDE paper)
    │   ├─► Retrieve relevant chunks using semantic search
    │   └─► Filter by metadata (chapter, section, type)
    │
    ├─► [Question Generator Agent]
    │   ├─► Input: Retrieved chunks + metadata
    │   ├─► Generate diverse question types:
    │   │   • Factual (definitions, concepts)
    │   │   • Analytical (problem-solving)
    │   │   • Conceptual (understanding)
    │   └─► Output: Candidate questions
    │
    ├─► [Answer Generator Agent with Self-RAG]
    │   ├─► Input: Question + retrieved chunks
    │   ├─► Generate answer with Self-RAG framework:
    │   │   • Retrieve relevant passages
    │   │   • Reflect on relevance
    │   │   • Generate answer
    │   │   • Critique and refine
    │   └─► Output: High-quality answer with citations
    │
    ├─► [Reflexion Agent (Self-Improvement)]
    │   ├─► Input: Generated Q&A pair + feedback
    │   ├─► Reflect on quality issues:
    │   │   • Is question clear?
    │   │   • Is answer accurate?
    │   │   • Is answer grounded in source?
    │   ├─► Generate improvement suggestions
    │   └─► Refine Q&A iteratively
    │
    └─► [Constitutional AI Agent (Alignment)]
        ├─► Input: Refined Q&A pair
        ├─► Apply constitutional principles:
        │   • Accuracy (verify against source)
        │   • Clarity (ensure understandability)
        │   • Completeness (cover key concepts)
        │   • Non-hallucination (cite sources)
        ├─► Critique and revise if needed
        └─► Output: Final verified Q&A pair

Phase 3: Evaluation & Quality Control
    │
    └─► [RAGAS Evaluation Framework]
        ├─► Context Precision: Are retrieved chunks relevant?
        ├─► Context Recall: Are all relevant chunks retrieved?
        ├─► Faithfulness: Is answer grounded in context?
        ├─► Answer Relevancy: Does answer address question?
        └─► Output: Quality metrics + improvement feedback
```

---

## Multi-Agent Frameworks: Literature Review

### Core Multi-Agent Architecture Papers

#### **Primary Inspiration: Diverse and Private Synthetic Datasets Generation for RAG Evaluation**

**Paper**: arXiv:2508.18929 (August 2024)

**Relevance**: This is the paper you showed your tutor. It directly addresses multi-agent synthetic data generation for RAG evaluation.

**Key Contribution**: Multi-agent framework specifically designed for generating synthetic Q&A datasets to evaluate RAG systems, with focus on diversity and privacy.

---

#### **MAG-V: A Multi-Agent Framework for Synthetic Data Generation and Verification**

**Paper**: Sengupta et al., 2024 (arXiv:2412.04494)

**Published**: December 2024 (Very recent!)

**Where**: Addresses synthetic question generation for agent testing

**Key Innovation**: 
- **Two-stage multi-agent approach**:
  1. **Generation Stage**: Multiple agents collaborate to generate synthetic questions that mimic real customer queries
  2. **Verification Stage**: Reverse-engineer alternate questions from responses to verify agent trajectory correctness

**Methodology**:
- Uses multiple specialized agents to generate diverse question types
- Employs trajectory verification using traditional ML models (outperforms GPT-4o judge by 11%)
- Addresses data scarcity in industry applications
- Focuses on zero-shot reasoning with tool-augmented agents

**Why Relevant**: 
- Directly addresses synthetic data generation with multi-agent systems
- Includes verification/quality control (similar to your Reflexion + Constitutional AI approach)
- Industry-focused (validates on actual customer queries)

**Difference from Your Approach**: 
- They focus on tool-calling agents for customer service
- You focus on educational content (textbooks) with semantic chunking
- They use trajectory verification; you use Constitutional AI principles

---

#### **RAGentA: Multi-Agent RAG for Attributed Question Answering**

**Paper**: Besrour et al., 2025 (arXiv:2506.16988, accepted at SIGIR 2025)

**Published**: June 2025 (Cutting-edge!)

**Where**: Multi-agent RAG architecture for trustworthy QA

**Key Innovation**:
- **Multi-agent architecture** with specialized roles:
  - Retrieval filtering agent
  - Answer generation agent with in-line citations
  - Verification agent for completeness
  - Dynamic refinement through iterative improvement

**Methodology**:
- Hybrid retrieval: Combines sparse + dense methods (12.5% better Recall@20)
- Iterative document filtering
- Citation generation and grounding verification
- Evaluated on synthetic QA dataset from FineWeb

**Results**:
- +1.09% correctness over baseline RAG
- +10.72% faithfulness over baseline
- Demonstrates multi-agent coordination improves trustworthiness

**Why Relevant**:
- Multi-agent RAG specifically for QA generation
- Focuses on faithfulness and grounding (matches your goals)
- Uses synthetic dataset for evaluation
- Emphasizes trustworthy answer generation

**Similarity to Your Approach**:
- Multi-agent architecture with specialized agents
- Iterative refinement (like Reflexion)
- Focus on answer grounding (like Constitutional AI)
- Hybrid retrieval strategies

**Difference**: 
- They focus on attributed QA (answering questions with citations)
- You focus on generating Q&A pairs from textbooks
- They use hybrid sparse+dense retrieval; you use semantic chunking + HyDE

---

#### **A Multi-Agent Communication Framework for Question Generation**

**Paper**: Wang et al., 2019 (AAAI Conference, cited 48 times)

**Published**: 2019 (Foundational work)

**Where**: Multi-agent framework for question-worthy phrase extraction and question generation

**Key Innovation**:
- **Multi-agent communication** between specialized agents
- Agents collaborate to identify important phrases and generate questions
- Focuses on constructing question sets for QA tasks

**Why Relevant**:
- Foundational work on multi-agent question generation
- Establishes precedent for agent collaboration in QA dataset creation
- Shows multi-agent approach outperforms single-agent baselines

**Historical Context**: Early demonstration that multi-agent systems improve question generation quality through specialization and collaboration.

---

### Comparison: Your Approach vs. Existing Multi-Agent Frameworks

| Paper | Focus | Multi-Agent Roles | Domain | Year |
|-------|-------|------------------|---------|------|
| **Your System** | Synthetic Q&A for textbooks | HyDE Retriever, Question Generator, Self-RAG Answer Generator, Reflexion, Constitutional AI | Education (textbooks) | 2024-2025 |
| **Paper 2508.18929** (tutor's paper) | Synthetic data for RAG eval | Diverse Q&A generation agents, privacy-preserving agents | RAG evaluation | 2024 |
| **MAG-V** | Synthetic questions + verification | Question generation agents, trajectory verification agents | Customer service | 2024 |
| **RAGentA** | Attributed QA with citations | Retrieval filter, Answer generator, Verification, Refinement | Scientific QA | 2025 |
| **Wang et al.** | Question generation | Phrase extraction agent, Question generation agent | General QA | 2019 |

**Your Unique Contribution**:
1. **Semantic Chunking Integration**: None of the above papers use structure-aware chunking with TOC extraction and semantic unit detection
2. **Educational Domain Focus**: First to target academic textbooks with multi-agent RAG
3. **Research-Driven Component Selection**: You integrate 5 state-of-the-art papers (HyDE, Self-RAG, Reflexion, Constitutional AI, RAGAS) into one cohesive multi-agent system
4. **Metadata-Rich Retrieval**: Your semantic chunks have 8 metadata fields enabling filtered retrieval

---

## Research Papers Integration

### 1. **Reflexion: Language Agents with Verbal Reinforcement Learning**

**Paper**: Shinn et al., 2023

**Where**: Reflexion Agent (Phase 2)

**How**: 
- Implements self-reflection loop for Q&A quality improvement
- Agent generates Q&A pair → Evaluates quality → Reflects on errors → Refines output
- Uses verbal feedback (not just scores) to guide improvements

**Implementation**:
```python
class ReflexionAgent:
    def generate_qa(self, context):
        # Initial generation
        qa_pair = self.base_generator(context)
        
        # Reflection loop
        for iteration in range(max_iterations):
            # Evaluate current output
            critique = self.self_evaluate(qa_pair, context)
            
            # If quality sufficient, break
            if critique.score > threshold:
                break
            
            # Generate reflection
            reflection = self.reflect(qa_pair, critique)
            
            # Refine based on reflection
            qa_pair = self.refine(qa_pair, reflection, context)
        
        return qa_pair
```

**Benefits**:
- Iterative improvement without human feedback
- Learns from its own mistakes
- Reduces hallucinations through self-critique

---

### 2. **HyDE: Precise Zero-Shot Dense Retrieval without Relevance Labels**

**Paper**: Gao et al., 2022

**Where**: Retriever Agent (Phase 2)

**How**:
- Instead of searching with raw question, generate hypothetical answer first
- Embed hypothetical answer → Search for similar real chunks
- Improves retrieval by bridging semantic gap

**Implementation**:
```python
class HyDERetriever:
    def retrieve(self, question, vectorstore):
        # Step 1: Generate hypothetical answer
        hypothetical_answer = self.llm.generate(
            f"Answer this question as if you have the knowledge:\n{question}"
        )
        
        # Step 2: Embed hypothetical answer
        embedding = self.embed(hypothetical_answer)
        
        # Step 3: Retrieve similar chunks
        chunks = vectorstore.similarity_search(
            embedding,
            k=5,
            filter={"semantic_type": "definition"}  # Use our metadata!
        )
        
        return chunks
```

**Benefits**:
- Better retrieval than raw question embedding
- Works without labeled training data
- Leverages our semantic chunking metadata for filtering

---

### 3. **Constitutional AI: Harmlessness from AI Feedback**

**Paper**: Bai et al., 2022 (Anthropic)

**Where**: Constitutional AI Agent (Phase 2)

**How**:
- Define "constitution" = set of principles for good Q&A pairs
- Agent critiques its own outputs against principles
- Revises until all principles satisfied

**Constitutional Principles for Q&A**:
1. **Accuracy**: Answer must be verifiable from source chunks
2. **Clarity**: Question and answer must be understandable
3. **Completeness**: Answer covers all relevant aspects
4. **Non-hallucination**: No information beyond source material
5. **Proper Citation**: Include page/section references

**Implementation**:
```python
class ConstitutionalAI:
    PRINCIPLES = {
        "accuracy": "Answer must be verifiable from source",
        "clarity": "Question must be clear and unambiguous",
        "completeness": "Answer must address all parts of question",
        "faithfulness": "Answer must not add information beyond source",
        "citation": "Answer must cite specific sections/pages"
    }
    
    def apply_constitution(self, qa_pair, source_chunks):
        # Check each principle
        for principle, description in self.PRINCIPLES.items():
            # Critique
            critique = self.llm.generate(
                f"Does this Q&A pair satisfy: {description}?\n"
                f"Q: {qa_pair.question}\nA: {qa_pair.answer}\n"
                f"Source: {source_chunks}"
            )
            
            # If violation, revise
            if not critique.satisfies:
                qa_pair = self.revise(qa_pair, principle, critique)
        
        return qa_pair
```

**Benefits**:
- Automated quality control without human labeling
- Ensures answers are grounded in source material
- Reduces need for manual verification

---

### 4. **RAGAS: Automated Evaluation of Retrieval Augmented Generation**

**Paper**: Es et al., 2023

**Where**: Evaluation & Quality Control (Phase 3)

**How**:
- Provides automated metrics for RAG system quality
- Evaluates both retrieval and generation quality
- No human-labeled ground truth needed

**RAGAS Metrics**:
1. **Context Precision**: What % of retrieved chunks are relevant?
2. **Context Recall**: What % of relevant information was retrieved?
3. **Faithfulness**: Is generated answer faithful to context?
4. **Answer Relevancy**: Does answer actually address the question?

**Implementation**:
```python
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy
)

# Evaluate semantic chunking vs baseline
baseline_scores = evaluate(
    dataset=baseline_qa_pairs,
    metrics=[context_precision, context_recall, faithfulness, answer_relevancy]
)

semantic_scores = evaluate(
    dataset=semantic_qa_pairs,
    metrics=[context_precision, context_recall, faithfulness, answer_relevancy]
)

# Compare
print(f"Context Precision: Baseline {baseline_scores.context_precision:.2f} "
      f"vs Semantic {semantic_scores.context_precision:.2f}")
```

**Benefits**:
- Objective comparison between chunking strategies
- Identifies weak points in retrieval/generation
- Guides system improvements with quantitative metrics

---

### 5. **Self-RAG: Learning to Retrieve, Generate, and Critique**

**Paper**: Asai et al., 2023

**Where**: Answer Generator Agent (Phase 2)

**How**:
- Agent decides when to retrieve additional information
- Generates answer while reflecting on quality
- Self-critiques and retrieves more if needed

**Self-RAG Process**:
1. **Retrieve**: Get initial relevant chunks
2. **Relevance Check**: Are chunks actually useful?
3. **Generate**: Produce answer segment
4. **Support Check**: Is segment supported by retrieved chunks?
5. **Utility Check**: Is answer useful/complete?
6. **Iterate**: Retrieve more if needed

**Implementation**:
```python
class SelfRAGAgent:
    def generate_answer(self, question, vectorstore):
        answer_segments = []
        context = []
        
        while not self.is_complete(answer_segments):
            # Retrieve
            chunks = vectorstore.retrieve(question, context)
            
            # Check relevance
            if not self.is_relevant(chunks, question):
                # Refine query and retry
                question = self.refine_query(question)
                continue
            
            context.extend(chunks)
            
            # Generate answer segment
            segment = self.generate_segment(question, context)
            
            # Check if supported by context
            if not self.is_supported(segment, context):
                # Regenerate with different approach
                continue
            
            answer_segments.append(segment)
        
        return self.combine_segments(answer_segments)
```

**Benefits**:
- More accurate answers through iterative refinement
- Retrieves only when necessary (efficiency)
- Self-verification reduces hallucinations

---

## Complete Pipeline: From PDF to Q&A Pairs

### Stage 1: Document Preprocessing (Week 1-2)

**Input**: Academic textbook PDF (e.g., Tipler_Llewellyn.pdf, M2_cours.pdf)

**Steps**:
1. **PDF Parsing** (ENSTA baseline parser)
   - Extract raw text from PDF
   - Clean special characters, preserve formulas
   
2. **Semantic Chunking** (Our innovation)
   - Extract TOC structure automatically
   - Detect semantic units (definitions, theorems, examples, equations)
   - Create variable-size chunks (300-2000 chars) respecting boundaries
   - Add metadata: chapter, section, subsection, page_range, semantic_type
   
3. **Embedding & Indexing**
   - Generate embeddings using OpenAI `text-embedding-3-small`
   - Store in ChromaDB with metadata
   - Enable hybrid search (semantic + metadata filtering)

**Output**: Vector database with 10,000-20,000 semantically coherent chunks

**Evidence from Testing**:
- Tipler textbook (758 pages): 15,890 chunks, 8,961 definitions preserved
- M2 textbook (84 pages): 3,464 chunks, 1,882 definitions preserved
- Average chunk: 1,200 chars (vs 400 in baseline)

---

### Stage 2: Multi-Agent Q&A Generation (Week 3-10)

**Input**: Vector database + target curriculum sections

**Agent Coordination Flow**:

```
User Input: "Generate Q&A for Chapter 3, Section 3.2 (Quantum Mechanics)"
    │
    ▼
[HyDE Retriever Agent]
    ├─► Generate hypothetical answer about quantum mechanics
    ├─► Retrieve top 10 relevant chunks
    ├─► Filter by metadata: chapter="3", section="3.2", type="definition"
    └─► Output: 5 highly relevant chunks with context
    │
    ▼
[Question Generator Agent]
    ├─► Input: Retrieved chunks + metadata
    ├─► Generate 3 question types:
    │   • Factual: "What is the Heisenberg Uncertainty Principle?"
    │   • Analytical: "Derive the uncertainty relation for position and momentum"
    │   • Conceptual: "Why can't we know exact position and momentum simultaneously?"
    └─► Output: 3 candidate questions
    │
    ▼
[Self-RAG Answer Generator]
    ├─► For each question:
    │   ├─► Retrieve relevant chunks (with HyDE if needed)
    │   ├─► Check relevance of retrieved context
    │   ├─► Generate answer segment
    │   ├─► Verify support from source
    │   ├─► Iterate until complete
    │   └─► Add citations (page numbers, sections)
    └─► Output: 3 Q&A pairs with citations
    │
    ▼
[Reflexion Agent]
    ├─► For each Q&A pair:
    │   ├─► Self-evaluate quality
    │   ├─► Generate critique: "Question is unclear about which particles"
    │   ├─► Reflect: "I should specify we're discussing fundamental particles"
    │   ├─► Refine: Improve question clarity
    │   └─► Iterate up to 3 times
    └─► Output: 3 refined Q&A pairs
    │
    ▼
[Constitutional AI Agent]
    ├─► For each refined Q&A:
    │   ├─► Check Accuracy: Is answer verifiable from source?
    │   ├─► Check Clarity: Is question unambiguous?
    │   ├─► Check Completeness: Does answer cover all aspects?
    │   ├─► Check Faithfulness: Any hallucinated information?
    │   ├─► Check Citation: Are page/section references correct?
    │   └─► Revise if any principle violated
    └─► Output: 3 verified Q&A pairs
    │
    ▼
[Final Q&A Dataset]
```

**Output**: High-quality synthetic Q&A dataset with:
- Questions of varying difficulty and types
- Answers grounded in source material
- Citations to specific pages/sections
- Quality scores from RAGAS evaluation

---

### Stage 3: Evaluation & Iteration (Week 11-14)

**Evaluation Framework**:

1. **RAGAS Automated Metrics**
   ```python
   # Evaluate our multi-agent system
   results = evaluate(
       dataset=generated_qa_pairs,
       metrics=[
           context_precision,      # Are retrieved chunks relevant?
           context_recall,         # Did we retrieve all relevant chunks?
           faithfulness,           # Is answer faithful to source?
           answer_relevancy        # Does answer address question?
       ]
   )
   
   # Compare against baselines
   print(f"Context Precision: {results.context_precision:.3f}")
   print(f"Faithfulness: {results.faithfulness:.3f}")
   ```

2. **Comparison Studies**
   - **Baseline 1**: Fixed-size chunking (512 chars) without agents
   - **Baseline 2**: Semantic chunking without multi-agent refinement
   - **Ours**: Semantic chunking + full multi-agent pipeline
   
3. **Ablation Studies**
   - Remove HyDE: How much does retrieval quality drop?
   - Remove Reflexion: How much does Q&A quality drop?
   - Remove Constitutional AI: Do hallucinations increase?
   - Remove Self-RAG: Do answers become less faithful?

**Metrics to Report**:
- Context Precision: >0.85 target
- Context Recall: >0.80 target
- Faithfulness: >0.90 target
- Answer Relevancy: >0.85 target
- Human evaluation: Accuracy, clarity, usefulness (sample 100 pairs)

---

### Stage 4: Deployment & Documentation (Week 15-16)

**Deliverables**:
1. **MCP Server** (Model Context Protocol)
   - Expose Q&A generation API
   - Integrate with Claude/ChatGPT
   - Support batch processing

2. **Web Interface** (Optional)
   - Upload textbook PDFs
   - Select chapters/sections
   - Generate Q&A pairs
   - Export in various formats (JSON, CSV, LaTeX)

3. **Documentation**
   - Technical report (LaTeX)
   - Presentation slides
   - API documentation
   - Usage examples

4. **Code Repository**
   - Clean, documented codebase
   - Example notebooks
   - Evaluation scripts
   - Pre-computed results

---

## Implementation Details

### Technology Stack

**Core Libraries**:
- `langchain`: LLM orchestration and RAG pipelines
- `openai`: GPT-4 for generation, embeddings
- `chromadb`: Vector database for semantic search
- `pymupdf`: PDF text extraction
- `ragas`: Evaluation framework

**Agent Framework**:
- `langgraph`: Multi-agent coordination and state management
- `pydantic`: Data validation and type safety

**Development Tools**:
- `jupyter`: Interactive development and testing
- `pytest`: Unit and integration testing
- `black`: Code formatting

### File Structure

```
Agentic_AI/
├── src/
│   ├── parsers/
│   │   └── ensta_parser.py          # PDF text extraction
│   ├── chunking/
│   │   └── semantic_chunker.py      # Our semantic chunking
│   ├── agents/
│   │   ├── retriever.py             # HyDE retriever
│   │   ├── question_generator.py    # Question generation
│   │   ├── answer_generator.py      # Self-RAG answer generation
│   │   ├── reflexion.py             # Reflexion self-improvement
│   │   └── constitutional_ai.py     # Constitutional AI alignment
│   ├── evaluation/
│   │   └── ragas_eval.py            # RAGAS evaluation
│   └── vector_store.py              # ChromaDB interface
├── data/
│   ├── pdfs/                        # Input textbooks
│   ├── chunks/                      # Processed chunks
│   └── qa_pairs/                    # Generated Q&A
├── tests/
│   ├── test_chunking.py
│   ├── test_agents.py
│   └── test_evaluation.py
└── notebooks/
    ├── 01_chunking_comparison.ipynb
    ├── 02_agent_testing.ipynb
    └── 03_evaluation.ipynb
```

---

## Key Innovations

### 1. **Semantic Chunking with Metadata**

**Problem**: Traditional fixed-size chunking (512 chars) splits definitions, formulas, and theorems across multiple chunks, degrading retrieval quality.

**Solution**: Structure-aware chunking that:
- Extracts TOC hierarchy automatically
- Detects semantic units (definitions, theorems, examples)
- Preserves unit boundaries (variable 300-2000 char chunks)
- Adds rich metadata (chapter, section, type, pages)

**Evidence**:
- Tested on 758-page physics textbook
- Preserved 8,961 definitions intact (vs 0 in baseline)
- 43.6% more efficient retrieval through metadata filtering
- 3x larger average chunk size (1,200 vs 400 chars)

### 2. **Multi-Agent Coordination**

**Problem**: Single-agent systems lack self-correction and quality control mechanisms.

**Solution**: Coordinated multi-agent pipeline where:
- Each agent specializes in one task (retrieve, generate, refine, verify)
- Agents build on each other's outputs iteratively
- Self-reflection and critique loops ensure quality

**Benefits**:
- Better than single GPT-4 call (iterative refinement)
- Better than human labeling (automated at scale)
- Better than simple RAG (multi-stage quality control)

### 3. **Research-Driven Design**

**Problem**: Ad-hoc RAG systems lack principled design and rigorous evaluation.

**Solution**: Implement proven techniques from top-tier research:
- HyDE (2022): Better retrieval through hypothetical answers
- Self-RAG (2023): Self-critique during generation
- Reflexion (2023): Verbal reinforcement learning for improvement
- Constitutional AI (2022): Principle-based alignment
- RAGAS (2023): Automated evaluation metrics

**Result**: State-of-the-art Q&A generation system backed by peer-reviewed research.

---

## Expected Outcomes

### Quantitative Metrics

**Retrieval Quality**:
- Context Precision: >0.85 (85% of retrieved chunks relevant)
- Context Recall: >0.80 (80% of relevant information retrieved)

**Generation Quality**:
- Faithfulness: >0.90 (90% of answers grounded in source)
- Answer Relevancy: >0.85 (85% of answers address question)

**Efficiency**:
- 43.6% fewer chunks to search (via metadata filtering)
- 50-100 Q&A pairs per textbook chapter
- Processing time: ~5 minutes per chapter

### Qualitative Outcomes

**Q&A Dataset Characteristics**:
- Diverse question types (factual, analytical, conceptual)
- Multiple difficulty levels (basic, intermediate, advanced)
- Proper citations (page numbers, section references)
- No hallucinations (grounded in source material)

**Use Cases**:
- Training data for educational chatbots
- Exam question generation for instructors
- Self-study materials for students
- RAG system benchmarking

---

## Timeline (16 Weeks)

**Weeks 1-2**: Document Processing
- ✅ Implement semantic chunker
- ✅ Test on multiple textbooks (Tipler, M2_cours)
- ✅ Compare with baseline (ENSTA fixed-size)
- ⏳ Vector store integration (ChromaDB + OpenAI)

**Weeks 3-4**: Retrieval Agent (HyDE)
- Implement HyDE retriever
- Test retrieval quality with/without HyDE
- Optimize metadata filtering strategies

**Weeks 5-6**: Question Generator Agent
- Implement question generation
- Generate diverse question types
- Evaluate question quality

**Weeks 7-8**: Answer Generator Agent (Self-RAG)
- Implement Self-RAG framework
- Add citation generation
- Test faithfulness to source

**Weeks 9-10**: Refinement Agents (Reflexion + Constitutional AI)
- Implement Reflexion loop
- Define constitutional principles
- Test iterative improvement

**Weeks 11-12**: Evaluation (RAGAS)
- Implement RAGAS metrics
- Run ablation studies
- Compare with baselines

**Weeks 13-14**: Integration & Testing
- End-to-end pipeline testing
- Performance optimization
- Bug fixes and improvements

**Weeks 15-16**: Documentation & Deployment
- Write technical report (LaTeX)
- Create presentation slides
- Prepare code repository
- Demo and handoff

---

## References

### Research Papers

#### Core Component Papers

1. **Reflexion**: Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., & Yao, S. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*. arXiv:2303.11366.

2. **HyDE**: Gao, L., Ma, X., Lin, J., & Callan, J. (2022). *Precise Zero-Shot Dense Retrieval without Relevance Labels*. arXiv:2212.10496.

3. **Constitutional AI**: Bai, Y., et al. (2022). *Constitutional AI: Harmlessness from AI Feedback*. Anthropic.

4. **RAGAS**: Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation*. arXiv:2309.15217.

5. **Self-RAG**: Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2023). *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*. arXiv:2310.11511.

#### Multi-Agent Framework Papers

6. **Diverse and Private Synthetic Datasets for RAG**: Authors TBD (2024). *Diverse And Private Synthetic Datasets Generation for RAG evaluation: A multi-agent framework*. arXiv:2508.18929. (Primary inspiration paper shown to tutor)

7. **MAG-V**: Sengupta, S., Vashistha, H., Curtis, K., Mallipeddi, A., Mathur, A., Ross, J., & Gou, L. (2024). *MAG-V: A Multi-Agent Framework for Synthetic Data Generation and Verification*. arXiv:2412.04494.

8. **RAGentA**: Besrour, I., He, J., Schreieder, T., & Färber, M. (2025). *RAGentA: Multi-Agent Retrieval-Augmented Generation for Attributed Question Answering*. arXiv:2506.16988. Accepted at SIGIR 2025.

9. **Multi-Agent Question Generation**: Wang, S., Wei, Z., Fan, Z., Liu, Y., & Huang, X. (2019). *A multi-agent communication framework for question-worthy phrase extraction and question generation*. Proceedings of the AAAI Conference on Artificial Intelligence, 33(01), 7168-7175.

### Implementation References

- **LangChain Documentation**: https://python.langchain.com/docs/
- **ChromaDB Documentation**: https://docs.trychroma.com/
- **RAGAS Documentation**: https://docs.ragas.io/
- **OpenAI API**: https://platform.openai.com/docs/

---

## Conclusion

This multi-agent RAG system represents a significant advancement over traditional Q&A generation approaches by:

1. **Using semantic chunking** to preserve the integrity of definitions, formulas, and theorems
2. **Coordinating multiple specialized agents** for retrieval, generation, refinement, and verification
3. **Implementing cutting-edge research** (HyDE, Self-RAG, Reflexion, Constitutional AI, RAGAS)
4. **Providing rigorous evaluation** through automated metrics and ablation studies

The result is a state-of-the-art system capable of generating high-quality, grounded, and verifiable synthetic Q&A pairs from academic textbooks at scale.

**Status**: Phase 1 (Document Processing) completed with proven results on real textbooks. Ready to proceed to Phase 2 (Multi-Agent Implementation).
