# RAG Benchmarking System Architecture

## 🎯 System Overview

A multi-agent system for generating high-quality **(Question, Answer, Context)** golden datasets from university textbooks (PDFs). Built with LangGraph orchestration and Model Context Protocol (MCP) for data access.

### Core Purpose
Generate diverse, factually accurate Q&A triplets that:
- Cover the entire curriculum systematically
- Avoid topic redundancy through intelligent memory
- Self-improve through reflexion loops
- Validate against ground truth via citation verification

---

## 🏗️ Architecture Components

```
┌─────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR (LangGraph)                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  State: {covered_topics, current_section, feedback}      │  │
│  │  Memory: Topic Embeddings Vector Store (Diversity Check) │  │
│  └──────────────────────────────────────────────────────────┘  │
└────┬────────────────────┬────────────────────┬──────────────────┘
     │                    │                    │
     │                    │                    │
     ▼                    ▼                    ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  GENERATOR  │    │  ANSWERER   │    │   CRITIC    │
│ (Professor) │    │  (Student)  │    │    (QA)     │
│             │    │             │    │             │
│ Tools:      │    │ Tools:      │    │ Tools:      │
│ - read_     │    │ - vector_   │    │ - verify_   │
│   curriculum│    │   search    │    │   citation  │
│ - read_     │    │ - keyword_  │    │ - ragas_    │
│   section   │    │   search    │    │   metrics   │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │
       │                  │                  │
       └──────────────────┴──────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  MCP TEXTBOOK SERVER  │
              ├───────────────────────┤
              │ • PDF Parser          │
              │ • ChromaDB Vector DB  │
              │ • Metadata Store      │
              │ • Citation Verifier   │
              └───────────────────────┘
                          │
                          ▼
                  [Golden Dataset]
                  questions.jsonl
```

---

## 🔄 The Reflexion Loop (Self-Improvement Mechanism)

### Phase 1: Generation
```
Orchestrator → Generator: "Create a question from Chapter 3.2"
Generator reads section content → Generates Question
```

### Phase 2: Answer Retrieval (RAG Simulation)
```
Orchestrator → Answerer: "Answer this question using retrieval"
Answerer:
  1. vector_search(question) → Top-K chunks
  2. keyword_search(entities) → Exact matches
  3. Synthesizes Answer + Cites Pages
```

### Phase 3: Validation & Feedback
```
Orchestrator → Critic: "Validate this Q&A pair"
Critic performs:
  1. Citation Verification:
     - verify_citation(page_42, "definition of entropy")
     - Detects hallucinations
  
  2. RAGAS Metrics:
     - Faithfulness: 0.92 ✓
     - Answer Relevance: 0.88 ✓
     - Context Precision: 0.65 ⚠️ (LOW)
  
  3. Generates Feedback:
     {
       "verdict": "REJECT",
       "reason": "Context lacks specificity. Question requires thermodynamics equations but retrieved chunks discuss general concepts.",
       "suggestions": {
         "generator": "Rephrase question to target specific theorem on page 43.",
         "answerer": "Use keyword_search for 'Carnot Cycle' formula."
       }
     }
```

### Phase 4: Reflexion (Loop Back)
```
Orchestrator updates state:
  state["feedback"] = critic_feedback
  state["iteration"] = 2

If verdict == "REJECT":
  1. Send feedback to Generator → Regenerate Question (with constraints)
  2. Send feedback to Answerer → Adjust retrieval strategy
  3. Repeat Phase 2 & 3 (Max 3 iterations)

If verdict == "ACCEPT":
  commit_to_dataset(q, a, context, metrics)
  Move to next section
```

### Reflexion Trigger Conditions
| Condition | Action |
|-----------|--------|
| Faithfulness < 0.85 | Answerer must cite different pages |
| Answer Relevance < 0.80 | Generator rephrases question |
| Context Precision < 0.70 | Answerer switches to keyword_search |
| Citation Failed Verification | Answerer retrieves new chunks |
| Max Iterations (3) Reached | Log as "Hard Question" + Skip |

---

## 🧠 Topic Memory (Diversity Enforcement)

### Problem
Without memory, the system generates 20 questions about "Entropy" and ignores "Enthalpy."

### Solution: Semantic Diversity Checking

#### Data Structure
```python
topic_memory = {
    "covered_topics": [
        {"keywords": ["entropy", "disorder", "thermodynamics"], "embedding": [0.12, -0.45, ...]},
        {"keywords": ["Newton's laws", "inertia"], "embedding": [0.67, 0.23, ...]},
        ...
    ],
    "section_coverage": {
        "chapter_1": 5,  # 5 questions generated
        "chapter_2": 3,
        ...
    }
}
```

#### Algorithm
```python
def check_topic_redundancy(new_question_keywords):
    """
    Returns: (is_redundant: bool, similarity_score: float)
    """
    # 1. Embed the new keywords
    new_embedding = embed(new_question_keywords)
    
    # 2. Compare against all covered topics
    for covered in topic_memory["covered_topics"]:
        similarity = cosine_similarity(new_embedding, covered["embedding"])
        
        # Threshold: If > 0.85, topics are too similar
        if similarity > 0.85:
            return (True, similarity)
    
    # 3. If unique, add to memory
    topic_memory["covered_topics"].append({
        "keywords": new_question_keywords,
        "embedding": new_embedding
    })
    return (False, 0.0)
```

#### Integration with Orchestrator
```python
# Before generation
if orchestrator.check_topic_redundancy(["entropy", "heat death"]):
    orchestrator.instruct_generator(
        "Topic already covered. Pick a different concept from Chapter 3."
    )
    return  # Skip generation
```

#### Coverage Balancing
```python
# Ensure each chapter gets equal representation
target_per_chapter = total_questions // num_chapters

if topic_memory["section_coverage"]["chapter_2"] >= target_per_chapter:
    orchestrator.select_next_section(exclude=["chapter_2"])
```

---

## 🛠️ MCP Server Tools (Data Layer)

### For Question Generator (The "Professor")
**Needs full curriculum visibility to design comprehensive questions.**

#### `read_curriculum_structure()`
```json
{
  "chapters": [
    {
      "id": "ch1",
      "title": "Classical Mechanics",
      "sections": [
        {"id": "1.1", "title": "Newton's Laws", "pages": [1, 12]},
        {"id": "1.2", "title": "Energy Conservation", "pages": [13, 24]}
      ]
    }
  ]
}
```

#### `read_section_content(section_id="1.2")`
```json
{
  "section_id": "1.2",
  "title": "Energy Conservation",
  "pages": [13, 24],
  "full_text": "The law of conservation of energy states...",
  "key_definitions": ["Kinetic Energy", "Potential Energy"],
  "equations": ["KE = ½mv²", "PE = mgh"]
}
```

### For Answer Generator (The "Student")
**Simulates RAG limitations—must search, cannot read full chapters.**

#### `vector_search(query, k=5, filters={"chapter": "ch1"})`
```json
{
  "chunks": [
    {
      "text": "Kinetic energy is the energy of motion...",
      "page_number": 14,
      "section_id": "1.2",
      "relevance_score": 0.94
    }
  ]
}
```

#### `keyword_search(query="Euler's Formula")`
```json
{
  "exact_matches": [
    {
      "text": "Euler's formula: e^(iπ) + 1 = 0",
      "page_number": 78,
      "context_window": 200  // chars before/after
    }
  ]
}
```

### For Critic (The "Quality Assurance")

#### `verify_citation(page_number=42, quote_snippet="entropy increases")`
```json
{
  "exists": true,
  "similarity_score": 0.96,
  "actual_text": "The entropy of an isolated system always increases.",
  "verdict": "ACCURATE"
}
```

#### `ragas_metric_calculator(question, answer, context)`
```json
{
  "faithfulness": 0.92,
  "answer_relevance": 0.88,
  "context_precision": 0.75,
  "context_recall": 0.82,
  "overall_score": 0.84
}
```

---

## 📊 Data Flow (End-to-End Example)

### Input
```
PDF: university_physics_textbook.pdf (500 pages)
Target: Generate 100 high-quality Q&A pairs
```

### Step-by-Step Flow

#### 1. Initialization
```python
orchestrator.load_pdf("university_physics_textbook.pdf")
orchestrator.build_vector_store()  # ChromaDB with metadata
orchestrator.initialize_topic_memory()
```

#### 2. Section Selection (Iteration 1)
```python
curriculum = mcp_server.read_curriculum_structure()
orchestrator.select_section("1.1")  # Start with Chapter 1, Section 1
```

#### 3. Question Generation
```python
# Generator reads full section
section_content = mcp_server.read_section_content("1.1")

# Generator creates question
question = generator.create_question(
    content=section_content,
    constraints={"difficulty": "intermediate", "type": "conceptual"}
)
# Output: "Explain how Newton's Third Law applies to rocket propulsion."
```

#### 4. Diversity Check
```python
keywords = ["Newton", "Third Law", "rocket", "propulsion"]
is_redundant = orchestrator.check_topic_redundancy(keywords)

if is_redundant:
    # Reject and regenerate with different angle
    orchestrator.feedback = "Topic too similar to Q#5. Focus on momentum instead."
    return  # Loop back to Step 3
```

#### 5. Answer Generation (RAG Simulation)
```python
# Answerer retrieves context
vector_results = mcp_server.vector_search(question, k=5)
keyword_results = mcp_server.keyword_search("Third Law")

# Synthesize answer
answer, citations = answerer.generate_answer(
    question=question,
    retrieved_chunks=vector_results + keyword_results
)
# Output: "According to Newton's Third Law (page 8), for every action..."
```

#### 6. Validation
```python
# Critic checks citations
citation_valid = mcp_server.verify_citation(
    page_number=8,
    quote_snippet="for every action there is an equal and opposite reaction"
)

# Critic calculates metrics
metrics = mcp_server.ragas_metric_calculator(question, answer, context)

# Critic decides
if metrics["faithfulness"] > 0.85 and citation_valid:
    verdict = "ACCEPT"
else:
    verdict = "REJECT"
    feedback = "Citation on page 8 is accurate, but context lacks specificity..."
```

#### 7A. Accept Path
```python
if verdict == "ACCEPT":
    orchestrator.commit_to_dataset({
        "question": question,
        "answer": answer,
        "context": context,
        "metadata": {
            "section": "1.1",
            "pages": [8, 9],
            "metrics": metrics
        }
    })
    orchestrator.select_next_section()  # Move to Section 1.2
```

#### 7B. Reject Path (Reflexion)
```python
if verdict == "REJECT":
    orchestrator.state["feedback"] = feedback
    orchestrator.state["iteration"] += 1
    
    if orchestrator.state["iteration"] < 3:
        # Loop back to Step 3 (regenerate with feedback)
        generator.regenerate_with_feedback(feedback)
    else:
        # Max iterations reached
        orchestrator.log_hard_question(question, reason=feedback)
        orchestrator.select_next_section()
```

### Output
```jsonl
{"question": "Explain how Newton's Third Law...", "answer": "According to...", "context": "...", "metrics": {...}}
{"question": "What is the relationship between...", "answer": "The relationship...", "context": "...", "metrics": {...}}
...
```

---

## 🎯 Key Design Decisions

### 1. Why MCP for Data Access?
- **Standardization**: Agents use tool-calling (LLM-friendly).
- **Separation of Concerns**: Vector DB changes don't affect agents.
- **Testability**: Mock MCP server for unit tests.

### 2. Why ChromaDB (Not Neo4j)?
- **Simplicity**: Vector + metadata filtering sufficient for MVP.
- **Metadata Indexing**: Store `chapter`, `page`, `section` as filters.
- **Future-Proof**: Can add Neo4j layer later for entity graphs.

### 3. Why Reflexion Loop?
- **Self-Correction**: LLMs make mistakes; iteration improves quality.
- **Data Efficiency**: Better than discarding bad Q&A pairs.
- **Mimics Human Process**: Professors revise questions after seeing student confusion.

### 4. Why Topic Memory?
- **Diversity**: Prevents clustering on "obvious" topics.
- **Coverage**: Ensures all chapters represented equally.
- **Efficiency**: No need to generate 1000 questions to get 100 unique ones.

---

## 🔧 Critical Architectural Improvements (Must-Have)

### 1. Academic Content Validator (Beyond RAGAS)

#### Problem
RAGAS metrics (Faithfulness, Context Precision) validate **retrieval quality** but NOT **academic correctness**.

**Example Failure Case**:
```json
{
  "question": "Explain the second law of thermodynamics",
  "answer": "Entropy always increases in isolated systems",
  "ragas_scores": {
    "faithfulness": 0.95,  // Answer matches retrieved context ✓
    "context_precision": 0.88  // Context is relevant ✓
  }
}
```

**Problem**: Answer is **INCOMPLETE** (missing reversible process exception, statistical interpretation, and heat flow direction). RAGAS can't detect this because it only validates answer-context alignment, not domain completeness.

#### Solution: Multi-Layer Validation

```python
class AcademicValidator:
    """
    Validates academic rigor beyond retrieval quality.
    
    Validation Layers:
    1. Completeness: Are key concepts covered?
    2. Precision: Is terminology accurate?
    3. Cognitive Level: Does it match question depth?
    4. Citation Density: Are claims properly supported?
    """
    
    def validate(self, qa_pair, section_content, metadata):
        """
        Returns:
            {
                "completeness_score": 0.0-1.0,
                "precision_score": 0.0-1.0,
                "cognitive_alignment": "match|mismatch",
                "citation_coverage": 0.0-1.0,
                "overall_academic_score": 0.0-1.0,
                "feedback": "Detailed explanation..."
            }
        """
        results = {
            "completeness": self._check_completeness(qa_pair, section_content),
            "precision": self._check_terminology_accuracy(qa_pair),
            "cognitive_level": self._check_cognitive_alignment(qa_pair),
            "citation_density": self._check_citation_coverage(qa_pair)
        }
        
        # Compute overall score
        weights = {"completeness": 0.4, "precision": 0.3, "cognitive_level": 0.2, "citation_density": 0.1}
        overall = sum(results[k]["score"] * weights[k] for k in weights)
        
        return {
            **results,
            "overall_academic_score": overall,
            "passes_threshold": overall >= 0.75,
            "feedback": self._generate_feedback(results)
        }
    
    def _check_completeness(self, qa_pair, section_content):
        """
        Ensure answer covers ALL key concepts from the domain.
        
        Algorithm:
        1. Extract key concepts from the section (definitions, theorems, laws)
        2. Identify which concepts are relevant to the question
        3. Check if the answer mentions all relevant concepts
        4. Penalize missing critical information
        """
        # Extract domain concepts (using NER + domain-specific patterns)
        key_concepts = self._extract_domain_concepts(section_content)
        
        # Determine which concepts are relevant to this question
        relevant_concepts = self._filter_relevant_concepts(qa_pair['question'], key_concepts)
        
        # Check coverage in the answer
        covered_concepts = [
            concept for concept in relevant_concepts
            if self._concept_mentioned_in_answer(concept, qa_pair['answer'])
        ]
        
        coverage_ratio = len(covered_concepts) / len(relevant_concepts) if relevant_concepts else 1.0
        
        return {
            "score": coverage_ratio,
            "total_concepts": len(relevant_concepts),
            "covered_concepts": len(covered_concepts),
            "missing_concepts": [c for c in relevant_concepts if c not in covered_concepts],
            "feedback": f"Answer covers {len(covered_concepts)}/{len(relevant_concepts)} key concepts"
        }
    
    def _extract_domain_concepts(self, text):
        """
        Extract academic concepts using:
        - Pattern matching: "X is defined as", "Theorem:", "Law of X"
        - Named Entity Recognition (scientific terms)
        - Keyword extraction (TF-IDF for domain terms)
        """
        concepts = []
        
        # Pattern 1: Definitions
        definition_pattern = r'([A-Z][a-z\w\s]+?)\s+(?:is defined as|means|refers to)'
        concepts.extend(re.findall(definition_pattern, text))
        
        # Pattern 2: Theorems/Laws/Principles
        theorem_pattern = r'((?:Theorem|Law|Principle|Rule)\s+of\s+[A-Z][a-z\w\s]+)'
        concepts.extend(re.findall(theorem_pattern, text))
        
        # Pattern 3: Bold terms (approximated by ALL CAPS or repeated terms)
        # In real implementation, parse PDF formatting
        
        return list(set(concepts))  # Deduplicate
    
    def _check_terminology_accuracy(self, qa_pair):
        """
        Validate that technical terms are used correctly.
        
        Checks:
        - Domain-specific terminology database
        - Common misconceptions (e.g., "weight" vs "mass")
        - Mathematical notation correctness
        """
        # Load domain terminology database
        correct_terms = self._load_terminology_database(qa_pair['metadata']['domain'])
        
        # Extract terms from answer
        answer_terms = self._extract_technical_terms(qa_pair['answer'])
        
        # Check for misuse
        errors = []
        for term in answer_terms:
            if term in self.COMMON_MISCONCEPTIONS:
                # Check if used in wrong context
                if self._is_misconception(term, qa_pair['answer'], qa_pair['question']):
                    errors.append(f"Potential misuse of '{term}'")
        
        accuracy = 1.0 - (len(errors) / max(len(answer_terms), 1))
        
        return {
            "score": accuracy,
            "errors": errors,
            "feedback": "; ".join(errors) if errors else "Terminology usage is accurate"
        }
    
    def _check_cognitive_alignment(self, qa_pair):
        """
        Verify question and answer are at the same cognitive level.
        
        Uses Bloom's Taxonomy:
        - Remember: Define, List, Recall
        - Understand: Explain, Describe, Summarize
        - Apply: Calculate, Demonstrate, Use
        - Analyze: Compare, Contrast, Examine
        - Evaluate: Justify, Critique, Assess
        - Create: Design, Develop, Formulate
        """
        question_level = self._classify_bloom_level(qa_pair['question'])
        answer_level = self._classify_bloom_level(qa_pair['answer'])
        
        # Answer should be at or above question level
        level_order = ["remember", "understand", "apply", "analyze", "evaluate", "create"]
        q_idx = level_order.index(question_level)
        a_idx = level_order.index(answer_level)
        
        alignment = "match" if a_idx >= q_idx else "mismatch"
        
        return {
            "score": 1.0 if alignment == "match" else 0.5,
            "question_level": question_level,
            "answer_level": answer_level,
            "alignment": alignment,
            "feedback": f"Question requires '{question_level}' but answer demonstrates '{answer_level}'"
        }
    
    def _classify_bloom_level(self, text):
        """Classify text by Bloom's taxonomy using verb analysis"""
        text_lower = text.lower()
        
        bloom_verbs = {
            "remember": ["define", "list", "recall", "name", "identify", "what is"],
            "understand": ["explain", "describe", "summarize", "interpret", "how does"],
            "apply": ["calculate", "demonstrate", "use", "apply", "solve"],
            "analyze": ["compare", "contrast", "examine", "distinguish", "why"],
            "evaluate": ["justify", "critique", "assess", "judge", "evaluate"],
            "create": ["design", "develop", "formulate", "construct", "derive"]
        }
        
        for level, verbs in bloom_verbs.items():
            if any(verb in text_lower for verb in verbs):
                return level
        
        return "understand"  # Default
    
    def _check_citation_coverage(self, qa_pair):
        """
        Ensure claims are properly cited.
        
        Heuristic: Ratio of factual claims to citations
        """
        # Count factual claims (sentences with technical terms or numbers)
        claims = self._extract_factual_claims(qa_pair['answer'])
        
        # Count citations
        citations = qa_pair.get('citations', [])
        
        # Ideal ratio: 1 citation per 2-3 claims
        expected_citations = len(claims) / 2.5
        citation_ratio = len(citations) / expected_citations if expected_citations > 0 else 1.0
        
        score = min(citation_ratio, 1.0)  # Cap at 1.0
        
        return {
            "score": score,
            "claim_count": len(claims),
            "citation_count": len(citations),
            "feedback": f"Answer has {len(citations)} citations for {len(claims)} factual claims"
        }
```

#### Integration with Critic Agent

```python
class CriticAgent:
    def __init__(self):
        self.ragas_evaluator = RAGASEvaluator()
        self.academic_validator = AcademicValidator()  # NEW
    
    async def validate(self, qa_pair, section_content):
        # Layer 1: RAGAS (retrieval quality)
        ragas_scores = await self.ragas_evaluator.evaluate(qa_pair)
        
        # Layer 2: Academic validation (content quality)
        academic_scores = self.academic_validator.validate(qa_pair, section_content, qa_pair['metadata'])
        
        # Combined decision
        passes_ragas = all(ragas_scores[m] >= threshold for m, threshold in RAGAS_THRESHOLDS.items())
        passes_academic = academic_scores['passes_threshold']
        
        verdict = "ACCEPT" if (passes_ragas and passes_academic) else "REJECT"
        
        feedback = {
            "ragas": ragas_scores,
            "academic": academic_scores,
            "verdict": verdict,
            "suggestions": self._generate_improvement_suggestions(ragas_scores, academic_scores)
        }
        
        return feedback
```

#### Configuration Updates

Add to `config.py`:
```python
class EvaluationConfig(BaseModel):
    # RAGAS thresholds
    min_faithfulness: float = Field(default=0.85)
    min_answer_relevance: float = Field(default=0.80)
    min_context_precision: float = Field(default=0.70)
    
    # Academic validation thresholds (NEW)
    min_completeness: float = Field(default=0.75)
    min_terminology_accuracy: float = Field(default=0.90)
    min_citation_coverage: float = Field(default=0.60)
    require_cognitive_alignment: bool = Field(default=True)
```

---

### 2. Multi-Dimensional Diversity Manager

#### Problem
Current design uses **only cosine similarity on keywords** to check diversity. This is insufficient because:

**Example**:
```
Question 1: "What is Newton's Second Law?"  (type: definition)
Question 2: "Calculate the force on a 5kg object accelerating at 2m/s²"  (type: calculation)

Cosine similarity: 0.75 (seems different enough)
BUT: Both questions test the SAME concept (F=ma), just different cognitive levels!
```

A proper golden dataset needs diversity across **multiple dimensions**.

#### Solution: Multi-Dimensional Diversity Scoring

```python
class MultiDimensionalDiversityManager:
    """
    Ensures diversity across 4 dimensions:
    1. Content (topic similarity)
    2. Question Type (factoid, reasoning, comparison, etc.)
    3. Difficulty (beginner, intermediate, expert)
    4. Cognitive Level (Bloom's taxonomy)
    """
    
    DIMENSION_WEIGHTS = {
        "content": 0.35,      # What topic?
        "type": 0.30,         # What kind of question?
        "difficulty": 0.20,   # How hard?
        "cognitive": 0.15     # What cognitive skill?
    }
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.history = []  # Store all generated questions
        
        # Track distribution for balancing
        self.type_distribution = defaultdict(int)
        self.difficulty_distribution = defaultdict(int)
        self.cognitive_distribution = defaultdict(int)
    
    def check_diversity(self, new_question_data, threshold=0.85):
        """
        Returns:
            (is_diverse: bool, diversity_score: float, dimension_scores: dict)
        """
        if not self.history:
            return True, 1.0, {}
        
        # Compute score for each dimension
        scores = {
            "content": self._content_diversity(new_question_data),
            "type": self._type_diversity(new_question_data),
            "difficulty": self._difficulty_diversity(new_question_data),
            "cognitive": self._cognitive_diversity(new_question_data)
        }
        
        # Weighted combination
        overall_diversity = sum(
            scores[dim] * self.DIMENSION_WEIGHTS[dim]
            for dim in self.DIMENSION_WEIGHTS
        )
        
        is_diverse = overall_diversity >= threshold
        
        return is_diverse, overall_diversity, scores
    
    def _content_diversity(self, new_q):
        """
        Semantic similarity to existing questions.
        
        Returns: 1.0 (very diverse) to 0.0 (duplicate)
        """
        new_embedding = self.embedding_model.embed(new_q['question'])
        
        max_similarity = 0.0
        for historical_q in self.history:
            similarity = cosine_similarity(new_embedding, historical_q['embedding'])
            max_similarity = max(max_similarity, similarity)
        
        # Invert: high similarity = low diversity
        diversity = 1.0 - max_similarity
        return diversity
    
    def _type_diversity(self, new_q):
        """
        Penalize overrepresented question types.
        
        Question Types:
        - factoid: "What is X?"
        - definition: "Define X"
        - process: "How does X work?"
        - comparison: "Compare X and Y"
        - application: "When to use X?"
        - reasoning: "Why does X happen?"
        - calculation: "Calculate X"
        """
        new_type = new_q['type']
        
        # Count how many questions of this type we already have
        type_count = self.type_distribution[new_type]
        total_questions = len(self.history)
        
        if total_questions == 0:
            return 1.0
        
        # Current proportion of this type
        current_proportion = type_count / total_questions
        
        # Target: Even distribution (1/7 for 7 types)
        num_types = 7
        target_proportion = 1.0 / num_types
        
        # Diversity = 1.0 when at target, decreases as we exceed target
        if current_proportion <= target_proportion:
            return 1.0  # Encourage this type
        else:
            # Penalize proportionally to how much we exceed target
            excess = current_proportion - target_proportion
            penalty = excess / target_proportion
            return max(0.0, 1.0 - penalty)
    
    def _difficulty_diversity(self, new_q):
        """
        Balance beginner/intermediate/expert questions.
        
        Target distribution:
        - Beginner: 30%
        - Intermediate: 50%
        - Expert: 20%
        """
        new_difficulty = new_q['difficulty']
        
        target_distribution = {
            "beginner": 0.30,
            "intermediate": 0.50,
            "expert": 0.20
        }
        
        difficulty_count = self.difficulty_distribution[new_difficulty]
        total = len(self.history)
        
        if total == 0:
            return 1.0
        
        current_proportion = difficulty_count / total
        target_proportion = target_distribution[new_difficulty]
        
        if current_proportion <= target_proportion:
            return 1.0
        else:
            excess = current_proportion - target_proportion
            penalty = excess / target_proportion
            return max(0.0, 1.0 - penalty)
    
    def _cognitive_diversity(self, new_q):
        """
        Balance Bloom's taxonomy levels.
        
        Target distribution (for academic rigor):
        - Remember: 15%
        - Understand: 30%
        - Apply: 25%
        - Analyze: 20%
        - Evaluate: 7%
        - Create: 3%
        """
        new_cognitive = new_q['cognitive_level']
        
        target_distribution = {
            "remember": 0.15,
            "understand": 0.30,
            "apply": 0.25,
            "analyze": 0.20,
            "evaluate": 0.07,
            "create": 0.03
        }
        
        cognitive_count = self.cognitive_distribution[new_cognitive]
        total = len(self.history)
        
        if total == 0:
            return 1.0
        
        current_proportion = cognitive_count / total
        target_proportion = target_distribution.get(new_cognitive, 0.1)
        
        if current_proportion <= target_proportion:
            return 1.0
        else:
            excess = current_proportion - target_proportion
            penalty = excess / target_proportion
            return max(0.0, 1.0 - penalty)
    
    def add_to_history(self, question_data):
        """Store question and update distribution trackers"""
        self.history.append(question_data)
        self.type_distribution[question_data['type']] += 1
        self.difficulty_distribution[question_data['difficulty']] += 1
        self.cognitive_distribution[question_data['cognitive_level']] += 1
    
    def get_statistics(self):
        """Return current distribution statistics"""
        total = len(self.history)
        return {
            "total_questions": total,
            "type_distribution": {k: v/total for k, v in self.type_distribution.items()},
            "difficulty_distribution": {k: v/total for k, v in self.difficulty_distribution.items()},
            "cognitive_distribution": {k: v/total for k, v in self.cognitive_distribution.items()}
        }
```

#### Question Type Classification

```python
class QuestionTypeClassifier:
    """Classify questions into types using LLM or rule-based approach"""
    
    QUESTION_TYPES = {
        "factoid": {
            "description": "Simple fact recall",
            "patterns": [r"^What is", r"^Who", r"^When", r"^Where"],
            "examples": ["What is entropy?", "Who discovered relativity?"]
        },
        "definition": {
            "description": "Request for formal definition",
            "patterns": [r"^Define", r"definition of"],
            "examples": ["Define momentum.", "What is the definition of energy?"]
        },
        "process": {
            "description": "Explain how something works",
            "patterns": [r"^How does", r"^How to", r"process of"],
            "examples": ["How does photosynthesis work?"]
        },
        "comparison": {
            "description": "Compare two or more concepts",
            "patterns": [r"compare", r"difference between", r"contrast"],
            "examples": ["Compare speed and velocity.", "What's the difference between mass and weight?"]
        },
        "application": {
            "description": "When/where to use a concept",
            "patterns": [r"when to use", r"application of", r"example of"],
            "examples": ["When would you use Newton's third law?"]
        },
        "reasoning": {
            "description": "Explain why something happens",
            "patterns": [r"^Why", r"reason for", r"cause of"],
            "examples": ["Why does ice float?", "What causes rainbows?"]
        },
        "calculation": {
            "description": "Mathematical computation",
            "patterns": [r"^Calculate", r"^Compute", r"^Find the value"],
            "examples": ["Calculate the force on a 5kg object."]
        }
    }
    
    def classify(self, question_text):
        """Classify question into one of the defined types"""
        question_lower = question_text.lower()
        
        # Rule-based classification
        for qtype, info in self.QUESTION_TYPES.items():
            for pattern in info["patterns"]:
                if re.search(pattern, question_lower, re.IGNORECASE):
                    return qtype
        
        # Fallback: Use LLM
        return self._llm_classify(question_text)
    
    def _llm_classify(self, question_text):
        """Use LLM to classify when rules fail"""
        prompt = f"""
        Classify this question into ONE of these types:
        - factoid: Simple fact recall
        - definition: Request for definition
        - process: How something works
        - comparison: Compare concepts
        - application: When to use
        - reasoning: Why something happens
        - calculation: Mathematical problem
        
        Question: {question_text}
        
        Type:"""
        
        # Call LLM (simplified)
        response = self.llm.complete(prompt)
        return response.strip().lower()
```

#### Integration with Orchestrator

```python
class Orchestrator:
    def __init__(self, config):
        self.diversity_manager = MultiDimensionalDiversityManager(embedding_model)
        self.type_classifier = QuestionTypeClassifier()
    
    async def generate_question(self, section):
        # Generate candidate question
        question_text = await self.generator.generate(section)
        
        # Classify and enrich
        question_data = {
            "question": question_text,
            "type": self.type_classifier.classify(question_text),
            "difficulty": self._estimate_difficulty(question_text),
            "cognitive_level": self._classify_cognitive_level(question_text),
            "embedding": self.embedding_model.embed(question_text)
        }
        
        # Check diversity
        is_diverse, score, dimension_scores = self.diversity_manager.check_diversity(question_data)
        
        if not is_diverse:
            # Provide feedback to generator
            feedback = self._generate_diversity_feedback(dimension_scores)
            # Regenerate with constraints
            return await self.generate_question_with_constraints(section, feedback)
        
        # Accept and store
        self.diversity_manager.add_to_history(question_data)
        return question_data
```

---

### 3. Semantic Chunking Strategy

#### Problem
**Fixed character-count chunking** (500 chars) breaks content at arbitrary points:

```
Chunk 1: "...Newton's Second Law states that F = ma, where F is force, m is ma"
Chunk 2: "ss, and a is acceleration. This fundamental principle..."
```

**Issues**:
- Theorems split across chunks → incomplete retrieval
- Equations separated from explanations → confusion
- Context lost at boundaries → lower precision

#### Solution: Semantic Chunking with Structure Awareness

```python
class SemanticChunker:
    """
    Chunk based on academic content structure, not character count.
    
    Parsing Strategy:
    1. Identify structural elements: paragraphs, theorems, equations, examples
    2. Keep atomic units together (never split theorem mid-statement)
    3. Add contextual headers to each chunk
    4. Track element types in metadata
    """
    
    ELEMENT_PATTERNS = {
        "theorem": re.compile(r'(Theorem \d+\.?\d*:?.+?)(?=\n\n|\nTheorem|\nProof|$)', re.DOTALL),
        "definition": re.compile(r'(Definition \d+\.?\d*:?.+?)(?=\n\n|Definition|$)', re.DOTALL),
        "example": re.compile(r'(Example \d+\.?\d*:?.+?)(?=\n\n|Example|$)', re.DOTALL),
        "equation": re.compile(r'([A-Za-z]\s*=\s*.+?)(?=\n|$)'),
        "proof": re.compile(r'(Proof:?.+?)(?=\n\n|Theorem|Example|$)', re.DOTALL)
    }
    
    def chunk(self, section_text, metadata, max_chunk_size=800):
        """
        Create semantically coherent chunks.
        
        Args:
            section_text: Raw text from section
            metadata: {chapter, section, pages}
            max_chunk_size: Target maximum (not strict)
        
        Returns:
            List of chunks with enhanced metadata
        """
        # Step 1: Parse structural elements
        elements = self._parse_structure(section_text)
        
        # Step 2: Group into chunks
        chunks = []
        current_chunk = []
        current_length = 0
        
        for element in elements:
            element_length = len(element['text'])
            
            # Rule 1: Theorems are ATOMIC (never split)
            if element['type'] in ['theorem', 'definition', 'proof']:
                # Flush current chunk
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, metadata))
                    current_chunk = []
                    current_length = 0
                
                # Add theorem as standalone chunk
                chunks.append(self._create_chunk([element], metadata))
                continue
            
            # Rule 2: Equations stay with surrounding context
            if element['type'] == 'equation':
                # Always include with previous element (explanation before equation)
                if current_chunk:
                    current_chunk.append(element)
                    current_length += element_length
                else:
                    # Orphaned equation, make standalone
                    chunks.append(self._create_chunk([element], metadata))
                continue
            
            # Rule 3: Regular paragraphs—chunk by size
            if current_length + element_length > max_chunk_size:
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, metadata))
                current_chunk = [element]
                current_length = element_length
            else:
                current_chunk.append(element)
                current_length += element_length
        
        # Flush remaining
        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, metadata))
        
        return chunks
    
    def _parse_structure(self, text):
        """
        Parse text into structural elements.
        
        Returns:
            [
                {"type": "paragraph", "text": "...", "order": 0},
                {"type": "theorem", "text": "Theorem 1: ...", "order": 1},
                {"type": "equation", "text": "F = ma", "order": 2},
                ...
            ]
        """
        elements = []
        remaining_text = text
        order = 0
        
        # Extract special elements first
        for element_type, pattern in self.ELEMENT_PATTERNS.items():
            for match in pattern.finditer(text):
                elements.append({
                    "type": element_type,
                    "text": match.group(1).strip(),
                    "start": match.start(),
                    "end": match.end(),
                    "order": order
                })
                order += 1
        
        # Sort by position in text
        elements.sort(key=lambda e: e['start'])
        
        # Fill gaps with regular paragraphs
        final_elements = []
        last_end = 0
        
        for elem in elements:
            # Text between last element and this one
            gap_text = text[last_end:elem['start']].strip()
            if gap_text:
                # Split into paragraphs
                paragraphs = gap_text.split('\n\n')
                for para in paragraphs:
                    if para.strip():
                        final_elements.append({
                            "type": "paragraph",
                            "text": para.strip(),
                            "order": len(final_elements)
                        })
            
            # Add the special element
            final_elements.append(elem)
            last_end = elem['end']
        
        # Remaining text after last element
        gap_text = text[last_end:].strip()
        if gap_text:
            paragraphs = gap_text.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    final_elements.append({
                        "type": "paragraph",
                        "text": para.strip(),
                        "order": len(final_elements)
                    })
        
        return final_elements
    
    def _create_chunk(self, elements, metadata):
        """
        Create a chunk with contextual header.
        
        Format:
        [Chapter X, Section Y.Z: Title]
        
        <content>
        """
        # Build contextual header
        header = f"[Chapter {metadata['chapter']}, Section {metadata['section_id']}"
        if 'section_title' in metadata:
            header += f": {metadata['section_title']}"
        header += "]\n\n"
        
        # Combine element texts
        content = "\n\n".join(e['text'] for e in elements)
        
        # Enhanced metadata
        chunk_metadata = {
            **metadata,
            "element_types": [e['type'] for e in elements],
            "has_theorem": any(e['type'] == 'theorem' for e in elements),
            "has_definition": any(e['type'] == 'definition' for e in elements),
            "has_equation": any(e['type'] == 'equation' for e in elements),
            "has_example": any(e['type'] == 'example' for e in elements),
            "num_elements": len(elements)
        }
        
        return {
            "text": header + content,
            "metadata": chunk_metadata
        }
```

#### Why This Matters: Concrete Example

**Before (Fixed Chunking)**:
```
Chunk 1: "...pressure is P = F/A, where P is pressure, F is the perpendicular for"
Chunk 2: "ce, and A is the area. Applications include hydraulic systems..."
```
- Equation split mid-definition
- Student retrieves Chunk 1 → incomplete information
- Answer quality: 0.65

**After (Semantic Chunking)**:
```
Chunk 1: "[Chapter 2, Section 2.3: Pressure]

Pressure is defined as the force per unit area. Mathematically:

P = F/A

where:
- P is pressure (Pa)
- F is the perpendicular force (N)
- A is the area (m²)

Applications include hydraulic systems, atmospheric pressure, and fluid dynamics."
```
- Complete concept with equation + explanation
- Context header helps with relevance
- Answer quality: 0.92 ✅

---

### 4. LangGraph Checkpointing & State Persistence

#### Problem
**Current Risk**: If generation crashes after 80 questions (e.g., rate limit, API timeout, power outage), you lose **all progress**.

#### Solution: Persistent State with SQLite Checkpointing

```python
from langgraph.checkpoint import SqliteSaver
from langgraph.graph import StateGraph, END
import sqlite3

class Orchestrator:
    """Orchestrator with automatic state persistence"""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize checkpoint database
        self.checkpointer = SqliteSaver.from_conn_string(
            "checkpoints.db"
        )
        
        # Build LangGraph state machine
        self.graph = self._build_graph()
        
        # Compile with checkpointing
        self.app = self.graph.compile(
            checkpointer=self.checkpointer,
            interrupt_before=["critic"],  # Optional: pause before critic for manual review
        )
    
    def _build_graph(self):
        """Build the agent workflow graph"""
        workflow = StateGraph(OrchestratorState)
        
        # Add nodes (agents)
        workflow.add_node("select_section", self.select_section)
        workflow.add_node("generate_question", self.generate_question)
        workflow.add_node("check_diversity", self.check_diversity)
        workflow.add_node("generate_answer", self.generate_answer)
        workflow.add_node("critic", self.critic_validate)
        workflow.add_node("save_result", self.save_to_dataset)
        
        # Define edges (workflow)
        workflow.set_entry_point("select_section")
        
        workflow.add_edge("select_section", "generate_question")
        workflow.add_edge("generate_question", "check_diversity")
        
        # Conditional: diversity check
        workflow.add_conditional_edges(
            "check_diversity",
            self._diversity_check_router,
            {
                "pass": "generate_answer",
                "fail": "generate_question"  # Loop back
            }
        )
        
        workflow.add_edge("generate_answer", "critic")
        
        # Conditional: critic verdict
        workflow.add_conditional_edges(
            "critic",
            self._critic_verdict_router,
            {
                "accept": "save_result",
                "reject": "generate_question"  # Reflexion loop
            }
        )
        
        workflow.add_edge("save_result", END)
        
        return workflow
    
    async def run(self, config_params):
        """
        Run the generation pipeline with automatic checkpointing.
        
        Features:
        - Auto-saves state after each step
        - Can resume from any point
        - Thread-based isolation (multiple runs don't conflict)
        """
        # Create unique thread ID
        thread_id = f"generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Check for existing checkpoint
        existing_state = self.checkpointer.get({"configurable": {"thread_id": thread_id}})
        
        if existing_state:
            print(f"📦 Resuming from checkpoint:")
            print(f"   - Questions generated: {len(existing_state['dataset'])}")
            print(f"   - Current section: {existing_state['current_section']}")
            print(f"   - Iteration: {existing_state['iteration']}")
            
            initial_state = existing_state
        else:
            print("🆕 Starting fresh generation")
            initial_state = self._create_initial_state(config_params)
        
        # Run the workflow
        final_state = None
        async for event in self.app.astream(
            initial_state,
            config={"configurable": {"thread_id": thread_id}}
        ):
            # State is automatically checkpointed after each step
            step_name = list(event.keys())[0]
            print(f"✓ Completed: {step_name}")
            final_state = event[step_name]
        
        return final_state
    
    def resume_from_checkpoint(self, thread_id):
        """Manually resume a specific checkpoint"""
        state = self.checkpointer.get({"configurable": {"thread_id": thread_id}})
        
        if not state:
            raise ValueError(f"No checkpoint found for thread {thread_id}")
        
        print(f"Resuming thread {thread_id}...")
        return self.app.astream(
            state,
            config={"configurable": {"thread_id": thread_id}}
        )
    
    def list_checkpoints(self):
        """List all available checkpoints"""
        # Query checkpoint database
        conn = sqlite3.connect("checkpoints.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT thread_id, checkpoint_id, created_at FROM checkpoints")
        checkpoints = cursor.fetchall()
        
        conn.close()
        
        return [
            {
                "thread_id": row[0],
                "checkpoint_id": row[1],
                "created_at": row[2]
            }
            for row in checkpoints
        ]


class OrchestratorState(TypedDict):
    """State object for LangGraph"""
    # Generation progress
    dataset: List[Dict]  # Generated Q&A pairs
    current_section: str
    iteration: int
    total_questions: int
    
    # Current work
    question_candidate: Optional[str]
    answer_candidate: Optional[str]
    context_retrieved: Optional[List[Dict]]
    
    # Feedback loops
    feedback: Optional[Dict]
    reflexion_count: int
    
    # Memory
    topic_memory: List[Dict]
    covered_sections: List[str]
```

#### Usage Example

```python
# Run generation
orchestrator = Orchestrator(config)

try:
    final_state = await orchestrator.run({
        "target_questions": 100,
        "chapters": ["ch1", "ch2"]
    })
except Exception as e:
    print(f"❌ Error: {e}")
    print("Don't worry—progress is saved!")

# Later: Resume
checkpoints = orchestrator.list_checkpoints()
print("Available checkpoints:")
for cp in checkpoints:
    print(f"  - {cp['thread_id']} at {cp['created_at']}")

# Resume specific run
await orchestrator.resume_from_checkpoint("generation_20251129_143022")
```

#### Checkpoint Inspection Tool

```python
def inspect_checkpoint(thread_id):
    """View checkpoint details without resuming"""
    checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
    state = checkpointer.get({"configurable": {"thread_id": thread_id}})
    
    print(f"Checkpoint: {thread_id}")
    print(f"  Progress: {len(state['dataset'])}/{state['total_questions']} questions")
    print(f"  Current section: {state['current_section']}")
    print(f"  Reflexion loops: {state['reflexion_count']}")
    print(f"  Covered sections: {len(state['covered_sections'])}")
    
    # Show last 3 questions
    print("\n  Last 3 questions:")
    for qa in state['dataset'][-3:]:
        print(f"    Q: {qa['question'][:60]}...")
```

---

---

## 🔧 Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Orchestration** | LangGraph | State management, agent coordination |
| **Data Layer** | Model Context Protocol | Tool interface for agents |
| **Vector DB** | ChromaDB | Semantic search with metadata |
| **LLM** | OpenAI GPT-4 / Claude | Question/Answer generation |
| **Evaluation** | RAGAS | Faithfulness, relevance metrics |
| **PDF Processing** | PyMuPDF | Extract text with page numbers |
| **Embeddings** | OpenAI text-embedding-3-small | Vector representations |

---

---

## 🚀 Should-Have Improvements (High-Impact)

### 5. Hard Negative Generation

#### Motivation
Real RAG benchmarks need **hard negatives** (plausible but incorrect answers) to test retrieval precision. Current design only generates positive Q&A pairs.

**Example**:
```json
{
  "question": "What is the Pauli Exclusion Principle?",
  "correct_answer": "No two fermions can occupy the same quantum state simultaneously.",
  "hard_negatives": [
    "No two bosons can occupy the same energy level.",  // Entity swap
    "All fermions must occupy different energy levels.", // Semantic drift
    "Fermions can share quantum states if they have opposite spin." // Relation inversion
  ]
}
```

**Use Case**: Test if retrieval system can distinguish between similar but incorrect information.

#### Implementation

```python
class HardNegativeGenerator:
    """
    Generate plausible but incorrect answers using three strategies:
    1. Entity Swapping: Replace key entities with similar ones
    2. Relation Inversion: Flip causal/logical relationships
    3. Adjacent Topic Confusion: Use facts from related sections
    """
    
    def generate(self, qa_pair, section_content, adjacent_sections):
        """
        Returns:
            {
                "hard_negatives": [str, str, str],
                "strategies_used": [str, str, str],
                "plausibility_scores": [float, float, float]
            }
        """
        negatives = []
        
        # Strategy 1: Entity Swapping
        entity_negative = self._entity_swap(qa_pair['answer'], section_content)
        negatives.append(entity_negative)
        
        # Strategy 2: Relation Inversion
        relation_negative = self._invert_relation(qa_pair['answer'])
        negatives.append(relation_negative)
        
        # Strategy 3: Adjacent Topic Confusion
        adjacent_negative = self._adjacent_confusion(qa_pair, adjacent_sections)
        negatives.append(adjacent_negative)
        
        return {
            "hard_negatives": negatives,
            "strategies_used": ["entity_swap", "relation_inversion", "adjacent_confusion"],
            "plausibility_scores": [self._score_plausibility(neg) for neg in negatives]
        }
    
    def _entity_swap(self, answer, section_content):
        """
        Replace key entities with similar but incorrect ones.
        
        Example:
        "Electrons are negatively charged" → "Protons are negatively charged"
        """
        # Extract entities from answer
        entities = self._extract_entities(answer)
        
        # Find similar entities from the same section
        similar_entities = self._find_similar_entities(entities, section_content)
        
        # Replace one entity
        if entities and similar_entities:
            original_entity = random.choice(entities)
            replacement = random.choice(similar_entities[original_entity])
            
            hard_negative = answer.replace(original_entity, replacement)
            return hard_negative
        
        return answer  # Fallback
    
    def _invert_relation(self, answer):
        """
        Flip causal or logical relationships.
        
        Example:
        "A causes B" → "B causes A"
        "If A, then B" → "If B, then A"
        """
        # Pattern 1: Causal relationships
        causal_pattern = r'(.+?)\s+causes?\s+(.+)'
        match = re.search(causal_pattern, answer, re.IGNORECASE)
        if match:
            cause, effect = match.groups()
            inverted = f"{effect.strip()} causes {cause.strip()}"
            return answer.replace(match.group(0), inverted)
        
        # Pattern 2: Conditional statements
        conditional_pattern = r'If\s+(.+?),\s+then\s+(.+)'
        match = re.search(conditional_pattern, answer, re.IGNORECASE)
        if match:
            condition, consequence = match.groups()
            inverted = f"If {consequence.strip()}, then {condition.strip()}"
            return answer.replace(match.group(0), inverted)
        
        # Pattern 3: "A increases B" → "A decreases B"
        for word, opposite in [("increases", "decreases"), ("higher", "lower"), ("above", "below")]:
            if word in answer.lower():
                return answer.replace(word, opposite)
        
        return answer  # Fallback: no inversion possible
    
    def _adjacent_confusion(self, qa_pair, adjacent_sections):
        """
        Use facts from nearby sections (same chapter but different topic).
        
        Example:
        Question: "What is mitosis?"
        Correct: "Cell division resulting in two identical daughter cells"
        Hard Negative: "Cell division resulting in four haploid cells" (actually meiosis)
        """
        # Find a related but different topic
        for adj_section in adjacent_sections:
            # Check if section is thematically close but not identical
            similarity = self._compute_topic_similarity(
                qa_pair['question'],
                adj_section['content']
            )
            
            if 0.4 < similarity < 0.8:  # Sweet spot: related but different
                # Extract a fact from adjacent section
                fact = self._extract_relevant_fact(adj_section['content'], qa_pair['question'])
                
                if fact:
                    return fact
        
        return qa_pair['answer']  # Fallback
    
    def _score_plausibility(self, hard_negative):
        """
        Score how plausible the hard negative is (1.0 = very believable).
        
        Uses LLM to judge: "Could a student reasonably believe this?"
        """
        prompt = f"""
        Rate the plausibility of this statement on a scale from 0.0 to 1.0:
        - 1.0 = Sounds correct, even experts might be fooled
        - 0.5 = Questionable, but not obviously wrong
        - 0.0 = Clearly nonsensical
        
        Statement: "{hard_negative}"
        
        Plausibility score (just the number):
        """
        
        response = self.llm.complete(prompt)
        try:
            return float(response.strip())
        except ValueError:
            return 0.5  # Default
```

#### Integration

```python
class AnswerGenerator:
    def __init__(self):
        self.hard_negative_generator = HardNegativeGenerator()
    
    async def generate(self, question, section_content, adjacent_sections):
        # Generate correct answer
        correct_answer = await self._generate_correct_answer(question, section_content)
        
        # Generate hard negatives
        hard_negatives_data = self.hard_negative_generator.generate(
            {"question": question, "answer": correct_answer},
            section_content,
            adjacent_sections
        )
        
        return {
            "question": question,
            "correct_answer": correct_answer,
            "hard_negatives": hard_negatives_data["hard_negatives"],
            "negative_strategies": hard_negatives_data["strategies_used"],
            "plausibility_scores": hard_negatives_data["plausibility_scores"]
        }
```

#### Updated Golden Dataset Schema

```json
{
  "question": "What is the Pauli Exclusion Principle?",
  "correct_answer": "No two fermions can occupy the same quantum state simultaneously.",
  "hard_negatives": [
    {
      "text": "No two bosons can occupy the same energy level.",
      "strategy": "entity_swap",
      "plausibility": 0.85
    },
    {
      "text": "All fermions must occupy different energy levels.",
      "strategy": "relation_inversion",
      "plausibility": 0.75
    },
    {
      "text": "Fermions can share quantum states if they have opposite spin.",
      "strategy": "adjacent_confusion",
      "plausibility": 0.90
    }
  ],
  "context": "...",
  "citations": ["page 142, section 5.3"]
}
```

---

### 6. Query Expansion with HyDE

#### Motivation
Simple vector search fails when question phrasing differs from textbook wording.

**Example Problem**:
```
Question: "How do plants convert light into energy?"
Textbook uses: "photosynthesis", "chloroplast", "ATP synthesis"
Student query uses: "convert light", "energy production"
→ Poor retrieval due to vocabulary mismatch
```

#### Solution: Hypothetical Document Embeddings (HyDE) + Synonym Expansion

```python
class QueryExpander:
    """
    Expand queries using two techniques:
    1. HyDE: Generate hypothetical answer, embed it, search with that
    2. Synonym Expansion: Add domain-specific synonyms
    """
    
    def __init__(self, embedding_model, llm):
        self.embedding_model = embedding_model
        self.llm = llm
        
        # Domain-specific synonym database
        self.synonyms = {
            "energy": ["ATP", "power", "work", "calories"],
            "cell division": ["mitosis", "meiosis", "cytokinesis"],
            "speed": ["velocity", "rate", "tempo"],
            # ...load from config or pre-built dictionary
        }
    
    async def expand_query(self, question, domain="general"):
        """
        Returns:
            {
                "original_query": str,
                "hypothetical_document": str,
                "synonyms": List[str],
                "expanded_queries": List[str]
            }
        """
        # Technique 1: HyDE
        hypothetical_doc = await self._generate_hypothetical_document(question)
        
        # Technique 2: Synonym expansion
        synonyms = self._expand_with_synonyms(question, domain)
        
        # Combine into multiple query variants
        expanded_queries = [
            question,  # Original
            hypothetical_doc,  # HyDE
            *synonyms  # Synonym variants
        ]
        
        return {
            "original_query": question,
            "hypothetical_document": hypothetical_doc,
            "synonyms": synonyms,
            "expanded_queries": expanded_queries
        }
    
    async def _generate_hypothetical_document(self, question):
        """
        Generate what the answer *might* look like.
        
        HyDE Insight: Even if the generated answer is wrong, its embedding
        will be closer to the correct passage than the question embedding.
        """
        prompt = f"""
        You are an expert. Write a detailed answer to this question.
        Don't worry if you don't know the exact answer—write what a typical
        textbook answer would look like.
        
        Question: {question}
        
        Hypothetical Answer:
        """
        
        response = await self.llm.acomplete(prompt)
        return response.strip()
    
    def _expand_with_synonyms(self, question, domain):
        """
        Replace key terms with domain synonyms.
        
        Example:
        "How do cells produce energy?" →
        [
          "How do cells produce ATP?",
          "How do cells generate power?",
          "What is cellular energy production?"
        ]
        """
        expanded = []
        question_lower = question.lower()
        
        # Find matching terms
        for term, syns in self.synonyms.items():
            if term in question_lower:
                # Create variant for each synonym
                for syn in syns:
                    variant = question.replace(term, syn)
                    expanded.append(variant)
        
        return expanded[:3]  # Limit to top 3


class MCPTextbookServer:
    """Updated MCP server with query expansion"""
    
    def __init__(self):
        self.query_expander = QueryExpander(embedding_model, llm)
    
    async def vector_search(self, query, top_k=5):
        """Enhanced vector search with query expansion"""
        
        # Expand query
        expansion = await self.query_expander.expand_query(query)
        
        # Search with all variants
        all_results = []
        for variant in expansion["expanded_queries"]:
            results = await self.vector_store.search(variant, top_k=top_k)
            all_results.extend(results)
        
        # Deduplicate and rerank
        unique_results = self._deduplicate_results(all_results)
        reranked = self._rerank_by_relevance(query, unique_results)
        
        return reranked[:top_k]
```

#### Configuration

```python
class GenerationConfig(BaseModel):
    # Query expansion settings
    enable_hyde: bool = Field(default=True)
    enable_synonym_expansion: bool = Field(default=True)
    max_query_variants: int = Field(default=5)
    
    # Synonym database path
    synonym_database: str = Field(default="data/synonyms.json")
```

---

### 7. Question Template System

#### Motivation
Ensure consistency and coverage of different question formats.

#### Implementation

```python
class QuestionTemplateSystem:
    """
    Pre-defined templates for different question types.
    
    Benefits:
    - Consistency in phrasing
    - Ensures diverse cognitive levels
    - Easier to generate specific types
    """
    
    TEMPLATES = {
        "definition": [
            "What is {concept}?",
            "Define {concept} in the context of {domain}.",
            "Explain the meaning of {concept}."
        ],
        "comparison": [
            "What is the difference between {concept_a} and {concept_b}?",
            "Compare and contrast {concept_a} and {concept_b}.",
            "How does {concept_a} differ from {concept_b}?"
        ],
        "application": [
            "When would you use {concept}?",
            "Provide an example of {concept} in practice.",
            "How is {concept} applied in {domain}?"
        ],
        "reasoning": [
            "Why does {phenomenon} occur?",
            "Explain the cause of {phenomenon}.",
            "What is the reason for {phenomenon}?"
        ],
        "process": [
            "How does {process} work?",
            "Describe the steps involved in {process}.",
            "Explain the mechanism of {process}."
        ],
        "calculation": [
            "Calculate {variable} given {conditions}.",
            "Find the value of {variable} when {conditions}.",
            "Solve for {variable}: {equation}."
        ],
        "analysis": [
            "What are the implications of {concept}?",
            "Analyze the relationship between {concept_a} and {concept_b}.",
            "Evaluate the impact of {concept} on {domain}."
        ]
    }
    
    def generate_from_template(self, question_type, **kwargs):
        """
        Fill a template with extracted concepts.
        
        Example:
            generate_from_template(
                "comparison",
                concept_a="mitosis",
                concept_b="meiosis"
            )
            → "What is the difference between mitosis and meiosis?"
        """
        templates = self.TEMPLATES.get(question_type, [])
        template = random.choice(templates)
        
        try:
            question = template.format(**kwargs)
            return question
        except KeyError as e:
            raise ValueError(f"Missing required parameter: {e}")
```

---

## 🌟 Nice-to-Have Improvements (Future Enhancements)

### 8. Lightweight Concept Graph

#### Motivation
Track relationships between concepts without full Neo4j overhead.

#### Implementation

```python
import networkx as nx

class ConceptGraph:
    """
    Lightweight knowledge graph using NetworkX.
    
    Tracks:
    - Prerequisite relationships ("understand X before Y")
    - Part-of relationships ("cells are part of tissues")
    - Comparison relationships ("X vs Y")
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
    
    def add_concept(self, concept, metadata=None):
        """Add concept node"""
        self.graph.add_node(concept, **(metadata or {}))
    
    def add_relationship(self, concept_a, concept_b, rel_type):
        """Add edge between concepts"""
        self.graph.add_edge(concept_a, concept_b, type=rel_type)
    
    def get_prerequisites(self, concept):
        """Find concepts that should be understood first"""
        predecessors = list(self.graph.predecessors(concept))
        return [p for p in predecessors if self.graph[p][concept]['type'] == 'prerequisite']
    
    def suggest_comparison_questions(self):
        """Find concept pairs that should be compared"""
        comparison_edges = [
            (u, v) for u, v, data in self.graph.edges(data=True)
            if data.get('type') == 'compare'
        ]
        return comparison_edges
```

---

### 9. Multi-Turn Question Sequences

#### Motivation
Test deeper understanding through question chains.

#### Example

```json
{
  "sequence_id": "seq_001",
  "questions": [
    {
      "order": 1,
      "question": "What is Newton's Second Law?",
      "type": "recall",
      "answer": "F = ma"
    },
    {
      "order": 2,
      "question": "Given a 5kg object accelerating at 2m/s², calculate the force.",
      "type": "application",
      "depends_on": [1],
      "answer": "10 N"
    },
    {
      "order": 3,
      "question": "If the force doubles but mass stays constant, what happens to acceleration?",
      "type": "analysis",
      "depends_on": [1, 2],
      "answer": "Acceleration doubles"
    }
  ]
}
```

---

### 10. Cross-Encoder Reranking

#### Motivation
Improve retrieval precision by reranking vector search results.

#### Implementation

```python
from sentence_transformers import CrossEncoder

class RetrievalReranker:
    """
    Rerank vector search results using cross-encoder.
    
    Difference from bi-encoder (used in ChromaDB):
    - Bi-encoder: Encode query and doc separately, compare embeddings
    - Cross-encoder: Encode query+doc together, output relevance score
    
    Cross-encoder is more accurate but slower (can't pre-compute embeddings).
    """
    
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query, candidates, top_k=5):
        """
        Rerank candidates by relevance to query.
        
        Args:
            query: Question text
            candidates: List of retrieved documents from vector search
            top_k: How many to return
        
        Returns:
            Reranked list of top_k most relevant documents
        """
        # Create query-document pairs
        pairs = [[query, doc['text']] for doc in candidates]
        
        # Score all pairs
        scores = self.model.predict(pairs)
        
        # Sort by score
        scored_candidates = [
            {**doc, "cross_encoder_score": score}
            for doc, score in zip(candidates, scores)
        ]
        scored_candidates.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
        
        return scored_candidates[:top_k]


# Integration
class MCPTextbookServer:
    def __init__(self):
        self.reranker = RetrievalReranker()
    
    async def vector_search(self, query, top_k=5):
        # Step 1: Vector search (retrieve 20 candidates)
        candidates = await self.vector_store.search(query, top_k=20)
        
        # Step 2: Rerank with cross-encoder
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)
        
        return reranked
```

---

## 📈 Quality Metrics

### Per Q&A Pair
- **Faithfulness**: Answer grounded in retrieved context
- **Answer Relevance**: Answer directly addresses question
- **Context Precision**: Retrieved chunks are relevant
- **Citation Accuracy**: Page references are correct

### Dataset-Level
- **Topic Diversity**: Cosine similarity < 0.85 between topics
- **Chapter Coverage**: ±20% variance across chapters
- **Acceptance Rate**: % of Q&A pairs passing first validation
- **Average Reflexion Loops**: Measure of question difficulty

---

## 🚀 Extensibility

### Phase 2 Enhancements
1. **Add Neo4j Knowledge Graph**
   - Extract entities (concepts, formulas)
   - Enable graph-based question templates
   - Example: "Compare X and Y" requires graph traversal

2. **Multi-Modal Questions**
   - Extract diagrams from PDF
   - Generate questions referencing figures
   - Answer must describe visual elements

3. **Difficulty Calibration**
   - Track student performance on generated questions
   - Adjust generator prompts based on failure patterns

4. **Collaborative Filtering**
   - Multiple critics vote on quality
   - Ensemble scoring for robustness

---

## 🎓 Usage Scenarios

### Scenario 1: Full Textbook Coverage
```bash
python main.py --pdf textbook.pdf --target 200 --strategy balanced
# Generates 200 Q&A pairs, evenly distributed across chapters
```

### Scenario 2: Focus on Hard Chapters
```bash
python main.py --pdf textbook.pdf --chapters 5,6,7 --difficulty hard
# Generates challenging questions from specific chapters
```

### Scenario 3: Topic-Specific Dataset
```bash
python main.py --pdf textbook.pdf --keywords "thermodynamics,entropy" --target 50
# Generates 50 questions about specific topics
```

---

## 📚 References

1. **Reflexion**: [Self-Reflective Agents Paper](https://arxiv.org/abs/2303.11366)
2. **RAGAS**: [RAG Assessment Framework](https://github.com/explodinggradients/ragas)
3. **MCP**: [Model Context Protocol](https://modelcontextprotocol.io/)
4. **LangGraph**: [LangChain Graph Framework](https://github.com/langchain-ai/langgraph)

---

## 🏁 Next Steps

1. ✅ Review architecture
2. ⏳ Implement MCP server
3. ⏳ Build agent modules
4. ⏳ Integrate LangGraph orchestrator
5. ⏳ Test with sample textbook
6. ⏳ Evaluate golden dataset quality
