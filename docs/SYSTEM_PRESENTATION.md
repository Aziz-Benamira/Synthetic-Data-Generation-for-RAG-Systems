# 🎯 Synthetic QA Dataset Generation System - Implementation Overview

**Date:** January 15, 2026  
**System:** Multi-Agent RAG Dataset Generation with Constitutional AI Critic

---

## 📋 System Architecture

```
PDF → Semantic Chunker → Question Generator → Answer Generator → Critic Agent
                                ↑                                      ↓
                                └──────── Feedback Loop ───────────────┘
                                         (Orchestrator)
```

---

## 1️⃣ Semantic Chunker

**Purpose:** Parse PDF textbooks into semantically coherent chunks preserving academic structure.

**Implementation:** `src/chunking/semantic_chunker.py`

**Key Features:**
- TOC-aware splitting (respects chapters/sections)
- Preserves definitions, theorems, equations intact
- Rich metadata (chapter, section, page range, semantic type)

**Core Function:**
```python
class SemanticChunk:
    content: str                    # Actual text
    chunk_id: str                   # e.g., "ch1.s2.c3"
    chapter_title: str              # Parent chapter
    section_title: str              # Parent section
    page_range: Tuple[int, int]     # [start, end]
    semantic_type: str              # "definition", "theorem", "example", "text"
```

**Algorithm:**
1. Extract PDF TOC hierarchy
2. Split by sections (natural boundaries)
3. Detect semantic units (definitions, theorems)
4. Add hierarchical metadata
5. Fallback: RecursiveCharacterTextSplitter for large sections

**Statistics:**
- Input: M2_cours.pdf (110 pages, probability theory)
- Output: 110 semantic chunks
- Average chunk: ~500-800 tokens

---

## 2️⃣ Question Generator Agent

**Purpose:** Generate diverse, pedagogically sound questions from chunks.

**Implementation:** `src/agents/question_generator.py`

**LLM Used:** **Mistral 7B Instruct** (local via Ollama)
- Model: `mistral:latest` (~4.5GB VRAM)
- Temperature: 0.7 (creative but controlled)

**Question Types:**
- Conceptual (definitions, understanding)
- Procedural (how-to, methods)
- Factual (specific facts, numbers)

**Difficulty Levels:**
- Easy (30%), Medium (50%), Hard (20%)

**Core Prompt:**
```python
f"""Tu es un expert pédagogique. Génère {num_questions} questions de type {question_type} 
basées UNIQUEMENT sur ce chunk:

{chunk.content}

Contraintes:
- Questions claires et non ambiguës
- Réponses doivent être trouvables dans le chunk
- Variété: éviter répétitions
- Difficulté: {difficulty}

Format JSON:
{{
  "questions": [
    {{"question": "...", "type": "{question_type}", "difficulty": "{difficulty}"}}
  ]
}}"""
```

**Output Format:**
```python
@dataclass
class CandidateQuestion:
    question: str
    question_type: QuestionType      # CONCEPTUAL, PROCEDURAL, FACTUAL
    difficulty: DifficultyLevel      # EASY, MEDIUM, HARD
    source_chunk_id: str
```

**Key Function:**
```python
def generate_from_chunk(self, chunk: SemanticChunk, num_questions: int = 2):
    """Generate questions from a single chunk"""
    # 1. Build prompt with chunk context
    # 2. Call Mistral via Ollama
    # 3. Parse JSON response
    # 4. Return CandidateQuestion objects
```

---

## 3️⃣ Answer Generator Agent

**Purpose:** Generate grounded answers anchored to chunk content.

**Implementation:** `src/agents/answer_generator.py`

**LLM Used:** **Mistral 7B Instruct** (same model, different role)
- Temperature: 0.7
- Focus: Factual accuracy, chunk grounding

**Core Prompt:**
```python
f"""Tu es un assistant pédagogique. Réponds à cette question en te basant STRICTEMENT 
sur le contenu fourni:

QUESTION: {question.question}

CONTENU DE RÉFÉRENCE:
{chunk.content}

RÈGLES CRITIQUES:
✓ Utilise UNIQUEMENT les informations du contenu
✓ Cite des extraits si possible
✓ Si info manquante: indique-le clairement
✗ JAMAIS d'informations externes
✗ JAMAIS d'inventions ou hallucinations

Réponds de manière claire et complète."""
```

**Output:**
```python
@dataclass
class QAPair:
    question: str
    answer: str
    source_file: str
    chunk_id: str
    question_type: str
    difficulty: str
    supporting_quotes: List[str]     # Extracted from chunk
```

**Key Function:**
```python
def generate_answer(self, question: CandidateQuestion, chunk: SemanticChunk) -> str:
    """Generate answer grounded in chunk"""
    # 1. Build prompt with question + chunk
    # 2. Call Mistral
    # 3. Extract supporting quotes
    # 4. Return answer string
```

---

## 4️⃣ Critic Agent (Constitutional AI)

**Purpose:** Evaluate QA pair quality with 5 criteria + hard rules.

**Implementation:** `src/agents/critic_agent.py` (merged with Seif's improvements)

**LLM Used:** **Llama 3 8B** (more strict than Mistral)
- Model: `llama3:8b` (~4.7GB VRAM)
- Temperature: 0.3 (consistent evaluation)
- **Different model** from generator (adversarial setup)

### **Evaluation Criteria (5):**

1. **Anchoring** (0-1): Answer derivable from chunk?
2. **Local Answerability** (0-1): Question answerable from chunk alone?
3. **Factual Accuracy** (0-1): No hallucinations or errors?
4. **Completeness** (0-1): Answer addresses question fully?
5. **Clarity** (0-1): Question and answer clear/unambiguous?

**Pass Threshold:** Score ≥ 0.85 per criterion

**Decision:**
- **Strict Mode:** ALL 5 criteria must pass
- **Lenient Mode:** ≥4/5 must pass

### **Adversarial Prompt (Seif's Innovation):**

```python
f"""Tu es un DÉTECTEUR DE DÉFAUTS, pas un validateur.

ASSUME QUE CHAQUE QA PAIR A DES PROBLÈMES (ils existent!).

TON PROCESSUS OBLIGATOIRE:
1. LIS la question et la réponse
2. CHERCHE activement des problèmes (ne sois pas indulgent)
3. LISTE tous les défauts trouvés
4. SEULEMENT APRÈS, score les critères

QUESTION: {qa.question}
RÉPONSE: {qa.answer}
CHUNK SOURCE: {chunk.content}

ÉVALUE selon 5 CRITÈRES:
- anchoring: Réponse dérivable du chunk? (cite preuves)
- local_answerability: Question répond depuis chunk seul?
- factual_accuracy: Aucune erreur factuelle?
- completeness: Réponse complète?
- clarity: Question/réponse claires?

Format JSON:
{{
  "criteria": {{
    "anchoring": {{"score": 0.0-1.0, "result": "pass/fail", "explanation": "..."}},
    ...
  }},
  "rejection_reasons": ["raison1", "raison2"]
}}"""
```

### **Hard Rules (6 Deterministic Checks):**

**Tuned by Seif, refined during integration:**

```python
def _apply_hard_rules(qa_pair, chunk, criteria_evaluations):
    """Override LLM scores with deterministic rules"""
    
    # RULE 1: Hallucinated numbers
    answer_numbers = extract_numbers(answer)
    chunk_numbers = extract_numbers(chunk.content)
    if numbers_not_in_chunk(answer_numbers, chunk_numbers, tolerance=0.02):
        criteria_evaluations["anchoring"].score = 0.0  # AUTO-REJECT
    
    # RULE 2: Why/How without causality
    if is_why_how_question(question):
        if not has_causal_markers(chunk.content) and not has_causal_markers(answer):
            criteria_evaluations["local_answerability"].score = 0.2
    
    # RULE 3: Short answer for complex question
    if len(question.split()) > 15 and len(answer) < 40:
        criteria_evaluations["completeness"].score = 0.4
    
    # RULE 4: Answer repeats question (>80% word overlap)
    if word_overlap(question, answer) > 0.80 and len(answer.split()) < 20:
        criteria_evaluations["completeness"].score = 0.3
    
    # RULE 5: Oral/informal language ("truc", "machin", "c'est quoi")
    if contains_oral_markers(question):
        criteria_evaluations["clarity"].score = 0.2
    
    # RULE 6: Vague pronouns ("il", "elle", "this") without referent
    if starts_with_vague_pronoun(answer):
        criteria_evaluations["clarity"].score -= 0.15
```

**Output:**
```python
@dataclass
class CriticEvaluation:
    decision: FinalDecision                # PASS or REJECT
    overall_score: float                   # Average of 5 criteria
    criteria_evaluations: Dict             # Detailed per-criterion
    passed_criteria: List[str]             # Which passed
    failed_criteria: List[str]             # Which failed
    rejection_reasons: List[str]           # Human-readable
```

---

## 5️⃣ Orchestrator (Feedback Loop)

**Purpose:** Coordinate multi-agent workflow with retry mechanism.

**Implementation:** `src/orchestrator/pipeline.py`

### **Pipeline Flow:**

```
1. Parse PDF → Chunks (SemanticChunker)
2. For each chunk:
   a. Generate questions (QuestionGenerator)
   b. Generate answers (AnswerGenerator)
   c. Evaluate QA pairs (CriticAgent)
   d. IF REJECTED:
      - Format detailed feedback
      - Send to QuestionGenerator
      - Regenerate question
      - Regenerate answer
      - Re-evaluate (max 2 retries)
   e. IF PASSED or max retries: Save to dataset
3. Export final dataset (HuggingFace format)
```

### **Feedback Loop Code:**

```python
def _evaluate_with_retries(self, qa_pairs, chunk) -> List[Tuple[QAPair, CriticEvaluation]]:
    """Evaluate with retry loop (AGENTIC WORKFLOW)"""
    passed = []
    
    for qa in qa_pairs:
        current_qa = qa
        
        for attempt in range(self.config.max_retries + 1):
            # Evaluate with Critic
            evaluation = self.critic.evaluate(current_qa, chunk)
            
            if evaluation.decision == FinalDecision.PASS:
                passed.append((current_qa, evaluation))
                if attempt > 0:
                    self.stats.passed_after_retry += 1
                break
            
            elif attempt < self.config.max_retries:
                # FORMAT FEEDBACK
                feedback = self.critic.format_feedback_for_retry(evaluation)
                
                # REGENERATE QUESTION with feedback
                new_question = self.question_generator.regenerate_with_feedback(
                    chunk=chunk,
                    previous_question=current_qa.question,
                    critic_feedback=feedback
                )
                
                # REGENERATE ANSWER with feedback
                new_answer = self.answer_generator.regenerate_with_feedback(
                    question=new_question,
                    chunk=chunk,
                    previous_answer=current_qa.answer,
                    critic_feedback=feedback
                )
                
                # Create new QAPair for retry
                current_qa = QAPair.from_question_and_answer(new_question, new_answer)
            
            else:
                # Max retries exceeded - REJECT definitively
                self.stats.rejected_qa_pairs += 1
    
    return passed
```

### **Feedback Format:**

```python
def format_feedback_for_retry(self, evaluation: CriticEvaluation) -> str:
    """Convert evaluation to actionable feedback"""
    feedback_parts = []
    
    for criterion, crit_eval in evaluation.criteria_evaluations.items():
        if crit_eval.result == CriterionResult.FAIL:
            feedback_parts.append(
                f"- {criterion.upper()}: {crit_eval.explanation}"
            )
    
    return "\n".join([
        " PROBLÈMES DÉTECTÉS:",
        *feedback_parts,
        "\n AMÉLIORE la question/réponse pour corriger ces défauts."
    ])
```

---

## 6️⃣ Configuration & Models

### **Ollama Setup (Local LLMs):**

```python
OLLAMA_MODELS = {
    "generator": "mistral:latest",    # 4.5GB - Creative generation
    "critic": "llama3:8b"             # 4.7GB - Strict evaluation
}

# Total VRAM: ~9GB (fits on RTX 5060 8GB with offloading)
```

**Why Different Models?**
- **Mistral:** Better at creative text generation
- **Llama 3:** More analytical, stricter evaluation
- **Adversarial setup:** Generator and critic have different "personalities"

### **Pipeline Configuration:**

```python
config = PipelineConfig(
    pdf_path="data/pdfs/M2_cours.pdf",
    max_chunks=5,                      # Limit for testing
    questions_per_chunk=2,             # 2 questions × 5 chunks = 10 QAs
    
    generator_model="mistral:latest",
    critic_model="llama3:8b",
    temperature=0.7,                   # Generator temp
    
    max_retries=2,                     # Retry loop limit
    language="fr",                     # French
    
    save_checkpoints=True,             # Resume capability
    checkpoint_frequency=10            # Every 10 chunks
)
```

---

##  Evaluation Metrics

### **Quality Metrics:**

| Metric | Formula | Target |
|--------|---------|--------|
| **Pass Rate** | Passed / Total QAs | ~50-70% |
| **Rejection Rate** | Rejected / Total QAs | ~30-50% |
| **Retry Rate** | Total Retries / Total QAs | ~0.5-1.5 |
| **Score Mean** | Avg(overall_score) | ~0.85-0.95 |
| **Score StdDev** | StdDev(overall_score) | ~0.10-0.20 |

### **Per-Criterion Metrics:**

```python
criterion_averages = {
    "anchoring": 0.92,
    "local_answerability": 0.89,
    "factual_accuracy": 0.95,
    "completeness": 0.88,
    "clarity": 0.91
}
```

### **Current Results (M2_cours.pdf test):**

**Before Seif's Improvements:**
- Rejection rate: ~0-10%
- Retries: 0
- Score variance: 0.027 (too uniform)

**After Seif's Improvements:**
- Rejection rate: 0% (but 1 retry triggered on borderline case)
- Retries: 1
- Score variance: 0.039 (+42% improvement in discrimination)
- Unit test on bad data: **70% rejection rate** ✅

---

## 🔧 Key Implementation Details

### **1. Question Regeneration with Feedback:**

```python
def regenerate_with_feedback(self, chunk, previous_question, critic_feedback):
    """Regenerate question incorporating critic feedback"""
    prompt = f"""La question précédente a été rejetée:
    
QUESTION REJETÉE: {previous_question}

FEEDBACK DU CRITIC:
{critic_feedback}

CHUNK SOURCE:
{chunk.content}

Génère une NOUVELLE question qui corrige ces problèmes."""
    
    response = self._call_llm(prompt)
    return self._parse_response(response, chunk)[0]
```

### **2. Hard Rule Helpers:**

```python
def extract_numbers(text: str) -> set:
    """Extract all numbers from text"""
    pattern = r'\b\d+(?:[.,]\d+)?\b'
    numbers = re.findall(pattern, text)
    return set(n.replace(',', '.') for n in numbers)

def has_causal_markers(text: str) -> bool:
    """Check for causal/explanatory markers"""
    causal_markers = [
        'car', 'parce que', 'puisque', 'donc', 'ainsi',
        'par conséquent', 'entraîne', 'provoque', 'cause',
        'en raison de', 'grâce à', 'dû à', 'résulte de'
    ]
    return any(marker in text.lower() for marker in causal_markers)

def is_why_how_question(question: str) -> bool:
    """Check if question requires deep explanation"""
    q_lower = question.lower().strip()
    return (q_lower.startswith('pourquoi') or 
            q_lower.startswith('comment se fait-il') or
            q_lower.startswith('expliquez'))
```

### **3. Dataset Export (HuggingFace Format):**

```python
{
  "metadata": {
    "source_file": "M2_cours.pdf",
    "generation_date": "2026-01-15",
    "total_entries": 9,
    "config": {...}
  },
  "data": [
    {
      "question": "Quelle est la définition d'une tribu?",
      "answer": "Une tribu sur Ω est une famille...",
      "source_file": "M2_cours.pdf",
      "chunk_id": "1.1.c1",
      "page_range": [7, 11],
      "chapter": "Généralités",
      "section": "Tribu",
      "question_type": "conceptual",
      "difficulty": "medium",
      "critic_score": 1.0,
      "criterion_scores": {...}
    }
  ]
}
```

---

##  Current Limitations & Solutions

### **Problem 1: Low Rejection Rate on High-Quality PDFs**

**Issue:** Math textbooks have perfect academic content → 0% rejection

**Why it happens:**
- Textbook definitions are clear, unambiguous
- No informal language
- Numbers are properly contextualized
- Hard rules don't trigger on good content

**Solution:**
✅ **This is actually GOOD** - we don't want to reject high-quality QAs
- The key metric is **score variance** (+42%) and **retry rate** (1 retry vs 0)
- Hard rules are safety nets for production edge cases
- Unit tests confirm 70% rejection on deliberately bad data



### **Problem 5: Feedback Loop Not Always Improving**

**Issue:** Sometimes regenerated QA fails again with same issue

**Why:** LLM doesn't fully understand feedback or chunk lacks info

**Solution (planned):**
1. More explicit feedback formatting
2. Add examples in feedback ("Instead of X, say Y")
3. Limit retries to avoid infinite loops (currently max 2)
4. Track rejection reasons to identify patterns

### **Problem 6: Inconsistent LLM Outputs**

**Issue:** Same prompt can produce different quality responses

**Mitigation:**
- Temperature 0.7 (not too random, not too deterministic)
- JSON output format enforcement
- Retry parsing on malformed responses
- Fallback evaluations when parsing fails

---

## 🎯 Next Steps

### **Short-term (This Week):**
1. ✅ Merge Seif's hard rules (DONE)
2. ✅ Tune thresholds for production (DONE)
3. ⏳ Create harder test cases to validate rules
4. ⏳ Run full pipeline on complete PDF (110 chunks)
5. ⏳ Analyze rejection patterns

### **Medium-term (Next 2 Weeks):**
1. Integrate 3 new validators:
   - `answer_quality_scorer.py` (hallucination detection)
   - `chain_of_thought_validator.py` (reasoning validation)
   - `active_learning_ui.py` (Gradio human review)
2. Fine-tune based on production data
3. Add diversity metrics (question type distribution)
4. Implement curriculum learning (easy → hard)

### **Long-term (Future):**
1. Multi-modal support (images, equations)
2. Cross-chunk questions (requires context aggregation)
3. Difficulty calibration (user testing)
4. Active learning loop (human feedback integration)

---

## 📚 Repository Structure

```
Agentic_AI/
├── data/
│   └── pdfs/
│       └── M2_cours.pdf                    # Test PDF
├── src/
│   ├── chunking/
│   │   └── semantic_chunker.py             # Chunk parser
│   ├── agents/
│   │   ├── question_generator.py           # Q generation
│   │   ├── answer_generator.py             # A generation
│   │   ├── critic_agent.py                 # Evaluation (merged)
│   │   ├── answer_quality_scorer.py        # (Not yet integrated)
│   │   └── chain_of_thought_validator.py   # (Not yet integrated)
│   ├── orchestrator/
│   │   └── pipeline.py                     # Main coordinator
│   └── utils/
│       ├── ollama_client.py                # Ollama setup
│       └── active_learning_ui.py           # (Not yet integrated)
├── demo_comparison.json                    # Test results
├── RAPPORT_FINAL_CRITIC.md                 # Original report
├── SEIF_CHANGES_ANALYSIS.md                # Seif's improvements
└── SYSTEM_PRESENTATION.md                  # This file
```

---

