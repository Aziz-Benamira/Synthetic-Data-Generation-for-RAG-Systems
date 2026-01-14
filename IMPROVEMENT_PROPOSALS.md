# 🚀 Synthetic Data Generation - Improvement Proposals

**Date:** January 14, 2026  
**Status:** Post-Critic Calibration - Ready for Enhancement

---

## 📊 Current System Status

### ✅ **Implemented & Working:**
- **3-Agent Pipeline**: QuestionGenerator → AnswerGenerator → CriticAgent
- **Critic Agent**: Properly calibrated (33% rejection rate, target: 30-50%)
- **Hard Rules**: 5 deterministic rejection patterns
- **Adversarial Prompting**: "Flaw detector" mindset
- **Retry Loop**: Agentic workflow with max 2 retries
- **Semantic Chunking**: PDF → meaningful chunks with metadata
- **Local LLM Support**: Ollama integration (Mistral 7B + Llama 3 8B)

### ⚠️ **Current Limitations:**
1. **No diversity enforcement** - Can generate similar questions
2. **Single-hop questions only** - No multi-chunk reasoning
3. **No difficulty classification** - All questions treated equally
4. **No question type distribution** - Random mix of types
5. **Limited answer formats** - No Chain-of-Thought, no hard negatives
6. **No reformulation guidance** - Retry just regenerates blindly

---

## 🎯 Priority 1: High-Impact Agents (Week 1-2)

### 1️⃣ **DiversityManager Agent**
**Problem:** Critic rejects duplicates, but generator keeps creating similar questions.

**Solution:** Add diversity checking BEFORE answer generation (cheaper + faster).

```python
class DiversityManager:
    """
    Multi-dimensional diversity checker using semantic similarity.
    Prevents redundant questions before expensive answer generation.
    """
    
    def __init__(self, embedding_model, history_size=50):
        self.embedding_model = embedding_model
        self.question_history = []  # Last N questions
        self.type_counts = defaultdict(int)
        self.difficulty_counts = defaultdict(int)
        
    def check_diversity(self, new_question: str, question_type: str, difficulty: str) -> Tuple[bool, str]:
        """
        Returns: (is_diverse, feedback_message)
        
        Checks:
        1. Content similarity (cosine < 0.85)
        2. Type distribution balance
        3. Difficulty distribution balance
        """
        # Semantic similarity check
        new_embedding = self.embedding_model.embed(new_question)
        for old_q in self.question_history:
            similarity = cosine_similarity(new_embedding, old_q['embedding'])
            if similarity > 0.85:
                return False, f"Too similar to: '{old_q['text'][:50]}...' (similarity: {similarity:.2f})"
        
        # Type balance check
        total = len(self.question_history)
        if total > 0:
            type_proportion = self.type_counts[question_type] / total
            if type_proportion > 0.30:  # Max 30% of any type
                return False, f"Type '{question_type}' overrepresented ({type_proportion:.1%}). Try: {self._suggest_underrepresented_type()}"
        
        # Difficulty balance check (30% easy, 50% medium, 20% hard)
        difficulty_targets = {"easy": 0.30, "medium": 0.50, "hard": 0.20}
        if total > 10:  # Start checking after 10 questions
            diff_proportion = self.difficulty_counts[difficulty] / total
            target = difficulty_targets.get(difficulty, 0.33)
            if diff_proportion > target + 0.15:
                return False, f"Difficulty '{difficulty}' overrepresented. Try: {self._suggest_difficulty()}"
        
        return True, "✓ Diverse"
    
    def add_to_history(self, question: str, question_type: str, difficulty: str, embedding):
        """Store accepted question"""
        self.question_history.append({
            'text': question,
            'type': question_type,
            'difficulty': difficulty,
            'embedding': embedding
        })
        self.type_counts[question_type] += 1
        self.difficulty_counts[difficulty] += 1
        
        # Keep only last N questions
        if len(self.question_history) > 50:
            removed = self.question_history.pop(0)
            self.type_counts[removed['type']] -= 1
            self.difficulty_counts[removed['difficulty']] -= 1
```

**Pipeline Integration:**
```python
# In DatasetPipeline._generate_questions():
questions = self.question_generator.generate_from_chunk(chunk)

for question in questions:
    # NEW: Diversity check before answer generation
    is_diverse, feedback = self.diversity_manager.check_diversity(
        new_question=question,
        question_type=self._classify_question_type(question),
        difficulty=self._estimate_difficulty(question)
    )
    
    if not is_diverse:
        logger.info(f"[DIVERSITY] Rejected: {feedback}")
        continue  # Skip this question, don't generate answer
    
    # Only proceed if diverse
    answer = self.answer_generator.generate(question, chunk)
    # ... rest of pipeline
```

**Impact:**
- ✅ Prevents 40-60% wasted answer generations
- ✅ Dataset has better coverage (not all "What is..." questions)
- ✅ Cheaper (diversity check = 1 embedding, answer gen = 1 full LLM call)

---

### 2️⃣ **QuestionTypeClassifier**
**Problem:** No control over question type distribution.

**Solution:** Lightweight classifier + distribution targets.

```python
class QuestionTypeClassifier:
    """
    Rule-based classifier for 7 question types.
    Fast, deterministic, no LLM calls needed.
    """
    
    QUESTION_TYPES = {
        "factoid": ["qui", "quoi", "où", "quand", "combien"],  # who, what, where, when, how many
        "definition": ["qu'est-ce que", "définir", "signifie", "désigne"],
        "comparison": ["différence", "comparer", "distinction", "versus", "contrairement"],
        "explanation": ["pourquoi", "comment", "expliquer", "raison"],
        "application": ["appliquer", "utiliser", "exemple", "cas"],
        "calculation": ["calculer", "déterminer", "trouver la valeur", "résoudre"],
        "analysis": ["analyser", "évaluer", "critique", "juger", "interpréter"]
    }
    
    def classify(self, question: str) -> str:
        """Returns question type based on keywords"""
        question_lower = question.lower()
        
        # Check each type's patterns
        for qtype, patterns in self.QUESTION_TYPES.items():
            if any(pattern in question_lower for pattern in patterns):
                return qtype
        
        return "other"
    
    def suggest_underrepresented_type(self, type_counts: Dict[str, int]) -> str:
        """Return the least common type"""
        if not type_counts:
            return "factoid"
        return min(type_counts, key=type_counts.get)
```

**Integration with Generator:**
```python
# In QuestionGenerator SYSTEM_PROMPT:
f"""
DISTRIBUTION GUIDELINES:
- Current type distribution: {self._format_distribution()}
- PRIORITIZE generating: {underrepresented_type} questions
- AVOID generating: {overrepresented_type} questions

Examples of {underrepresented_type} questions:
{self._get_type_examples(underrepresented_type)}
"""
```

**Impact:**
- ✅ Balanced dataset (not 80% definitions, 5% analysis)
- ✅ Better RAG evaluation (tests different reasoning types)

---

### 3️⃣ **DifficultyEstimator Agent**
**Problem:** All questions treated equally, but some are trivial, others complex.

**Solution:** Automatic difficulty grading for balanced distribution.

```python
class DifficultyEstimator:
    """
    Estimates question difficulty using multiple heuristics.
    No LLM needed - fast rule-based system.
    """
    
    def estimate(self, question: str, chunk: SemanticChunk) -> Tuple[str, float]:
        """
        Returns: (difficulty_label, confidence_score)
        difficulty_label: "easy" | "medium" | "hard"
        """
        score = 0.0
        
        # Factor 1: Question complexity (word count, structure)
        word_count = len(question.split())
        if word_count < 8:
            score += 0.0  # "What is X?" = easy
        elif word_count < 15:
            score += 0.5
        else:
            score += 1.0  # Long, complex phrasing = hard
        
        # Factor 2: Question type
        qtype = self.classifier.classify(question)
        type_difficulty = {
            "factoid": 0.0,
            "definition": 0.2,
            "comparison": 0.6,
            "explanation": 0.7,
            "application": 0.8,
            "calculation": 0.9,
            "analysis": 1.0
        }
        score += type_difficulty.get(qtype, 0.5)
        
        # Factor 3: Requires multi-sentence answer?
        if "expliquer" in question or "pourquoi" in question or "comment" in question:
            score += 0.5
        
        # Factor 4: Contains technical terms not in chunk?
        question_terms = set(self._extract_technical_terms(question))
        chunk_terms = set(self._extract_technical_terms(chunk.content))
        if len(question_terms - chunk_terms) > 2:
            score += 0.3  # Requires external knowledge
        
        # Normalize to 0-1
        normalized = min(score / 3.0, 1.0)
        
        # Map to labels
        if normalized < 0.4:
            return "easy", normalized
        elif normalized < 0.7:
            return "medium", normalized
        else:
            return "hard", normalized
    
    def _extract_technical_terms(self, text: str) -> List[str]:
        """Extract capitalized terms, Greek letters, formulas"""
        # Simple heuristic: words starting with uppercase, multi-syllable
        words = text.split()
        technical = [w for w in words if len(w) > 4 and w[0].isupper()]
        return technical
```

**Impact:**
- ✅ Dataset suitable for testing at multiple difficulty levels
- ✅ Can filter by difficulty for different use cases
- ✅ Helps diversity manager balance distribution

---

## 🎯 Priority 2: Answer Enhancement Agents (Week 3-4)

### 4️⃣ **ChainOfThoughtGenerator**
**Problem:** Answers are direct, no reasoning steps shown.

**Solution:** Add CoT reasoning for complex questions.

```python
class ChainOfThoughtGenerator:
    """
    Adds step-by-step reasoning to answers for complex questions.
    Improves answer quality and makes reasoning transparent.
    """
    
    def should_use_cot(self, question: str, difficulty: str) -> bool:
        """Only use CoT for medium/hard questions of certain types"""
        if difficulty == "easy":
            return False
        
        qtype = self.classifier.classify(question)
        cot_types = ["explanation", "comparison", "application", "analysis", "calculation"]
        return qtype in cot_types
    
    def generate_with_cot(self, question: str, chunk: SemanticChunk) -> Dict:
        """
        Returns: {
            "final_answer": str,
            "reasoning_steps": List[str],
            "confidence": float
        }
        """
        prompt = f"""
Question: {question}

Context: {chunk.content}

Répondez en suivant ces étapes:
1. IDENTIFIER les concepts clés dans la question
2. LOCALISER les informations pertinentes dans le contexte
3. RAISONNER étape par étape
4. CONCLURE avec la réponse finale

Format:
Étape 1: [Concepts identifiés]
Étape 2: [Informations localisées]
Étape 3: [Raisonnement]
Réponse finale: [Réponse concise]
"""
        
        response = self.llm_client.generate(prompt)
        
        # Parse reasoning steps
        steps = self._parse_cot_response(response)
        
        return {
            "final_answer": steps["final_answer"],
            "reasoning_steps": steps["steps"],
            "confidence": self._estimate_confidence(steps)
        }
```

**Impact:**
- ✅ Better training data for reasoning-capable RAG systems
- ✅ Easier to debug answer quality issues
- ✅ Can use reasoning steps as additional supervision signal

---

### 5️⃣ **HardNegativeGenerator**
**Problem:** Only positive examples (correct answers), no contrastive learning.

**Solution:** Generate plausible-but-wrong answers for each question.

```python
class HardNegativeGenerator:
    """
    Generates 2-3 hard negative answers per question.
    Useful for training ranking models and contrastive learning.
    """
    
    STRATEGIES = [
        "partial_answer",      # Answers only part of the question
        "adjacent_confusion",  # Uses info from nearby chunk
        "temporal_confusion",  # Mixes past/future concepts
        "entity_swap",         # Swaps similar entities
        "negation",           # Negates the correct statement
    ]
    
    def generate(self, question: str, correct_answer: str, chunk: SemanticChunk, 
                 adjacent_chunks: List[SemanticChunk]) -> List[Dict]:
        """
        Returns list of hard negatives:
        [{
            "text": "...",
            "strategy": "adjacent_confusion",
            "plausibility_score": 0.7
        }]
        """
        negatives = []
        
        # Strategy 1: Partial answer
        if len(correct_answer.split('.')) > 1:
            partial = correct_answer.split('.')[0] + '.'
            negatives.append({
                "text": partial,
                "strategy": "partial_answer",
                "plausibility_score": 0.8
            })
        
        # Strategy 2: Adjacent chunk confusion
        if adjacent_chunks:
            adjacent_content = adjacent_chunks[0].content[:200]
            prompt = f"""
Question: {question}
Correct answer: {correct_answer}
Adjacent context: {adjacent_content}

Generate a PLAUSIBLE BUT INCORRECT answer that mixes information from the adjacent context.
The answer should sound believable but be factually wrong for THIS question.
"""
            negative_answer = self.llm_client.generate(prompt)
            negatives.append({
                "text": negative_answer,
                "strategy": "adjacent_confusion",
                "plausibility_score": 0.6
            })
        
        # Strategy 3: Entity swap (for factoid questions)
        entities = self._extract_entities(correct_answer)
        if len(entities) > 1:
            swapped = correct_answer
            for i in range(len(entities)-1):
                swapped = swapped.replace(entities[i], entities[i+1])
            negatives.append({
                "text": swapped,
                "strategy": "entity_swap",
                "plausibility_score": 0.5
            })
        
        return negatives
```

**Dataset Format with Negatives:**
```json
{
  "question": "Qu'est-ce qu'une tribu en théorie des probabilités?",
  "correct_answer": "Une tribu est une famille de sous-ensembles...",
  "hard_negatives": [
    {
      "text": "Une tribu est simplement un ensemble de points.",
      "strategy": "partial_answer",
      "plausibility": 0.8
    },
    {
      "text": "Une tribu est une mesure de probabilité normalisée.",
      "strategy": "entity_swap",
      "plausibility": 0.6
    }
  ],
  "metadata": {
    "type": "definition",
    "difficulty": "medium"
  }
}
```

**Impact:**
- ✅ Training data for contrastive learning
- ✅ Better RAG evaluation (test if model picks correct answer vs plausible wrong one)
- ✅ Can measure retrieval precision (do hard negatives get retrieved?)

---

## 🎯 Priority 3: Meta-Agents (Week 5-6)

### 6️⃣ **ReformulatorAgent**
**Problem:** When critic rejects, generator blindly regenerates. No guidance on HOW to improve.

**Solution:** Reformulator analyzes rejection reasons and suggests specific fixes.

```python
class ReformulatorAgent:
    """
    Analyzes critic feedback and provides actionable reformulation guidance.
    Acts as a "coach" between Critic and Generator.
    """
    
    def analyze_rejection(self, evaluation: CriticEvaluation) -> Dict[str, str]:
        """
        Returns: {
            "issue_summary": "...",
            "reformulation_strategy": "...",
            "example_fix": "..."
        }
        """
        failed_criteria = evaluation.failed_criteria
        
        strategies = []
        
        # Anchoring failures → Use only chunk content
        if "anchoring" in failed_criteria:
            strategies.append({
                "criterion": "anchoring",
                "issue": "Answer contains information not in chunk",
                "fix": "Rewrite answer using ONLY facts from the provided chunk. Remove examples, generalizations, or external knowledge.",
                "example": f"Instead of adding examples, quote directly: 'According to the text, [quote]...'"
            })
        
        # Local answerability → Simplify question
        if "local_answerability" in failed_criteria:
            strategies.append({
                "criterion": "local_answerability",
                "issue": "Question cannot be answered from this chunk alone",
                "fix": "Either: (1) Make question more specific to chunk content, or (2) Mark as multi-hop and skip",
                "example": "Change 'Why does X happen?' to 'What does this section say about X?'"
            })
        
        # Completeness → Add more detail
        if "completeness" in failed_criteria:
            strategies.append({
                "criterion": "completeness",
                "issue": "Answer is too short or misses key points",
                "fix": "Expand answer to address all parts of the question. Check if question has multiple sub-questions.",
                "example": "For 'What and why...?', answer both the 'what' part AND the 'why' part"
            })
        
        return {
            "issue_summary": f"Failed {len(failed_criteria)} criteria: {', '.join(failed_criteria)}",
            "strategies": strategies,
            "priority_fix": strategies[0] if strategies else None
        }
    
    def create_reformulation_prompt(self, original_qa: QAPair, rejection_analysis: Dict) -> str:
        """Generate a detailed prompt for the generator to fix the QA pair"""
        return f"""
REFORMULATION REQUEST

Original Question: {original_qa.question}
Original Answer: {original_qa.answer}

REJECTION REASON: {rejection_analysis['issue_summary']}

SPECIFIC FIXES NEEDED:
{self._format_strategies(rejection_analysis['strategies'])}

Please regenerate the question and/or answer following these guidelines.
Focus on: {rejection_analysis['priority_fix']['criterion']}
"""
```

**Integration:**
```python
# In DatasetPipeline retry loop:
if not evaluation.decision.is_valid:
    # NEW: Get reformulation guidance
    reformulation = self.reformulator.analyze_rejection(evaluation)
    
    # Generate with specific guidance (not blind regeneration)
    new_qa = self.question_generator.regenerate_with_guidance(
        original_qa=current_qa,
        chunk=chunk,
        reformulation_guidance=reformulation
    )
```

**Impact:**
- ✅ Faster convergence (fix specific issues, not random retry)
- ✅ Higher retry success rate (currently fails often)
- ✅ Better learning signal for generator improvements

---

### 7️⃣ **MetaEvaluatorAgent**
**Problem:** Critic might be miscalibrated over time, no feedback loop.

**Solution:** Periodically evaluate the critic's performance.

```python
class MetaEvaluatorAgent:
    """
    Evaluates the critic agent's calibration.
    Checks for:
    - Inter-rater reliability (consistency)
    - Criterion correlation (are some criteria redundant?)
    - Rejection rate drift (is critic getting too strict/lenient?)
    """
    
    def evaluate_critic_batch(self, qa_pairs: List[QAPair], 
                              evaluations: List[CriticEvaluation]) -> Dict:
        """
        Analyze last N evaluations for calibration issues.
        """
        # Check rejection rate trend
        rejection_rates = []
        window_size = 10
        for i in range(0, len(evaluations), window_size):
            window = evaluations[i:i+window_size]
            rejects = sum(1 for e in window if not e.decision.is_valid)
            rejection_rates.append(rejects / len(window))
        
        # Check if drift (getting too strict or lenient over time)
        if len(rejection_rates) > 5:
            trend = self._calculate_trend(rejection_rates)
            if trend > 0.1:
                warning = "⚠️ Critic becoming stricter over time"
            elif trend < -0.1:
                warning = "⚠️ Critic becoming more lenient over time"
            else:
                warning = None
        
        # Check criterion correlation
        # If two criteria always fail together, they might be redundant
        criterion_failures = self._analyze_criterion_correlations(evaluations)
        
        return {
            "rejection_rate_mean": np.mean(rejection_rates),
            "rejection_rate_std": np.std(rejection_rates),
            "trend_warning": warning,
            "criterion_correlations": criterion_failures,
            "recommendation": self._generate_recommendation(rejection_rates, criterion_failures)
        }
    
    def _generate_recommendation(self, rates, correlations) -> str:
        """Suggest prompt adjustments if needed"""
        mean_rate = np.mean(rates)
        
        if mean_rate < 0.25:
            return "Critic may be too lenient. Consider: (1) Adding more hard rules, (2) Increasing threshold to 0.85"
        elif mean_rate > 0.55:
            return "Critic may be too strict. Consider: (1) Reviewing hard rules, (2) Lowering threshold to 0.75"
        else:
            return "✓ Critic calibration looks good"
```

**Usage:**
```python
# Run every 50 questions
if len(dataset) % 50 == 0:
    meta_report = meta_evaluator.evaluate_critic_batch(
        qa_pairs=dataset[-50:],
        evaluations=critic_evaluations[-50:]
    )
    
    logger.info(f"[META-EVAL] {meta_report['recommendation']}")
    
    if meta_report['trend_warning']:
        logger.warning(meta_report['trend_warning'])
```

**Impact:**
- ✅ Catches critic drift early
- ✅ Maintains 30-50% rejection rate over long runs
- ✅ Data-driven prompt adjustments

---

## 🎯 Priority 4: Advanced Features (Week 7+)

### 8️⃣ **MultiHopQuestionGenerator**
**Problem:** All questions answerable from single chunk. Real RAG needs multi-chunk reasoning.

**Solution:** Generate questions requiring 2-3 chunks.

```python
class MultiHopQuestionGenerator:
    """
    Generates questions that require synthesizing information from multiple chunks.
    Example: "How does concept from Chapter 2 relate to theorem from Chapter 5?"
    """
    
    def find_connectable_chunks(self, chunk: SemanticChunk, 
                                all_chunks: List[SemanticChunk]) -> List[Tuple[SemanticChunk, str]]:
        """
        Find chunks that have conceptual connections.
        Returns: [(connected_chunk, relationship_type)]
        """
        connections = []
        
        # Extract key concepts from current chunk
        concepts_1 = self._extract_concepts(chunk)
        
        for other_chunk in all_chunks:
            if other_chunk.chunk_id == chunk.chunk_id:
                continue
            
            # Check for relationships
            concepts_2 = self._extract_concepts(other_chunk)
            
            # Type 1: Prerequisite (Chapter N references Chapter N-1)
            if self._is_prerequisite(chunk, other_chunk):
                connections.append((other_chunk, "prerequisite"))
            
            # Type 2: Application (theorem → example)
            elif self._is_application(chunk, other_chunk):
                connections.append((other_chunk, "application"))
            
            # Type 3: Comparison (two similar concepts in different chapters)
            elif len(concepts_1 & concepts_2) > 2:
                connections.append((other_chunk, "comparison"))
        
        return connections
    
    def generate_multi_hop(self, chunk1: SemanticChunk, chunk2: SemanticChunk, 
                          relationship: str) -> Dict:
        """
        Generate question requiring both chunks.
        """
        prompts_by_type = {
            "prerequisite": f"""
Chunk 1 (foundational): {chunk1.content[:300]}
Chunk 2 (builds on it): {chunk2.content[:300]}

Generate a question that requires understanding BOTH chunks.
Example: "How does [concept from chunk2] build upon [concept from chunk1]?"
""",
            "comparison": f"""
Chunk 1: {chunk1.content[:300]}
Chunk 2: {chunk2.content[:300]}

Generate a comparative question.
Example: "Compare [concept from chunk1] and [concept from chunk2]. What are their similarities and differences?"
""",
        }
        
        question = self.llm_client.generate(prompts_by_type[relationship])
        
        return {
            "question": question,
            "required_chunks": [chunk1.chunk_id, chunk2.chunk_id],
            "hop_count": 2,
            "relationship_type": relationship
        }
```

**Impact:**
- ✅ Tests true RAG capabilities (retrieval + reasoning)
- ✅ More realistic evaluation (academic questions often span sections)
- ✅ Catches poor retrieval (if RAG retrieves only 1 of 2 needed chunks)

---

### 9️⃣ **Bloom's Taxonomy Classifier**
**Problem:** No explicit cognitive level tracking.

**Solution:** Classify questions by Bloom's taxonomy level.

```python
class BloomsTaxonomyClassifier:
    """
    Classifies questions into 6 Bloom's levels:
    1. Remember (recall facts)
    2. Understand (explain concepts)
    3. Apply (use in new situations)
    4. Analyze (break down, compare)
    5. Evaluate (judge, critique)
    6. Create (design, synthesize)
    """
    
    KEYWORD_PATTERNS = {
        "remember": ["définir", "lister", "nommer", "identifier", "qu'est-ce que"],
        "understand": ["expliquer", "décrire", "résumer", "interpréter", "pourquoi"],
        "apply": ["appliquer", "utiliser", "démontrer", "calculer", "résoudre"],
        "analyze": ["analyser", "comparer", "différence", "relation", "distinguer"],
        "evaluate": ["évaluer", "juger", "critiquer", "justifier", "argumenter"],
        "create": ["concevoir", "créer", "proposer", "formuler", "développer"]
    }
    
    def classify(self, question: str) -> Tuple[str, int]:
        """
        Returns: (level_name, level_number)
        level_number: 1-6 (1=remember, 6=create)
        """
        question_lower = question.lower()
        
        for level_num, (level_name, patterns) in enumerate(self.KEYWORD_PATTERNS.items(), 1):
            if any(pattern in question_lower for pattern in patterns):
                return level_name, level_num
        
        return "understand", 2  # Default to level 2
```

**Dataset Enhancement:**
```json
{
  "question": "Comparez les propriétés d'une tribu et d'une σ-algèbre",
  "bloom_level": "analyze",
  "bloom_level_num": 4,
  "cognitive_demand": "high"
}
```

**Impact:**
- ✅ Can filter dataset by cognitive level
- ✅ Ensure distribution across Bloom's levels
- ✅ Better evaluation (test recall vs analysis vs creation)

---

### 🔟 **ExplanationEnhancer**
**Problem:** Answers don't explain WHY, just state facts.

**Solution:** Add pedagogical explanations.

```python
class ExplanationEnhancer:
    """
    Adds 'why this matters' and 'intuition' sections to answers.
    Makes dataset more educational.
    """
    
    def enhance(self, qa_pair: QAPair, chunk: SemanticChunk) -> Dict:
        """
        Returns: {
            "base_answer": "...",
            "intuition": "Think of it as...",
            "why_it_matters": "This is important because...",
            "common_mistakes": "Students often confuse..."
        }
        """
        prompt = f"""
Question: {qa_pair.question}
Answer: {qa_pair.answer}

Add pedagogical enhancements:
1. INTUITION: A simple analogy or "think of it as..." explanation
2. WHY IT MATTERS: Why is this concept important?
3. COMMON MISTAKES: What do students typically get wrong?

Format:
Intuition: ...
Importance: ...
Common mistakes: ...
"""
        
        enhanced = self.llm_client.generate(prompt)
        parsed = self._parse_enhancements(enhanced)
        
        return {
            "base_answer": qa_pair.answer,
            **parsed
        }
```

---

## 📋 Implementation Roadmap

### **Phase 1: Quick Wins** (Week 1)
- [ ] Implement **QuestionTypeClassifier** (1 day)
- [ ] Implement **DifficultyEstimator** (1 day)
- [ ] Add metadata to pipeline (1 day)
- [ ] Update dataset export with metadata (0.5 days)
- [ ] Test with 50 questions (0.5 days)

**Effort:** 4 days  
**Impact:** 🟢🟢🟢 High (immediate dataset improvement)

---

### **Phase 2: Diversity & Quality** (Week 2)
- [ ] Implement **DiversityManager** (2 days)
- [ ] Implement **ReformulatorAgent** (2 days)
- [ ] Integrate into retry loop (1 day)
- [ ] Test with 100 questions (1 day)

**Effort:** 6 days  
**Impact:** 🟢🟢🟢 High (better quality, fewer retries)

---

### **Phase 3: Answer Enhancement** (Week 3)
- [ ] Implement **ChainOfThoughtGenerator** (2 days)
- [ ] Implement **HardNegativeGenerator** (2 days)
- [ ] Update dataset schema (1 day)
- [ ] Test generation pipeline (1 day)

**Effort:** 6 days  
**Impact:** 🟢🟢 Medium-High (richer training data)

---

### **Phase 4: Meta-Evaluation** (Week 4)
- [ ] Implement **MetaEvaluatorAgent** (2 days)
- [ ] Add calibration monitoring (1 day)
- [ ] Create calibration dashboard (1 day)

**Effort:** 4 days  
**Impact:** 🟢 Medium (long-term quality assurance)

---

### **Phase 5: Advanced Features** (Week 5-6)
- [ ] Implement **MultiHopQuestionGenerator** (3 days)
- [ ] Implement **BloomsTaxonomyClassifier** (1 day)
- [ ] Implement **ExplanationEnhancer** (2 days)
- [ ] Full integration test (2 days)

**Effort:** 8 days  
**Impact:** 🟢🟢 Medium-High (state-of-the-art dataset)

---

## 🎯 Quick Start: Implement Priority 1 Today

To get started immediately, I recommend implementing these three agents in this order:

### 1. **QuestionTypeClassifier** (2 hours)
- File: `src/agents/question_type_classifier.py`
- Easy, no LLM needed
- Immediate value

### 2. **DifficultyEstimator** (3 hours)
- File: `src/agents/difficulty_estimator.py`
- Reuses classifier
- Fast heuristics

### 3. **DiversityManager** (4 hours)
- File: `src/agents/diversity_manager.py`
- Needs sentence-transformers (already installed)
- Biggest impact

**Total time: 1 day**  
**Result: Dataset with balanced types, difficulties, and no duplicates**

Would you like me to implement any of these agents now?
