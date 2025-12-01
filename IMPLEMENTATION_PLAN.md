# Multi-Agent RAG System - Implementation Plan

## Current Status

We have already implemented:
1. Configuration system (config.py) - Manages all settings via Pydantic
2. PDF Processor (pdf_processor.py) - Extracts text, TOC, definitions, equations
3. Vector Store (vector_store.py) - ChromaDB integration (need to verify)
4. MCP Server (textbook_mcp_server.py) - Data access layer (need to verify)

## What We Need to Build (In Order)

### Phase 1: Foundation (Week 1-2)
**Goal:** Get data processing working end-to-end

1. **Test and fix PDF processor**
   - Load a sample PDF
   - Extract curriculum structure
   - Verify text extraction works correctly

2. **Implement basic chunking strategy**
   - Start with simple fixed-size chunks (500 chars, 50 overlap)
   - Add metadata (page, chapter, section)
   - Test on sample document

3. **Setup vector store**
   - Initialize ChromaDB
   - Index chunks with embeddings
   - Test retrieval with simple queries

4. **Verify MCP server**
   - Check if it provides the 5 tools we need
   - Test each tool individually
   - Fix any issues

**Deliverable:** Can load PDF, chunk it, index it, and retrieve relevant passages

---

### Phase 2: First Agent - Question Generator (Week 3-4)
**Goal:** Generate diverse questions from document sections

1. **Design Question Generator agent**
   - Input: Section content + context
   - Output: Question + metadata (type, difficulty)
   - Prompt engineering for question quality

2. **Implement question type classifier**
   - 7 types: factoid, definition, comparison, reasoning, calculation, application, analysis
   - Rule-based classification using patterns

3. **Add basic diversity checker**
   - Cosine similarity to avoid duplicate questions
   - Track question types distribution

4. **Test on sample section**
   - Generate 10 questions from one chapter
   - Manually evaluate quality
   - Iterate on prompts

**Deliverable:** Agent that generates diverse questions from any section

---

### Phase 3: Second Agent - Answer Generator (Week 5-6)
**Goal:** Generate accurate answers with retrieval

1. **Design Answer Generator agent**
   - Input: Question
   - Process: Retrieve context via MCP, generate answer
   - Output: Answer + retrieved chunks + confidence

2. **Implement HyDE query expansion**
   - Generate hypothetical answer
   - Use it to improve retrieval
   - Compare with direct question embedding

3. **Add Self-RAG pre-filtering**
   - Check if retrieved chunks are relevant (ISREL)
   - Check if answer is supported (ISSUP)
   - Trigger regeneration if quality too low

4. **Test on generated questions**
   - Generate answers for questions from Phase 2
   - Manually verify accuracy
   - Measure retrieval quality

**Deliverable:** Agent that generates accurate, grounded answers

---

### Phase 4: Third Agent - Critic (Week 7-8)
**Goal:** Evaluate and provide feedback

1. **Integrate RAGAS metrics**
   - Faithfulness: answer supported by context
   - Answer relevancy: answer addresses question
   - Context precision: retrieved chunks relevant

2. **Implement Constitutional AI validator**
   - Define academic constitution (6 principles)
   - Check compliance with principles
   - Generate specific feedback

3. **Build Academic Validator**
   - Completeness: covers key concepts
   - Terminology: uses correct terms
   - Bloom's taxonomy: appropriate cognitive level

4. **Test evaluation pipeline**
   - Evaluate Phase 3 Q&A pairs
   - Analyze rejection reasons
   - Tune thresholds

**Deliverable:** Critic that reliably evaluates Q&A quality

---

### Phase 5: Reflexion Loop (Week 9-10)
**Goal:** Enable iterative improvement

1. **Design reflexion loop**
   - Critic generates structured feedback
   - Generator/Answerer use feedback to improve
   - Maximum 3 iterations

2. **Implement feedback mechanism**
   - Feedback includes: what's wrong, why, how to fix
   - Agents incorporate feedback in prompts

3. **Add iteration tracking**
   - Store all versions
   - Track score improvements
   - Stop when converged or max iterations

4. **Test reflexion**
   - Run on 10 Q&A pairs
   - Measure score improvement per iteration
   - Analyze when it helps vs doesn't

**Deliverable:** Full reflexion loop improving quality

---

### Phase 6: Orchestration (Week 11-12)
**Goal:** Coordinate all agents

1. **Design LangGraph state machine**
   - States: select_section, generate_q, check_diversity, generate_a, critic, reflexion, save
   - Edges: conditional routing based on scores
   - State: all agent inputs/outputs

2. **Implement checkpointing**
   - SQLite-based checkpoint after each step
   - Can resume from any point
   - Prevents data loss on crashes

3. **Add batch processing**
   - Process multiple questions in parallel (if budget allows)
   - Progress tracking
   - Cost monitoring

4. **Test end-to-end**
   - Generate 50 Q&A pairs from one chapter
   - Verify all components work together
   - Measure total cost and time

**Deliverable:** Complete orchestrated pipeline

---

### Phase 7: Advanced Features (Week 13-14)
**Goal:** Add remaining improvements

1. **Semantic chunking**
   - Preserve theorems, definitions, equations as atomic units
   - Add contextual headers
   - Reindex vector store

2. **Multi-dimensional diversity**
   - Track 4 dimensions: content, type, difficulty, cognitive level
   - Balance distribution across dimensions

3. **Hard negative generation**
   - Entity swap, relation inversion, adjacent topic confusion
   - Plausibility scoring

**Deliverable:** Production-quality system

---

### Phase 8: Evaluation & Documentation (Week 15-16)
**Goal:** Benchmark and document

1. **Compare with baselines**
   - LlamaIndex
   - Single-agent approach
   - Manual annotation

2. **Generate final dataset**
   - 200 Q&A pairs from full textbook
   - With hard negatives
   - JSONL format

3. **Write documentation**
   - API docs
   - Usage examples
   - Results analysis

**Deliverable:** Complete project ready for submission

---

## Recommended Starting Point

**Start with Phase 1, Task 1: Test PDF Processor**

Why start here:
- It's the foundation of everything
- Simple to test and understand
- Gives you quick feedback
- Builds confidence

Next steps:
1. Find or create a sample PDF (academic textbook, preferably physics or CS)
2. Test PDFProcessor on it
3. Verify it extracts text correctly
4. Check if TOC extraction works

Once PDF processing works, we move to chunking and indexing.

## Key Principles for Learning

1. **One component at a time** - Master each before moving on
2. **Test immediately** - Don't write code without testing it
3. **Understand why** - Ask about any concept you don't understand
4. **Iterate quickly** - Small changes, frequent tests
5. **Document learnings** - Note what works and what doesn't

Ready to start with Phase 1, Task 1?
