"""
Detailed Critic Test - Shows Full Evaluation for Each QA Pair
==============================================================

This test runs the pipeline with detailed logging to show:
- Each question generated
- Each answer generated  
- Complete critic evaluation (all 5 criteria)
- Pass/Reject decision with reasoning
- Retry loop in action
"""

import sys
sys.path.insert(0, 'src')

from openai import OpenAI
from orchestrator.pipeline import DatasetPipeline, PipelineConfig
from datetime import datetime
import json

print("=" * 80)
print("🔍 DETAILED CRITIC EVALUATION TEST")
print("=" * 80)
print()

# Create Ollama client
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Create pipeline
config = PipelineConfig(
    pdf_path="data/pdfs/M2_cours.pdf",
    output_dir="output/critic_test",
    generator_model="mistral:latest",
    critic_model="llama3:8b",
    language="fr",
    questions_per_chunk=2,  # 2 questions per chunk for faster test
    max_retries=2,
    max_chunks=2,  # Only 2 chunks for detailed view
    temperature=0.7
)

print(f"Configuration:")
print(f"  Generator: {config.generator_model}")
print(f"  Critic: {config.critic_model}")
print(f"  Max retries: {config.max_retries}")
print(f"  Questions per chunk: {config.questions_per_chunk}")
print(f"  Chunks to process: {config.max_chunks}")
print()

# Custom pipeline to capture detailed evaluations
from chunking.semantic_chunker import SemanticChunker
from agents.question_generator import QuestionGenerator
from agents.answer_generator import AnswerGenerator
from agents.critic_agent import CriticAgent, FinalDecision
import time

print(f"[{datetime.now().strftime('%H:%M:%S')}] Initializing agents...")
print()

# Initialize agents
chunker = SemanticChunker(config.pdf_path)
q_generator = QuestionGenerator(client, config.generator_model, config.language, config.temperature)
a_generator = AnswerGenerator(client, config.generator_model, config.language, temperature=0.3)
critic = CriticAgent(client, config.critic_model, config.language, temperature=0.2, strict_mode=True)

print("✅ All agents initialized")
print()

# Get chunks
print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading PDF and extracting chunks...")
all_chunks = chunker.chunk_document()
chunks = all_chunks[:config.max_chunks]
print(f"✅ {len(chunks)} chunks ready")
print()

# Track statistics
total_generated = 0
total_passed = 0
total_rejected = 0
total_retries = 0
evaluation_details = []

# Process each chunk
for chunk_idx, chunk in enumerate(chunks, 1):
    print("=" * 80)
    print(f"CHUNK {chunk_idx}/{len(chunks)}: {chunk.chunk_id}")
    print("=" * 80)
    print(f"Type: {chunk.semantic_type}")
    print(f"Section: {chunk.section_title}")
    print(f"Length: {len(chunk.content)} chars")
    print(f"Content preview: {chunk.content[:200]}...")
    print()
    
    # Generate questions
    print(f"📝 Generating {config.questions_per_chunk} questions...")
    questions = q_generator.generate_from_chunk(chunk, config.questions_per_chunk)
    print(f"✅ {len(questions)} questions generated")
    print()
    
    # Process each question
    for q_idx, question in enumerate(questions, 1):
        print("─" * 80)
        print(f"QUESTION {q_idx}/{len(questions)}")
        print("─" * 80)
        print(f"Q: {question.question}")
        print(f"Type: {question.question_type} | Difficulty: {question.difficulty}")
        print()
        
        total_generated += 1
        attempt = 0
        max_attempts = config.max_retries + 1
        
        current_question = question
        current_answer = None
        
        while attempt < max_attempts:
            attempt += 1
            
            # Generate answer
            if attempt == 1:
                print(f"💬 Generating answer (attempt {attempt})...")
            else:
                print(f"💬 Regenerating answer (attempt {attempt})...")
                
            current_answer = a_generator.generate_answer(current_question, chunk)
            
            print(f"A: {current_answer.answer[:300]}...")
            print()
            
            # Create QA pair
            from agents.answer_generator import QAPair
            qa_pair = QAPair.from_question_and_answer(current_question, current_answer)
            
            # Evaluate with critic
            print(f"🔍 CRITIC EVALUATION (attempt {attempt}/{max_attempts}):")
            print("─" * 40)
            
            evaluation = critic.evaluate(qa_pair, chunk)
            
            # Show detailed evaluation
            print(f"Decision: {evaluation.decision.value.upper()}")
            print(f"Overall Score: {evaluation.overall_score:.2f}")
            print()
            
            print("Criteria Scores:")
            for criterion_name, criterion_eval in evaluation.criteria_evaluations.items():
                emoji = "✅" if criterion_eval.result.value == "pass" else "❌"
                print(f"  {emoji} {criterion_name.upper()}: {criterion_eval.score:.2f}")
                print(f"     {criterion_eval.explanation}")
                if criterion_eval.evidence:
                    print(f"     Evidence: {criterion_eval.evidence[:2]}")
                print()
            
            # Check decision
            if evaluation.decision == FinalDecision.PASS:
                total_passed += 1
                if attempt > 1:
                    print(f"✅ PASSED after {attempt-1} retry attempt(s)")
                else:
                    print(f"✅ PASSED on first attempt")
                
                # Save evaluation details
                evaluation_details.append({
                    "chunk": chunk.chunk_id,
                    "question": question.question,
                    "answer": current_answer.answer,
                    "attempts": attempt,
                    "decision": "PASS",
                    "overall_score": evaluation.overall_score,
                    "criteria": {
                        name: {
                            "score": eval.score,
                            "result": eval.result.value,
                            "explanation": eval.explanation
                        }
                        for name, eval in evaluation.criteria_evaluations.items()
                    }
                })
                break
                
            else:
                # REJECTED
                if attempt < max_attempts:
                    total_retries += 1
                    print(f"❌ REJECTED - Will retry ({attempt}/{config.max_retries})")
                    print()
                    print("Feedback for regeneration:")
                    feedback = critic.format_feedback_for_retry(evaluation)
                    print(feedback)
                    print()
                    
                    # For retry, we keep same question but regenerate answer
                    # (In real pipeline, both could be regenerated)
                    time.sleep(1)  # Rate limiting
                    
                else:
                    # Max retries exceeded
                    total_rejected += 1
                    print(f"❌ REJECTED DEFINITIVELY after {config.max_retries} retries")
                    
                    evaluation_details.append({
                        "chunk": chunk.chunk_id,
                        "question": question.question,
                        "answer": current_answer.answer,
                        "attempts": attempt,
                        "decision": "REJECT",
                        "overall_score": evaluation.overall_score,
                        "criteria": {
                            name: {
                                "score": eval.score,
                                "result": eval.result.value,
                                "explanation": eval.explanation
                            }
                            for name, eval in evaluation.criteria_evaluations.items()
                        }
                    })
                    break
        
        print()
        time.sleep(0.5)  # Rate limiting between questions

# Final summary
print()
print("=" * 80)
print("📊 FINAL SUMMARY")
print("=" * 80)
print()
print(f"Total QA pairs generated: {total_generated}")
print(f"  ✅ Passed: {total_passed} ({total_passed/total_generated*100:.1f}%)")
print(f"  ❌ Rejected: {total_rejected} ({total_rejected/total_generated*100:.1f}%)")
print(f"  🔄 Total retry attempts: {total_retries}")
print()

if total_retries > 0:
    initial_rejection_rate = (total_retries / total_generated) * 100
    print(f"🎯 Initial rejection rate: {initial_rejection_rate:.1f}%")
    if 30 <= initial_rejection_rate <= 50:
        print(f"   ✅ Within target range (30-50%)!")
    elif initial_rejection_rate < 30:
        print(f"   ⚠️  Below target (too lenient)")
    else:
        print(f"   ⚠️  Above target (too strict)")
print()

# Save detailed results
output_file = "output/critic_test/detailed_evaluations.json"
import os
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump({
        "summary": {
            "total_generated": total_generated,
            "total_passed": total_passed,
            "total_rejected": total_rejected,
            "total_retries": total_retries,
            "initial_rejection_rate": total_retries / total_generated * 100 if total_generated > 0 else 0
        },
        "evaluations": evaluation_details
    }, f, indent=2, ensure_ascii=False)

print(f"💾 Detailed evaluations saved to: {output_file}")
print()
print("✅ Test complete!")
