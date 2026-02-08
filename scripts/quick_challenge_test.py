"""
QUICK CHALLENGING TEST - No Heavy Imports
==========================================

Direct test without semantic_chunker to avoid torch import issues.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

# Mock chunk class
@dataclass
class TestChunk:
    chunk_id: str
    content: str
    chapter_title: str
    section_title: str
    page_range: Tuple[int, int] = (1, 1)
    semantic_type: str = "text"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {"source": "test_chunk"}
    
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.ollama_client import create_ollama_client, OLLAMA_MODELS
from src.agents.question_generator import QuestionGenerator
from src.agents.answer_generator import AnswerGenerator, QAPair
from src.agents.critic_agent import CriticAgent

# CHALLENGING CHUNKS
CHUNKS = [
    {
        "id": "ambiguous_1",
        "content": "Le système fonctionne de manière optimale. Il peut atteindre 95% de performance. Cette valeur dépend des conditions externes. Dans certains cas, on observe 82%. Les facteurs incluent la température et d'autres paramètres.",
        "title": "Performance du système",
        "issues": ["Vague 'Il'", "Numbers 95%, 82% without context"]
    },
    {
        "id": "vague_1",
        "content": "La méthode proposée résout le problème. Elle est plus efficace. Cependant, il faut considérer les limitations. Nous avons observé des résultats intéressants.",
        "title": "Évaluation",
        "issues": ["No concrete method", "Vague 'résultats intéressants'"]
    },
    {
        "id": "numbers_1",
        "content": "Le taux d'erreur est de 0.15 dans le cas général. Pour les cas spécifiques, il varie entre 0.08 et 0.32. 67% des échantillons répondent aux critères.",
        "title": "Analyse",
        "issues": ["Numbers without explanation", "Vague 'critères'"]
    }
]

print("=" * 80)
print("  QUICK CHALLENGING TEST - Seif's Critic vs Current")
print("=" * 80)
print()

# Connect
print("🔌 Connecting to Ollama...")
client = create_ollama_client()
print("✅ Connected!\n")

# Create agents
q_gen = QuestionGenerator(client, OLLAMA_MODELS["generator"], temperature=0.8, language="fr")
a_gen = AnswerGenerator(client, OLLAMA_MODELS["generator"], temperature=0.7, language="fr")
critic = CriticAgent(client, OLLAMA_MODELS["critic"], temperature=0.3, language="fr")

results = {"chunks": [], "total_qa": 0, "total_rejected": 0, "total_retries": 0}

for i, chunk_data in enumerate(CHUNKS, 1):
    print(f"{'─' * 80}")
    print(f"CHUNK {i}/{len(CHUNKS)}: {chunk_data['title']}")
    print(f"Expected issues: {', '.join(chunk_data['issues'])}")
    print(f"{'─' * 80}\n")
    
    chunk = TestChunk(
        chunk_id=chunk_data['id'],
        content=chunk_data['content'],
        chapter_title="Test",
        section_title=chunk_data['title']
    )
    
    # Generate 1 question
    print("   📝 Generating question...")
    questions = q_gen.generate_from_chunk(chunk, num_questions=1)
    if not questions:
        print("      ❌ No questions generated\n")
        continue
    
    q = questions[0]
    print(f"      Q: {q.question}\n")
    
    # Generate answer
    print("   💬 Generating answer...")
    answer_obj = a_gen.generate_answer(q, chunk)
    answer = answer_obj.answer if hasattr(answer_obj, 'answer') else str(answer_obj)
    print(f"      A: {answer[:80]}...\n")
    
    qa = QAPair.from_question_and_answer(q, answer_obj)
    results["total_qa"] += 1
    
    # Evaluate with retry
    print("   🔍 Critic evaluation (max 2 retries)...")
    for attempt in range(3):
        eval = critic.evaluate(qa, chunk)
        
        # SHOW DETAILED SCORES
        print(f"\n      📊 Attempt {attempt+1}:")
        print(f"         Decision: {eval.decision.value.upper()} | Score: {eval.overall_score:.3f}")
        print(f"         Criteria:")
        for crit_name, crit_eval in eval.criteria_evaluations.items():
            emoji = "✅" if crit_eval.result.value == "pass" else "❌"
            print(f"           {emoji} {crit_name}: {crit_eval.score:.2f}")
            if crit_eval.result.value != "pass":
                print(f"              → {crit_eval.explanation[:80]}...")
        
        if eval.decision.value == "pass":
            if attempt > 0:
                print(f"\n      ✅ FINAL: PASSED after {attempt} retries")
                results["total_retries"] += attempt
            else:
                print(f"\n      ✅ FINAL: PASSED")
            break
        else:
            if attempt < 2:
                print(f"\n      🔄 Regenerating with feedback...")
                feedback = critic.format_feedback_for_retry(eval)
                print(f"         💬 {feedback[:120]}...")
                new_q = q_gen.regenerate_with_feedback(chunk, qa.question, feedback)
                if new_q:
                    new_a = a_gen.regenerate_with_feedback(new_q, chunk, answer, feedback)
                    qa = QAPair.from_question_and_answer(new_q, new_a)
                    print(f"         📝 New Q: {qa.question[:70]}...")
                else:
                    break
            else:
                print(f"\n      ❌ FINAL: REJECTED after 2 retries")
                results["total_rejected"] += 1
    
    print()

print("=" * 80)
print("  RESULTS SUMMARY")
print("=" * 80)
print(f"Total QAs: {results['total_qa']}")
print(f"Rejected: {results['total_rejected']} ({results['total_rejected']/results['total_qa']*100:.1f}%)")
print(f"Total retries: {results['total_retries']}")
print(f"Avg retries/QA: {results['total_retries']/results['total_qa']:.2f}")
print()
print("✅ Test complete!")

with open("quick_test_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("💾 Results saved: quick_test_results.json")
