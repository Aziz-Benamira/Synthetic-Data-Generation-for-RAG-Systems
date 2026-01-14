"""
Show Critic Rejections with Full Feedback
==========================================

Creates intentionally bad QA pairs and shows the critic's detailed
rejection feedback including all failed criteria and explanations.
"""

import sys
sys.path.insert(0, 'src')

from dataclasses import dataclass
from openai import OpenAI
from agents.critic_agent import CriticAgent, FinalDecision
from agents.answer_generator import QAPair

@dataclass
class FakeChunk:
    """Mock chunk for testing"""
    chunk_id: str
    content: str
    chapter_title: str = "Test Chapter"
    section_title: str = "Test Section"
    page_range: tuple = (1, 1)
    semantic_type: str = "text"

# Sample chunk from probability theory
chunk = FakeChunk(
    chunk_id="1.1.c2",
    content="""
    Proposition 1.1.1 Une intersection de tribus est une tribu.
    Attention : ce n'est pas vrai pour la réunion : une réunion de tribus n'est pas une tribu.
    
    Soit F une tribu. Une sous-tribu de F est une tribu G telle que G ⊂ F, 
    soit A ∈ G implique A ∈ F.
    
    La tribu engendrée par une famille de parties est la plus petite tribu 
    contenant cette famille. Elle contient l'ensemble vide et Ω.
    """
)

print("=" * 80)
print("🔍 CRITIC REJECTION EXAMPLES WITH FULL FEEDBACK")
print("=" * 80)
print()

# Create critic
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
critic = CriticAgent(
    llm_client=client,
    model_name="llama3:8b",
    language="fr",
    temperature=0.2,
    strict_mode=True
)

print("✅ Critic initialized (llama3:8b in strict mode)")
print()

# Test cases with intentionally bad QA pairs
test_cases = [
    {
        "name": "HALLUCINATION - Numbers not in chunk",
        "question": "Quand a été définie la notion de tribu ?",
        "answer": "La notion de tribu a été définie par Émile Borel en 1895 et formalisée par Felix Hausdorff en 1914.",
    },
    {
        "name": "FACTUAL ERROR - Contradicts chunk",
        "question": "Qu'est-ce qu'une intersection de tribus ?",
        "answer": "Une intersection de tribus n'est PAS une tribu. C'est la réunion de tribus qui est toujours une tribu.",
    },
    {
        "name": "WHY question without explanation",
        "question": "Pourquoi une intersection de tribus est-elle une tribu ?",
        "answer": "Une intersection de tribus est une tribu.",
    },
    {
        "name": "TOO SHORT - Incomplete answer",
        "question": "Expliquez la construction de la tribu engendrée par une famille de parties.",
        "answer": "C'est la plus petite tribu.",
    },
    {
        "name": "INFORMAL LANGUAGE",
        "question": "Qu'est-ce qu'une sous-tribu ?",
        "answer": "Une sous-tribu c'est genre une tribu G qui est un truc inclus dans une autre tribu F.",
    },
]

print(f"Testing {len(test_cases)} intentionally bad QA pairs...")
print()

for i, test in enumerate(test_cases, 1):
    print("\n" + "=" * 80)
    print(f"TEST CASE #{i}: {test['name']}")
    print("=" * 80)
    print()
    
    print("📄 CHUNK CONTENT:")
    print(f"   {chunk.content[:150]}...")
    print()
    
    print("❓ QUESTION:")
    print(f"   {test['question']}")
    print()
    
    print("💬 ANSWER:")
    print(f"   {test['answer']}")
    print()
    
    # Create QA pair
    qa_pair = QAPair(
        question=test['question'],
        answer=test['answer'],
        question_type="conceptual",
        difficulty="medium",
        supporting_quotes=[],
        confidence_score=0.8
    )
    
    # Evaluate
    print("🔍 CRITIC EVALUATION:")
    print("─" * 80)
    
    evaluation = critic.evaluate(qa_pair, chunk)
    
    print(f"Decision: {evaluation.decision.value.upper()}")
    print(f"Overall Score: {evaluation.overall_score:.2f}/1.00")
    print()
    
    if evaluation.decision == FinalDecision.REJECT:
        print("❌ REJECTED - Detailed Feedback:")
        print()
        
        # Show failed criteria
        for criterion_name in evaluation.failed_criteria:
            criterion_eval = evaluation.criteria_evaluations[criterion_name]
            print(f"   ❌ {criterion_name.upper()}: {criterion_eval.score:.2f}")
            print(f"      Reason: {criterion_eval.explanation}")
            if criterion_eval.evidence:
                print(f"      Evidence: {criterion_eval.evidence}")
            print()
        
        # Show all criteria details
        print("   Full Criterion Breakdown:")
        for criterion_name, criterion_eval in evaluation.criteria_evaluations.items():
            emoji = "✅" if criterion_eval.result.value == "pass" else "❌"
            print(f"   {emoji} {criterion_name}: {criterion_eval.score:.2f}")
            print(f"      {criterion_eval.explanation}")
            print()
        
        # Show feedback for retry
        print("   🔄 Feedback for regeneration:")
        feedback = critic.format_feedback_for_retry(evaluation)
        print(f"   {feedback}")
        
    else:
        print("✅ PASSED (unexpected - should have been rejected!)")
        for criterion_name, criterion_eval in evaluation.criteria_evaluations.items():
            print(f"   {criterion_name}: {criterion_eval.score:.2f}")
    
    print()
    input("Press Enter for next test case...")

print("\n" + "=" * 80)
print("✅ All test cases completed!")
print("=" * 80)
