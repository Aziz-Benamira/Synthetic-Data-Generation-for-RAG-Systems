"""
Demo: Advanced Validation Pipeline
===================================

Demonstrates the 3 new advanced agents working together:
1. AnswerQualityScorer - Catches hallucinations
2. ChainOfThoughtValidator - Verifies reasoning
3. Active Learning UI - Human review

This script shows a complete validation cascade for generated QA pairs.
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'agents'))

from answer_quality_scorer import AnswerQualityScorer, AnswerQualityScore
from chain_of_thought_validator import ChainOfThoughtValidator, ChainOfThoughtValidation


def demo_validation_cascade():
    """Demonstrate validation cascade with example QA pairs."""
    print("=" * 80)
    print("ADVANCED VALIDATION PIPELINE - DEMO")
    print("=" * 80)
    print()
    
    # Initialize validators
    print("🔧 Initializing validators...")
    quality_scorer = AnswerQualityScorer()
    reasoning_validator = ChainOfThoughtValidator()
    print("✅ Validators ready\n")
    
    # Test data
    chunk_text = """
    1.1 Définition d'une Tribu
    
    Une tribu (ou σ-algèbre) est une collection de sous-ensembles d'un ensemble Ω
    qui satisfait les propriétés suivantes:
    1. Ω appartient à la tribu
    2. Si A appartient à la tribu, alors son complémentaire aussi
    3. La tribu est stable par union dénombrable
    
    Les tribus sont fondamentales en théorie de la mesure et en probabilités car
    elles permettent de définir rigoureusement les événements mesurables.
    """
    
    test_cases = [
        {
            "name": "✅ GOOD: Well-grounded with reasoning",
            "question": "Pourquoi les tribus sont-elles importantes?",
            "answer": "Les tribus sont importantes car elles permettent de définir "
                     "rigoureusement les événements mesurables. En effet, elles fournissent "
                     "la structure mathématique nécessaire pour la théorie de la mesure. "
                     "Par conséquent, elles sont fondamentales en probabilités.",
            "expected_quality": "high",
            "expected_reasoning": "valid"
        },
        {
            "name": "❌ BAD: Hallucination detected",
            "question": "Qu'est-ce qu'une tribu?",
            "answer": "Une tribu est une structure inventée par Borel en 1895 pour "
                     "formaliser la théorie des ensembles ordonnés. Elle utilise les "
                     "axiomes de Zermelo-Fraenkel.",
            "expected_quality": "low",
            "expected_reasoning": "n/a"
        },
        {
            "name": "⚠️  WARNING: Poor reasoning",
            "question": "Pourquoi utilise-t-on des tribus?",
            "answer": "On utilise des tribus. Elles sont utiles. Les mathématiques "
                     "les utilisent beaucoup.",
            "expected_quality": "medium",
            "expected_reasoning": "invalid"
        },
        {
            "name": "✅ GOOD: Sequential reasoning",
            "question": "Comment construire une tribu?",
            "answer": "D'abord, on prend un ensemble Ω. Ensuite, on choisit des "
                     "sous-ensembles qui contiennent Ω. Puis, on vérifie la stabilité "
                     "par complémentaire. Enfin, on vérifie la stabilité par union "
                     "dénombrable.",
            "expected_quality": "high",
            "expected_reasoning": "valid"
        }
    ]
    
    # Process each test case
    for i, test in enumerate(test_cases, 1):
        print("─" * 80)
        print(f"\nTest Case {i}: {test['name']}")
        print("─" * 80)
        print(f"Question: {test['question']}")
        print(f"Answer: {test['answer'][:80]}...")
        print()
        
        # Step 1: Answer Quality Scoring
        print("1️⃣ Answer Quality Scorer")
        print("   " + "-" * 70)
        
        quality = quality_scorer.score_answer(
            question=test['question'],
            answer=test['answer'],
            chunk_content=chunk_text
        )
        
        # Determine quality status
        if quality.overall_score >= 0.7 and quality.is_grounded:
            quality_status = "✅ PASS"
        elif quality.overall_score >= 0.5:
            quality_status = "⚠️  WARNING"
        else:
            quality_status = "❌ FAIL"
        
        print(f"   {quality_status} | Score: {quality.overall_score:.2f} | "
              f"Grounded: {quality.is_grounded}")
        print(f"   Components: entity={quality.entity_overlap_score:.2f}, "
              f"keyword={quality.keyword_overlap_score:.2f}, "
              f"length={quality.length_score:.2f}")
        
        if quality.issues:
            print(f"   🚨 Issues:")
            for issue in quality.issues:
                print(f"      • {issue}")
        
        if quality.missing_entities and len(quality.missing_entities) > 0:
            print(f"   🔍 Potential hallucinations: {', '.join(quality.missing_entities[:3])}")
        
        print()
        
        # Step 2: Chain-of-Thought Validation (for reasoning questions)
        if 'pourquoi' in test['question'].lower() or 'comment' in test['question'].lower():
            print("2️⃣ Chain-of-Thought Validator")
            print("   " + "-" * 70)
            
            reasoning = reasoning_validator.validate(
                question=test['question'],
                answer=test['answer']
            )
            
            # Determine reasoning status
            if reasoning.is_valid:
                reasoning_status = "✅ VALID"
            elif reasoning.overall_score >= 0.5:
                reasoning_status = "⚠️  WARNING"
            else:
                reasoning_status = "❌ INVALID"
            
            print(f"   {reasoning_status} | Score: {reasoning.overall_score:.2f}")
            print(f"   Type: {reasoning.reasoning_type.value} | Steps: {reasoning.num_steps}")
            print(f"   Causality: {reasoning.has_causality}, "
                  f"Flow: {reasoning.has_logical_flow}, "
                  f"Circular: {reasoning.has_circular_reasoning}")
            
            if reasoning.issues:
                print(f"   🚨 Issues:")
                for issue in reasoning.issues:
                    print(f"      • {issue}")
            
            print()
        
        # Step 3: Final Decision
        print("3️⃣ Final Decision")
        print("   " + "-" * 70)
        
        # Determine if should accept
        passes_quality = quality.overall_score >= 0.7 and quality.is_grounded
        
        if 'pourquoi' in test['question'].lower() or 'comment' in test['question'].lower():
            reasoning = reasoning_validator.validate(test['question'], test['answer'])
            passes_reasoning = reasoning.is_valid or reasoning.overall_score >= 0.6
        else:
            passes_reasoning = True  # Not a reasoning question
        
        if passes_quality and passes_reasoning:
            decision = "✅ ACCEPT - Add to dataset"
            action = "Pipeline continues to CriticAgent"
        elif passes_quality:
            decision = "⚠️  CONDITIONAL ACCEPT - Reasoning issues but factually correct"
            action = "Add to dataset with warning flag"
        else:
            decision = "❌ REJECT - Quality issues detected"
            action = "Regenerate answer or mark for human review"
        
        print(f"   {decision}")
        print(f"   Action: {action}")
        print()
    
    # Summary
    print("=" * 80)
    print("VALIDATION CASCADE COMPLETE")
    print("=" * 80)
    print()
    print("📊 Summary:")
    print("   • AnswerQualityScorer: Catches hallucinations and poor grounding")
    print("   • ChainOfThoughtValidator: Verifies reasoning quality")
    print("   • Combined: Multi-layer quality assurance")
    print()
    print("🎯 Next Steps:")
    print("   1. Integrate into pipeline (pipeline.py)")
    print("   2. Generate full dataset with validation")
    print("   3. Launch Active Learning UI for human review:")
    print("      python src/utils/active_learning_ui.py output/dataset.json")
    print()
    print("=" * 80)


if __name__ == "__main__":
    demo_validation_cascade()
