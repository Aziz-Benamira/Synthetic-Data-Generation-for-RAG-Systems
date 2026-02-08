"""
Test script to verify Seif's new validators work correctly.
"""

import sys
sys.path.insert(0, 'seif_changes_review')

from answer_quality_scorer import AnswerQualityScorer, AnswerQualityScore
from chain_of_thought_validator import ChainOfThoughtValidator, ChainOfThoughtValidation

# Test 1: AnswerQualityScorer with hallucination
print("=" * 80)
print("TEST 1: AnswerQualityScorer - Detecting Hallucinations")
print("=" * 80)

scorer = AnswerQualityScorer()

# Chunk content (real)
chunk = """
Une tribu (ou σ-algèbre) sur un ensemble Ω est une collection de sous-ensembles 
de Ω qui vérifie trois propriétés:
1. Ω appartient à la tribu
2. Si A est dans la tribu, alors son complémentaire l'est aussi
3. La réunion dénombrable d'éléments de la tribu est aussi dans la tribu
"""

# Answer WITH hallucination (adds numbers not in chunk)
question1 = "Qu'est-ce qu'une tribu?"
answer_hallucinated = """
Une tribu est une collection de sous-ensembles qui vérifie trois propriétés.
Par exemple, on peut montrer que toute tribu contient au moins 2^8 = 256 éléments.
"""

score1 = scorer.score_answer(question1, answer_hallucinated, chunk)
print(f"\n📊 Question: {question1}")
print(f"📝 Answer: {answer_hallucinated[:100]}...")
print(f"🎯 Overall Score: {score1.overall_score:.2f}")
print(f"✅ Grounded: {score1.is_grounded}")
print(f"⚠️  Issues: {score1.issues}")

# Answer WITHOUT hallucination (clean)
answer_clean = """
Une tribu est une collection de sous-ensembles de Ω qui vérifie trois propriétés:
Ω appartient à la tribu, si A est dans la tribu alors son complémentaire aussi,
et la réunion dénombrable d'éléments de la tribu est aussi dans la tribu.
"""

score2 = scorer.score_answer(question1, answer_clean, chunk)
print(f"\n📊 Question: {question1}")
print(f"📝 Answer: {answer_clean[:100]}...")
print(f"🎯 Overall Score: {score2.overall_score:.2f}")
print(f"✅ Grounded: {score2.is_grounded}")
print(f"⚠️  Issues: {score2.issues}")

# Test 2: ChainOfThoughtValidator
print("\n" + "=" * 80)
print("TEST 2: ChainOfThoughtValidator - Checking Logical Reasoning")
print("=" * 80)

validator = ChainOfThoughtValidator()

# Why question with GOOD reasoning (has causal markers)
question2 = "Pourquoi les tribus sont importantes en théorie des probabilités?"
answer_good_reasoning = """
Les tribus sont importantes car elles permettent de définir les événements mesurables.
En effet, pour qu'une probabilité soit bien définie, on doit pouvoir mesurer les ensembles.
Ainsi, la structure de tribu garantit que les opérations d'union, intersection et complément
restent dans l'espace des événements mesurables. Par conséquent, on peut calculer
des probabilités pour des combinaisons complexes d'événements.
"""

result1 = validator.validate(question2, answer_good_reasoning)
print(f"\n📊 Question: {question2}")
print(f"📝 Answer: {answer_good_reasoning[:100]}...")
print(f"✅ Valid: {result1.is_valid}")
print(f"🎯 Overall Score: {result1.overall_score:.2f}")
print(f"🔗 Reasoning Type: {result1.reasoning_type.value}")
print(f"📋 Steps: {result1.num_steps}")
print(f"🔄 Has Causality: {result1.has_causality}")
print(f"⚠️  Issues: {result1.issues}")

# Why question with BAD reasoning (no causal links)
answer_bad_reasoning = """
Les tribus sont des collections de sous-ensembles. Elles ont trois propriétés.
C'est un concept important.
"""

result2 = validator.validate(question2, answer_bad_reasoning)
print(f"\n📊 Question: {question2}")
print(f"📝 Answer: {answer_bad_reasoning[:100]}...")
print(f"✅ Valid: {result2.is_valid}")
print(f"🎯 Overall Score: {result2.overall_score:.2f}")
print(f"🔗 Reasoning Type: {result2.reasoning_type.value}")
print(f"📋 Steps: {result2.num_steps}")
print(f"🔄 Has Causality: {result2.has_causality}")
print(f"⚠️  Issues: {result2.issues}")

print("\n" + "=" * 80)
print("✅ ALL TESTS COMPLETED")
print("=" * 80)
print("\n📝 Summary:")
print(f"  - AnswerQualityScorer: {'✅ Working' if score1.overall_score < score2.overall_score else '❌ Not discriminating'}")
print(f"  - ChainOfThoughtValidator: {'✅ Working' if result1.is_valid and not result2.is_valid else '❌ Not discriminating'}")
print("\n💡 Next step: Integrate these validators into pipeline.py")
