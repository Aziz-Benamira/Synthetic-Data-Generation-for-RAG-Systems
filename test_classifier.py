"""Quick test of QuestionTypeClassifier"""

import sys
sys.path.insert(0, 'src/agents')

from question_type_classifier import QuestionTypeClassifier

# Create classifier
classifier = QuestionTypeClassifier(language="french")

# Test cases
test_cases = [
    ("Qu'est-ce qu'une tribu?", "definition"),
    ("Quelle est la différence entre A et B?", "comparison"),
    ("Pourquoi X se produit-il?", "explanation"),
    ("Combien d'éléments?", "factoid"),
    ("Calculer la valeur de X.", "calculation"),
    ("Comment appliquer ce théorème?", "application"),
    ("Analyser les limites du modèle.", "analysis"),
]

print("=" * 70)
print("TESTING QUESTION TYPE CLASSIFIER")
print("=" * 70)
print()

correct = 0
for question, expected in test_cases:
    predicted = classifier.classify(question)
    status = "✓" if predicted == expected else "✗"
    print(f"{status} {question[:40]:40s} -> {predicted:12s} (expected: {expected})")
    if predicted == expected:
        correct += 1

print()
print(f"Accuracy: {correct}/{len(test_cases)} = {correct/len(test_cases)*100:.0f}%")
print()

# Test distribution
all_questions = [q for q, _ in test_cases]
print("Distribution Report:")
print(classifier.format_distribution_report(all_questions))
