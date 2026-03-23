"""
Critic Evaluation Analyzer
===========================

Analyzes the critic evaluations from a pipeline run to show:
- Each QA pair with all criterion scores
- Pass/Reject decisions
- Rejection patterns
"""

import json
import sys
from pathlib import Path

# Load the dataset
dataset_path = "output/quick_test/dataset.json"

print("=" * 80)
print("🔍 CRITIC EVALUATION ANALYSIS")
print("=" * 80)
print()

with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

metadata = data['metadata']
entries = data['data']

# Show summary stats
print("📊 PIPELINE STATISTICS")
print("=" * 80)
stats = metadata['stats']
print(f"Chunks processed: {stats['processed_chunks']}/{stats['total_chunks']}")
print(f"Questions generated: {stats['total_questions_generated']}")
print(f"QA pairs passed: {stats['passed_qa_pairs']}")
print(f"QA pairs rejected (final): {stats['rejected_qa_pairs']}")
print(f"Retry attempts: {stats['total_retries']}")
print(f"Pass rate: {stats['pass_rate']*100:.1f}%")
print()

if stats['total_retries'] > 0:
    initial_rejection_rate = (stats['total_retries'] / stats['total_questions_generated']) * 100
    print(f"🎯 Initial rejection rate: {initial_rejection_rate:.1f}%")
    if 30 <= initial_rejection_rate <= 50:
        print(f"   ✅ Within target range (30-50%)!")
    elif initial_rejection_rate < 30:
        print(f"   ⚠️  Below target (critic too lenient)")
    else:
        print(f"   ⚠️  Above target (critic too strict)")
    print()

# Show detailed evaluation for each QA pair
print()
print("=" * 80)
print("🔍 DETAILED CRITIC EVALUATIONS")
print("=" * 80)
print()

for idx, entry in enumerate(entries, 1):
    print(f"\n{'═' * 80}")
    print(f"QA PAIR #{idx}")
    print(f"{'═' * 80}")
    print()
    
    # Basic info
    print(f"📄 Source: {entry['chunk_id']} | {entry['section']}")
    print(f"📊 Type: {entry['question_type']} | Difficulty: {entry['difficulty']}")
    print()
    
    # Question
    print(f"❓ QUESTION:")
    print(f"   {entry['question']}")
    print()
    
    # Answer (truncated)
    print(f"💬 ANSWER:")
    answer_preview = entry['answer'][:300] + "..." if len(entry['answer']) > 300 else entry['answer']
    print(f"   {answer_preview}")
    print()
    
    # Critic evaluation
    print(f"🔍 CRITIC EVALUATION:")
    print(f"   Overall Score: {entry['critic_score']:.2f}/1.00")
    print()
    
    # Show each criterion
    print("   Criterion Breakdown:")
    criterion_scores = entry['criterion_scores']
    
    criteria_order = ['anchoring', 'local_answerability', 'factual_accuracy', 'completeness', 'clarity']
    criterion_names = {
        'anchoring': 'ANCHORING (Answer from chunk)',
        'local_answerability': 'LOCAL ANSWERABILITY (Question answerable)',
        'factual_accuracy': 'FACTUAL ACCURACY (No errors)',
        'completeness': 'COMPLETENESS (Addresses question)',
        'clarity': 'CLARITY (Clear & unambiguous)'
    }
    
    for criterion in criteria_order:
        if criterion in criterion_scores:
            score = criterion_scores[criterion]
            emoji = "✅" if score >= 0.7 else "⚠️" if score >= 0.5 else "❌"
            name = criterion_names.get(criterion, criterion.upper())
            print(f"   {emoji} {name}: {score:.2f}")
    
    print()
    
    # Decision
    if entry['critic_score'] >= 0.7:
        print(f"   ✅ DECISION: PASS")
    else:
        print(f"   ❌ DECISION: REJECT (or borderline)")
    
    # Show chunk context
    if 'supporting_quotes' in entry and entry['supporting_quotes']:
        print()
        print(f"   📝 Supporting quotes from chunk:")
        for quote in entry['supporting_quotes'][:2]:
            quote_preview = quote[:150] + "..." if len(quote) > 150 else quote
            print(f"      • {quote_preview}")

print()
print()
print("=" * 80)
print("📈 CRITERION PERFORMANCE")
print("=" * 80)
print()

# Calculate average scores per criterion
criterion_totals = {}
criterion_counts = {}

for entry in entries:
    for criterion, score in entry['criterion_scores'].items():
        criterion_totals[criterion] = criterion_totals.get(criterion, 0) + score
        criterion_counts[criterion] = criterion_counts.get(criterion, 0) + 1

print("Average scores across all QA pairs:")
for criterion in ['anchoring', 'local_answerability', 'factual_accuracy', 'completeness', 'clarity']:
    if criterion in criterion_totals:
        avg_score = criterion_totals[criterion] / criterion_counts[criterion]
        emoji = "✅" if avg_score >= 0.8 else "⚠️" if avg_score >= 0.6 else "❌"
        print(f"  {emoji} {criterion.upper()}: {avg_score:.2f}")

print()
print("=" * 80)
print("✅ Analysis complete!")
print()
print(f"💡 Note: The critic evaluated {len(entries)} QA pairs that PASSED.")
print(f"   Rejected pairs (after retries) are not in the dataset.")
print(f"   Initial rejections: {stats['total_retries']} (triggered retry loop)")
print()
