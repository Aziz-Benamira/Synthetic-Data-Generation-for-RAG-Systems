"""
Show Critic Rejections from Test Results
=========================================

Displays rejected QA pairs with full critic feedback including:
- All failed criteria with scores
- Detailed explanations
- Rejection reasons
"""

import json

print("=" * 80)
print("🔍 CRITIC REJECTION EXAMPLES")
print("=" * 80)
print()

# Load test results
with open("test_borderline_results.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

# Filter rejected cases
rejected_cases = [d for d in data['details'] if d['decision'] == 'reject']

print(f"Found {len(rejected_cases)} REJECTED QA pairs from borderline tests")
print()

for i, case in enumerate(rejected_cases, 1):
    print("\n" + "=" * 80)
    print(f"REJECTION CASE #{i}: {case['label']}")
    print("=" * 80)
    print()
    
    print(f"Expected Issues: {', '.join(case['expected_issues'])}")
    print()
    
    print("❓ QUESTION:")
    print(f"   {case['question']}")
    print()
    
    print("💬 ANSWER:")
    print(f"   {case['answer']}")
    print()
    
    print("🔍 CRITIC EVALUATION:")
    print("─" * 80)
    print(f"   Decision: ❌ REJECT")
    print(f"   Overall Score: {case['overall_score']:.2f}/1.00 (threshold: 0.70)")
    print(f"   Criteria Passed: {case['criteria_passed']}/5")
    print()
    
    print("   Criterion Breakdown:")
    criteria = case['criteria_scores']
    
    criteria_info = [
        ('anchoring', 'ANCHORING', 'Answer grounded in chunk'),
        ('local_answerability', 'LOCAL ANSWERABILITY', 'Question answerable from chunk'),
        ('factual_accuracy', 'FACTUAL ACCURACY', 'No errors or hallucinations'),
        ('completeness', 'COMPLETENESS', 'Fully addresses question'),
        ('clarity', 'CLARITY', 'Clear and unambiguous')
    ]
    
    failed = []
    passed = []
    
    for key, name, desc in criteria_info:
        if key in criteria:
            score = criteria[key]
            if score < 0.7:
                emoji = "❌"
                failed.append(name)
            else:
                emoji = "✅"
                passed.append(name)
            
            print(f"   {emoji} {name}: {score:.2f} - {desc}")
    
    print()
    print(f"   ❌ FAILED CRITERIA: {', '.join(failed)}")
    print(f"   ✅ PASSED CRITERIA: {', '.join(passed)}")
    print()
    
    # Show specific rejection reasons
    print("   📝 WHY IT WAS REJECTED:")
    
    if criteria.get('anchoring', 1.0) < 0.7:
        score = criteria['anchoring']
        print(f"      • ANCHORING ({score:.2f}): Answer contains information not present")
        print(f"        in the source chunk or goes beyond what can be directly inferred")
    
    if criteria.get('local_answerability', 1.0) < 0.7:
        score = criteria['local_answerability']
        print(f"      • LOCAL ANSWERABILITY ({score:.2f}): Question requires external")
        print(f"        knowledge not contained in the chunk to be answered")
    
    if criteria.get('factual_accuracy', 1.0) < 0.7:
        score = criteria['factual_accuracy']
        print(f"      • FACTUAL ACCURACY ({score:.2f}): Answer contains errors,")
        print(f"        contradictions, or misinterpretations of the source")
    
    if criteria.get('completeness', 1.0) < 0.7:
        score = criteria['completeness']
        print(f"      • COMPLETENESS ({score:.2f}): Answer is too brief, omits")
        print(f"        important details, or doesn't fully address the question")
    
    if criteria.get('clarity', 1.0) < 0.7:
        score = criteria['clarity']
        print(f"      • CLARITY ({score:.2f}): Answer uses vague language,")
        print(f"        informal style, or confusing formulation")
    
    print()
    print("   🔄 FEEDBACK FOR REGENERATION:")
    print(f"      The answer scored {case['overall_score']:.2f} which is below the")
    print(f"      required threshold of 0.70. Focus on improving:")
    
    # Generate specific feedback
    worst_criterion = min(criteria.items(), key=lambda x: x[1])
    print(f"      • {worst_criterion[0].upper()} (lowest score: {worst_criterion[1]:.2f})")
    print()

print("\n" + "=" * 80)
print("📊 REJECTION SUMMARY")
print("=" * 80)
print()

# Calculate rejection statistics
all_cases = data['details']
total_rejected = len(rejected_cases)
total_tested = len(all_cases)

print(f"Total QA pairs tested: {total_tested}")
print(f"Rejected: {total_rejected} ({total_rejected/total_tested*100:.1f}%)")
print(f"Passed: {total_tested - total_rejected} ({(total_tested-total_rejected)/total_tested*100:.1f}%)")
print()

# Most common failure reasons
failure_counts = {}
for case in rejected_cases:
    for criterion, score in case['criteria_scores'].items():
        if score < 0.7:
            failure_counts[criterion] = failure_counts.get(criterion, 0) + 1

print("Most common failure criteria:")
for criterion, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
    print(f"  • {criterion.upper()}: {count} failures")

print()
print("✅ Analysis complete!")
