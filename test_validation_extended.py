"""
Extended Validation Test - 20+ Chunks with Detailed Logging
===========================================================
Tests the improved critic agent with comprehensive statistics and rejection analysis.
"""

import sys
import os
import json
from datetime import datetime
from collections import defaultdict

# Add source directories to path
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/agents')
sys.path.insert(0, 'src/parsers')
sys.path.insert(0, 'src/chunking')
sys.path.insert(0, 'src/orchestrator')
sys.path.insert(0, 'src/utils')

from openai import OpenAI
from orchestrator.pipeline import DatasetPipeline, PipelineConfig

print("=" * 80)
print("EXTENDED VALIDATION TEST - CRITIC CALIBRATION")
print("=" * 80)
print()

# Configuration
NUM_CHUNKS = 25  # 25 chunks for statistical significance
QUESTIONS_PER_CHUNK = 3
MAX_RETRIES = 2
THRESHOLD = 0.90  # Strict threshold

print(f"📋 CONFIGURATION")
print("-" * 80)
print(f"PDF: data/pdfs/M2_cours.pdf")
print(f"Chunks: {NUM_CHUNKS}")
print(f"Questions/chunk: {QUESTIONS_PER_CHUNK}")
print(f"Max retries: {MAX_RETRIES}")
print(f"Threshold: {THRESHOLD} (strict mode)")
print(f"Expected QA pairs: {NUM_CHUNKS * QUESTIONS_PER_CHUNK} = {NUM_CHUNKS * QUESTIONS_PER_CHUNK}")
print()

# Detailed logging structures
rejection_log = []
statistics = {
    "total_questions": 0,
    "first_attempt_pass": 0,
    "first_attempt_reject": 0,
    "retry_1_pass": 0,
    "retry_1_reject": 0,
    "retry_2_pass": 0,
    "retry_2_reject": 0,
    "final_accept": 0,
    "final_reject": 0,
    "criteria_failures": defaultdict(int),
    "hard_rule_triggers": defaultdict(int)
}

# Create Ollama client
print("🔌 Connexion à Ollama local...")
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)
print("✅ Client créé!")
print()

# Create pipeline with detailed logging callback
def detailed_logging_callback(event_type, data):
    """Enhanced callback to track rejections and retries"""
    global rejection_log, statistics
    
    if event_type == "qa_evaluation_start":
        statistics["total_questions"] += 1
        
    elif event_type == "qa_evaluation_result":
        qa_pair = data.get("qa_pair")
        result = data.get("result")
        retry_count = data.get("retry_count", 0)
        
        # Track attempt outcome
        if retry_count == 0:
            if result.decision.is_valid:
                statistics["first_attempt_pass"] += 1
            else:
                statistics["first_attempt_reject"] += 1
        elif retry_count == 1:
            if result.decision.is_valid:
                statistics["retry_1_pass"] += 1
            else:
                statistics["retry_1_reject"] += 1
        elif retry_count == 2:
            if result.decision.is_valid:
                statistics["retry_2_pass"] += 1
            else:
                statistics["retry_2_reject"] += 1
        
        # Log rejection details
        if not result.decision.is_valid:
            rejection_entry = {
                "timestamp": datetime.now().isoformat(),
                "attempt": retry_count + 1,
                "question": qa_pair.question[:100],
                "answer": qa_pair.answer[:100],
                "overall_score": result.decision.overall_score,
                "failed_criteria": [],
                "hard_rules_triggered": []
            }
            
            # Track which criteria failed
            for criterion_name, evaluation in result.criteria_evaluations.items():
                if evaluation.result.value in ["FAIL", "WARNING"]:
                    statistics["criteria_failures"][criterion_name] += 1
                    rejection_entry["failed_criteria"].append({
                        "name": criterion_name,
                        "result": evaluation.result.value,
                        "score": evaluation.score,
                        "reason": evaluation.explanation[:150]
                    })
                    
                    # Detect hard rule triggers (score = 0.0, 0.1, 0.2, 0.3)
                    if evaluation.score in [0.0, 0.1, 0.2, 0.3]:
                        if "HARD RULE" in evaluation.explanation.upper():
                            rule_name = f"{criterion_name}_hard_rule"
                            statistics["hard_rule_triggers"][rule_name] += 1
                            rejection_entry["hard_rules_triggered"].append(rule_name)
            
            rejection_log.append(rejection_entry)
    
    elif event_type == "qa_final_decision":
        is_accepted = data.get("accepted", False)
        if is_accepted:
            statistics["final_accept"] += 1
        else:
            statistics["final_reject"] += 1

print("🚀 Démarrage du pipeline avec logging détaillé...")
print()

# Create pipeline configuration
config = PipelineConfig(
    pdf_path="data/pdfs/M2_cours.pdf",
    output_dir="output/validation",
    generator_model="mistral:latest",
    critic_model="llama3:8b",
    language="fr",
    questions_per_chunk=QUESTIONS_PER_CHUNK,
    max_retries=MAX_RETRIES,
    max_chunks=NUM_CHUNKS,
    temperature=0.7
)

pipeline = DatasetPipeline(
    config=config,
    llm_client=client,
    progress_callback=None
)

# Run pipeline
try:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing {NUM_CHUNKS} chunks...")
    print("=" * 80)
    
    dataset = pipeline.run()
    
    print()
    print("=" * 80)
    print("✅ PIPELINE COMPLETED!")
    print("=" * 80)
    print()
    
except KeyboardInterrupt:
    print()
    print("=" * 80)
    print("⚠️  PIPELINE INTERRUPTED BY USER")
    print("=" * 80)
    print()
    print("Generating statistics from partial results...")
    print()

except Exception as e:
    print()
    print("=" * 80)
    print(f"❌ PIPELINE ERROR: {e}")
    print("=" * 80)
    print()
    print("Generating statistics from partial results...")
    print()

# Generate comprehensive statistics report
print("=" * 80)
print("📊 DETAILED VALIDATION STATISTICS")
print("=" * 80)
print()

# Overall metrics
print("1️⃣ OVERALL METRICS")
print("-" * 80)
total_processed = statistics["total_questions"]
first_pass_rate = (statistics["first_attempt_pass"] / total_processed * 100) if total_processed > 0 else 0
first_reject_rate = (statistics["first_attempt_reject"] / total_processed * 100) if total_processed > 0 else 0
final_accept_rate = (statistics["final_accept"] / total_processed * 100) if total_processed > 0 else 0

print(f"Total questions processed: {total_processed}")
print(f"Final accepted: {statistics['final_accept']} ({final_accept_rate:.1f}%)")
print(f"Final rejected: {statistics['final_reject']}")
print()
print(f"First attempt pass: {statistics['first_attempt_pass']} ({first_pass_rate:.1f}%)")
print(f"First attempt reject: {statistics['first_attempt_reject']} ({first_reject_rate:.1f}%)")
print()
print(f"Retry 1 pass: {statistics['retry_1_pass']}")
print(f"Retry 1 reject: {statistics['retry_1_reject']}")
print(f"Retry 2 pass: {statistics['retry_2_pass']}")
print(f"Retry 2 reject: {statistics['retry_2_reject']}")
print()

# Retry effectiveness
if statistics['first_attempt_reject'] > 0:
    retry_recovery_rate = (statistics['retry_1_pass'] + statistics['retry_2_pass']) / statistics['first_attempt_reject'] * 100
    print(f"📈 Retry recovery rate: {retry_recovery_rate:.1f}% (rejected → accepted after retry)")
print()

# Criteria failure analysis
print("2️⃣ CRITERIA FAILURE BREAKDOWN")
print("-" * 80)
if statistics["criteria_failures"]:
    total_failures = sum(statistics["criteria_failures"].values())
    print(f"Total criterion failures: {total_failures}")
    print()
    for criterion, count in sorted(statistics["criteria_failures"].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_failures * 100) if total_failures > 0 else 0
        print(f"  • {criterion:25s}: {count:3d} failures ({percentage:5.1f}%)")
    print()
else:
    print("  No criterion failures recorded")
    print()

# Hard rules analysis
print("3️⃣ HARD RULES TRIGGER ANALYSIS")
print("-" * 80)
if statistics["hard_rule_triggers"]:
    total_triggers = sum(statistics["hard_rule_triggers"].values())
    print(f"Total hard rule triggers: {total_triggers}")
    print()
    for rule, count in sorted(statistics["hard_rule_triggers"].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_processed * 100) if total_processed > 0 else 0
        print(f"  • {rule:35s}: {count:3d} triggers ({percentage:5.1f}%)")
    print()
else:
    print("  No hard rules triggered (LLM judgments only)")
    print()

# Rejection rate comparison
print("4️⃣ REJECTION RATE ANALYSIS")
print("-" * 80)
print(f"Target rejection rate: 30-50%")
print(f"Actual first-attempt rejection rate: {first_reject_rate:.1f}%")
print()
if 30 <= first_reject_rate <= 50:
    print("✅ STATUS: Within target range!")
elif first_reject_rate < 30:
    print("⚠️  STATUS: Below target (critic may still be too lenient)")
else:
    print("⚠️  STATUS: Above target (critic may be too strict)")
print()

# Sample rejections
print("5️⃣ SAMPLE REJECTIONS (First 5)")
print("-" * 80)
for i, rejection in enumerate(rejection_log[:5], 1):
    print(f"Rejection #{i}:")
    print(f"  Attempt: {rejection['attempt']}")
    print(f"  Question: {rejection['question']}...")
    print(f"  Overall score: {rejection['overall_score']:.2f}")
    print(f"  Failed criteria:")
    for criterion in rejection['failed_criteria']:
        print(f"    - {criterion['name']}: {criterion['result']} (score: {criterion['score']:.2f})")
        print(f"      Reason: {criterion['reason']}...")
    if rejection['hard_rules_triggered']:
        print(f"  Hard rules: {', '.join(rejection['hard_rules_triggered'])}")
    print()

# Save detailed logs to files
print("=" * 80)
print("💾 SAVING DETAILED LOGS")
print("=" * 80)

# Save rejection log
rejection_log_path = "validation_rejection_log.json"
with open(rejection_log_path, 'w', encoding='utf-8') as f:
    json.dump(rejection_log, f, indent=2, ensure_ascii=False)
print(f"✅ Rejection log saved: {rejection_log_path}")

# Save statistics
stats_path = "validation_statistics.json"
stats_output = {
    "test_config": {
        "date": datetime.now().isoformat(),
        "num_chunks": NUM_CHUNKS,
        "questions_per_chunk": QUESTIONS_PER_CHUNK,
        "max_retries": MAX_RETRIES,
        "threshold": THRESHOLD
    },
    "metrics": statistics,
    "total_rejections": len(rejection_log)
}
with open(stats_path, 'w', encoding='utf-8') as f:
    json.dump(stats_output, f, indent=2, ensure_ascii=False)
print(f"✅ Statistics saved: {stats_path}")

# Generate markdown report
report_path = "VALIDATION_REPORT.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(f"# Critic Validation Report\n\n")
    f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"## Test Configuration\n\n")
    f.write(f"- Chunks processed: {NUM_CHUNKS}\n")
    f.write(f"- Questions per chunk: {QUESTIONS_PER_CHUNK}\n")
    f.write(f"- Max retries: {MAX_RETRIES}\n")
    f.write(f"- Quality threshold: {THRESHOLD}\n\n")
    
    f.write(f"## Results Summary\n\n")
    f.write(f"- **Total questions:** {total_processed}\n")
    f.write(f"- **First-attempt rejection rate:** {first_reject_rate:.1f}%\n")
    f.write(f"- **Final acceptance rate:** {final_accept_rate:.1f}%\n")
    f.write(f"- **Retry recovery rate:** {retry_recovery_rate:.1f}%\n\n")
    
    f.write(f"## Target Achievement\n\n")
    if 30 <= first_reject_rate <= 50:
        f.write(f"✅ **SUCCESS** - Rejection rate {first_reject_rate:.1f}% is within target range (30-50%)\n\n")
    else:
        f.write(f"⚠️  **NEEDS ADJUSTMENT** - Rejection rate {first_reject_rate:.1f}% is outside target range (30-50%)\n\n")
    
    f.write(f"## Most Common Failure Criteria\n\n")
    if statistics["criteria_failures"]:
        for criterion, count in sorted(statistics["criteria_failures"].items(), key=lambda x: x[1], reverse=True)[:5]:
            percentage = (count / sum(statistics["criteria_failures"].values()) * 100)
            f.write(f"- **{criterion}**: {count} failures ({percentage:.1f}%)\n")
    f.write(f"\n")
    
    f.write(f"## Hard Rules Impact\n\n")
    if statistics["hard_rule_triggers"]:
        total_triggers = sum(statistics["hard_rule_triggers"].values())
        trigger_rate = (total_triggers / statistics["first_attempt_reject"] * 100) if statistics["first_attempt_reject"] > 0 else 0
        f.write(f"- **Total triggers:** {total_triggers}\n")
        f.write(f"- **Percentage of rejections due to hard rules:** {trigger_rate:.1f}%\n\n")
        for rule, count in sorted(statistics["hard_rule_triggers"].items(), key=lambda x: x[1], reverse=True):
            f.write(f"  - {rule}: {count}\n")
    else:
        f.write(f"No hard rules triggered during this test.\n")
    f.write(f"\n")

print(f"✅ Markdown report saved: {report_path}")
print()

print("=" * 80)
print("✅ VALIDATION COMPLETE!")
print("=" * 80)
print()
print("📁 Generated files:")
print(f"  1. {rejection_log_path} - Detailed rejection logs")
print(f"  2. {stats_path} - Statistics in JSON format")
print(f"  3. {report_path} - Human-readable report")
