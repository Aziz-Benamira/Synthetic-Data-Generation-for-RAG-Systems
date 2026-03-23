"""
Simplified Extended Validation Test - 20+ Chunks
================================================
Tests the improved critic agent with comprehensive statistics.
Analyzes the generated dataset for rejection patterns.
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
NUM_CHUNKS = 20  # 20 chunks for faster test
QUESTIONS_PER_CHUNK = 3
MAX_RETRIES = 2

print(f"📋 CONFIGURATION")
print("-" * 80)
print(f"PDF: data/pdfs/M2_cours.pdf")
print(f"Chunks: {NUM_CHUNKS}")
print(f"Questions/chunk: {QUESTIONS_PER_CHUNK}")
print(f"Max retries: {MAX_RETRIES}")
print(f"Expected QA pairs: {NUM_CHUNKS * QUESTIONS_PER_CHUNK} = {NUM_CHUNKS * QUESTIONS_PER_CHUNK}")
print()

# Create Ollama client
print("🔌 Connexion à Ollama local...")
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)
print("✅ Client créé!")
print()

# Create pipeline configuration
print("⚙️  Configuration du pipeline...")
config = PipelineConfig(
    pdf_path="data/pdfs/M2_cours.pdf",
    output_dir="output/validation",
    generator_model="mistral:latest",
    critic_model="llama3:8b",
    language="fr",
    questions_per_chunk=QUESTIONS_PER_CHUNK,
    max_retries=MAX_RETRIES,
    max_chunks=NUM_CHUNKS,
    temperature=0.7,
    save_checkpoints=True,
    checkpoint_frequency=5
)

pipeline = DatasetPipeline(
    config=config,
    llm_client=client,
    progress_callback=None
)
print("✅ Pipeline configuré!")
print()

# Run pipeline
print("🚀 Démarrage du pipeline...")
print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing {NUM_CHUNKS} chunks...")
print("=" * 80)
print()

try:
    dataset = pipeline.run()
    
    print()
    print("=" * 80)
    print("✅ PIPELINE COMPLETED!")
    print("=" * 80)
    print()
    
    # Export dataset
    output_file = "output/validation/dataset.json"
    pipeline.export(output_file)
    print(f"💾 Dataset exported to: {output_file}")
    print()
    
except KeyboardInterrupt:
    print()
    print("=" * 80)
    print("⚠️  PIPELINE INTERRUPTED BY USER")
    print("=" * 80)
    print()
    output_file = "output/validation/dataset_partial.json"
    if pipeline.dataset:
        pipeline.export(output_file)
        print(f"💾 Partial dataset exported to: {output_file}")
    print()

except Exception as e:
    print()
    print("=" * 80)
    print(f"❌ PIPELINE ERROR: {e}")
    print("=" * 80)
    import traceback
    traceback.print_exc()
    print()
    output_file = "output/validation/dataset_partial.json"
    if pipeline.dataset:
        pipeline.export(output_file)
        print(f"💾 Partial dataset exported to: {output_file}")
    print()

# Analyze pipeline statistics
print("=" * 80)
print("📊 PIPELINE STATISTICS")
print("=" * 80)
print()

stats = pipeline.stats
print(f"Chunks processed: {stats.processed_chunks}/{stats.total_chunks}")
print(f"Questions generated: {stats.total_questions_generated}")
print(f"QA pairs created: {stats.total_qa_pairs}")
print(f"QA pairs passed: {stats.passed_qa_pairs}")
print(f"QA pairs rejected: {stats.rejected_qa_pairs}")
print()

if stats.total_qa_pairs > 0:
    rejection_rate = (stats.rejected_qa_pairs / (stats.passed_qa_pairs + stats.rejected_qa_pairs)) * 100
    acceptance_rate = (stats.passed_qa_pairs / stats.total_qa_pairs) * 100
    
    print(f"📈 KEY METRICS:")
    print(f"  First-attempt rejection rate: {rejection_rate:.1f}%")
    print(f"  Final acceptance rate: {acceptance_rate:.1f}%")
    print()
    
    if 30 <= rejection_rate <= 50:
        print("✅ STATUS: Rejection rate within target range (30-50%)!")
    elif rejection_rate < 30:
        print("⚠️  STATUS: Rejection rate below target (critic may still be too lenient)")
    else:
        print("⚠️  STATUS: Rejection rate above target (critic may be too strict)")
    print()

# Analyze dataset for patterns
print("=" * 80)
print("📋 DATASET ANALYSIS")
print("=" * 80)
print()

if pipeline.dataset:
    print(f"Total entries in dataset: {len(pipeline.dataset)}")
    
    # Count by chunk type
    chunk_types = defaultdict(int)
    for entry in pipeline.dataset:
        chunk_types[entry.chunk.chunk_type] += 1
    
    print(f"\nEntries by chunk type:")
    for chunk_type, count in sorted(chunk_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {chunk_type:15s}: {count:3d} entries")
    
    # Sample some QA pairs
    print(f"\n📝 SAMPLE QA PAIRS (First 3):")
    for i, entry in enumerate(pipeline.dataset[:3], 1):
        print(f"\n{i}. Chunk: {entry.chunk.chunk_id} ({entry.chunk.chunk_type})")
        print(f"   Q: {entry.question[:100]}...")
        print(f"   A: {entry.answer[:100]}...")
    print()

# Generate summary report
report_path = "VALIDATION_REPORT.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(f"# Critic Validation Report\n\n")
    f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write(f"## Test Configuration\n\n")
    f.write(f"- Chunks processed: {NUM_CHUNKS}\n")
    f.write(f"- Questions per chunk: {QUESTIONS_PER_CHUNK}\n")
    f.write(f"- Max retries: {MAX_RETRIES}\n")
    f.write(f"- Generator model: {config.generator_model}\n")
    f.write(f"- Critic model: {config.critic_model}\n\n")
    
    f.write(f"## Pipeline Statistics\n\n")
    f.write(f"- **Chunks processed:** {stats.processed_chunks}/{stats.total_chunks}\n")
    f.write(f"- **Questions generated:** {stats.total_questions_generated}\n")
    f.write(f"- **QA pairs created:** {stats.total_qa_pairs}\n")
    f.write(f"- **QA pairs passed:** {stats.passed_qa_pairs}\n")
    f.write(f"- **QA pairs rejected:** {stats.rejected_qa_pairs}\n\n")
    
    if stats.total_qa_pairs > 0:
        rejection_rate = (stats.rejected_qa_pairs / (stats.passed_qa_pairs + stats.rejected_qa_pairs)) * 100
        acceptance_rate = (stats.passed_qa_pairs / stats.total_qa_pairs) * 100
        
        f.write(f"## Key Metrics\n\n")
        f.write(f"- **Rejection rate:** {rejection_rate:.1f}%\n")
        f.write(f"- **Acceptance rate:** {acceptance_rate:.1f}%\n\n")
        
        f.write(f"## Target Achievement\n\n")
        if 30 <= rejection_rate <= 50:
            f.write(f"✅ **SUCCESS** - Rejection rate {rejection_rate:.1f}% is within target range (30-50%)\n\n")
        elif rejection_rate < 30:
            f.write(f"⚠️  **NEEDS ADJUSTMENT** - Rejection rate {rejection_rate:.1f}% is below target\n\n")
        else:
            f.write(f"⚠️  **TOO STRICT** - Rejection rate {rejection_rate:.1f}% is above target\n\n")
    
    f.write(f"## Dataset Summary\n\n")
    if pipeline.dataset:
        f.write(f"- **Total entries:** {len(pipeline.dataset)}\n")
        f.write(f"- **Entries by chunk type:**\n")
        for chunk_type, count in sorted(chunk_types.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  - {chunk_type}: {count}\n")
    f.write(f"\n")

print(f"✅ Report saved: {report_path}")
print()

print("=" * 80)
print("✅ VALIDATION COMPLETE!")
print("=" * 80)
print()
print("📁 Generated files:")
print(f"  1. output/validation/dataset.json - Full dataset")
print(f"  2. {report_path} - Summary report")
