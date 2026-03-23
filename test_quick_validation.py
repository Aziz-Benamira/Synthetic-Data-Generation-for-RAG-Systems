"""
Quick Test - 3 Chunks Only
===========================
Fast test to verify the pipeline works correctly.
"""

import sys
sys.path.insert(0, 'src')

from openai import OpenAI
from orchestrator.pipeline import DatasetPipeline, PipelineConfig
from datetime import datetime

print("=" * 80)
print("QUICK VALIDATION TEST - 3 CHUNKS")
print("=" * 80)
print()

# Create Ollama client
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Create pipeline
config = PipelineConfig(
    pdf_path="data/pdfs/M2_cours.pdf",
    output_dir="output/quick_test",
    generator_model="mistral:latest",
    critic_model="llama3:8b",
    language="fr",
    questions_per_chunk=3,
    max_retries=2,
    max_chunks=3,
    temperature=0.7
)

pipeline = DatasetPipeline(config=config, llm_client=client)

print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting pipeline with 3 chunks...")
print()

try:
    dataset = pipeline.run()
    
    print()
    print("=" * 80)
    print("✅ SUCCESS!")
    print("=" * 80)
    print()
    
    stats = pipeline.stats
    print(f"📊 Results:")
    print(f"  Chunks: {stats.processed_chunks}/{stats.total_chunks}")
    print(f"  Questions: {stats.total_questions_generated}")
    print(f"  QA pairs passed: {stats.passed_qa_pairs}")
    print(f"  QA pairs rejected (after all retries): {stats.rejected_qa_pairs}")
    print(f"  Total retry attempts: {stats.total_retries}")
    
    if stats.total_questions_generated > 0:
        initial_rejection_rate = (stats.total_retries / stats.total_questions_generated) * 100
        print(f"  \n🎯 Initial rejection rate (triggers retry): {initial_rejection_rate:.1f}%")
        
    if stats.passed_qa_pairs + stats.rejected_qa_pairs > 0:
        final_rejection_rate = stats.rejected_qa_pairs / (stats.passed_qa_pairs + stats.rejected_qa_pairs) * 100
        print(f"  🏁 Final rejection rate (after retries): {final_rejection_rate:.1f}%")
        
        if 30 <= initial_rejection_rate <= 50:
            print(f"  ✅ Initial rejection within target range (30-50%)!")
        elif initial_rejection_rate < 30:
            print(f"  ⚠️  Initial rejection below target")
        else:
            print(f"  ⚠️  Initial rejection above target (too strict)")
    print()
    
    # Export
    pipeline.export_json("output/quick_test/dataset.json")
    print("💾 Dataset exported to: output/quick_test/dataset.json")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
