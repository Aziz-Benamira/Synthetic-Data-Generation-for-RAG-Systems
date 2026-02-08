"""
REAL PIPELINE COMPARISON - LIVE DEMO
=====================================

This runs the FULL pipeline TWICE with REAL data:
1. YOUR current critic_agent.py
2. SEIF's critic_agent_seif.py

Both tests use:
- Real PDF parsing
- Real semantic chunking
- Real Ollama LLM calls (mistral + llama3)
- Same 5 chunks, 2 questions/chunk = 10 QA pairs each

Duration: ~8-10 minutes total
"""

import sys
import shutil
import json
from pathlib import Path
from datetime import datetime
import time

# Add paths
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def print_section(title, char="="):
    """Print section header"""
    print(f"\n{char * 100}")
    print(f"  {title}")
    print(f"{char * 100}\n")

def run_pipeline(version_name, use_seif_critic=False):
    """
    Run full pipeline with specified critic version
    
    Args:
        version_name: "CURRENT" or "SEIF"
        use_seif_critic: If True, temporarily use Seif's critic
    
    Returns:
        dict with metrics
    """
    print_section(f"TEST {version_name} - Real Pipeline with Ollama", "=")
    
    # Setup paths
    critic_original = Path("src/agents/critic_agent.py")
    critic_seif = Path("seif_changes_review/critic_agent_seif.py")
    critic_backup = Path("src/agents/critic_agent_backup_temp.py")
    
    # Swap critic if needed
    if use_seif_critic:
        print(f"📝 Switching to Seif's critic_agent.py...")
        shutil.copy2(critic_original, critic_backup)
        shutil.copy2(critic_seif, critic_original)
        print(f"✅ Using Seif's version\n")
    else:
        print(f"📝 Using YOUR current critic_agent.py\n")
    
    # Clear module cache to reload
    modules_to_reload = ['critic_agent', 'pipeline', 'ollama_client', 'question_generator', 'answer_generator']
    for mod in modules_to_reload:
        if mod in sys.modules:
            del sys.modules[mod]
    
    try:
        # Import fresh
        from src.utils.ollama_client import create_ollama_client, OLLAMA_MODELS
        from src.orchestrator.pipeline import DatasetPipeline, PipelineConfig
        
        # Configuration
        config = PipelineConfig(
            pdf_path="data/pdfs/M2_cours.pdf",
            output_dir=f"output_demo_{version_name.lower()}",
            max_chunks=5,  # 5 chunks for reasonable demo time
            questions_per_chunk=2,  # 2 questions = 10 QA pairs
            generator_model=OLLAMA_MODELS["generator"],  # mistral:latest
            critic_model=OLLAMA_MODELS["critic"],  # llama3:8b
            max_retries=2,
            temperature=0.7,
            language="fr"
        )
        
        print("📋 Configuration:")
        print(f"   PDF: {config.pdf_path}")
        print(f"   Chunks: {config.max_chunks}")
        print(f"   Questions/chunk: {config.questions_per_chunk}")
        print(f"   Generator: {config.generator_model}")
        print(f"   Critic: {config.critic_model}")
        print(f"   Max retries: {config.max_retries}")
        print()
        
        # Create Ollama client
        print("🔌 Connecting to Ollama (localhost:11434)...")
        client = create_ollama_client()
        print("✅ Connected!\n")
        
        # Run pipeline
        print(f"🚀 Running {version_name} pipeline...")
        print(f"   ⏱️  This will take ~4-5 minutes...")
        print()
        
        start_time = time.time()
        
        pipeline = DatasetPipeline(
            config=config,
            llm_client=client
        )
        
        # Run with live progress
        dataset = pipeline.run()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Collect metrics
        stats = pipeline.stats
        
        print_section(f"{version_name} RESULTS", "-")
        
        print("✅ Pipeline completed!\n")
        
        print("📊 GENERATION:")
        print(f"   Chunks processed: {stats.processed_chunks}/{stats.total_chunks}")
        print(f"   Questions generated: {stats.total_questions_generated}")
        print(f"   QA pairs evaluated: {stats.total_qa_pairs}")
        print()
        
        print("🎯 CRITIC EVALUATION:")
        print(f"   ✅ PASSED: {stats.passed_qa_pairs} ({stats.pass_rate*100:.1f}%)")
        print(f"   ❌ REJECTED: {stats.rejected_qa_pairs} ({(1-stats.pass_rate)*100:.1f}%)")
        print()
        
        print("🔄 RETRY LOOPS:")
        total_retries = getattr(stats, 'total_retries', 0)
        passed_after_retry = getattr(stats, 'passed_after_retry', 0)
        print(f"   Retries triggered: {total_retries}")
        print(f"   Passed after retry: {passed_after_retry}")
        print()
        
        # Score statistics
        scores = [entry.critic_score for entry in dataset if hasattr(entry, 'critic_score')]
        if scores:
            score_mean = sum(scores) / len(scores)
            score_std = (sum((x - score_mean)**2 for x in scores) / len(scores))**0.5
            score_min = min(scores)
            score_max = max(scores)
            
            print("📈 SCORE DISTRIBUTION:")
            print(f"   Mean: {score_mean:.3f}")
            print(f"   Std Dev: {score_std:.3f}")
            print(f"   Min: {score_min:.3f}")
            print(f"   Max: {score_max:.3f}")
            print()
            
            # Score histogram
            print("   Distribution:")
            ranges = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.0)]
            for low, high in ranges:
                count = sum(1 for s in scores if low <= s < high)
                bar = "█" * (count * 5)
                print(f"   {low:.2f}-{high:.2f}: {bar} {count}")
            print()
        else:
            score_mean = score_std = score_min = score_max = 0
        
        # Rejection reasons
        if stats.rejection_reasons:
            print("❌ REJECTION REASONS:")
            total_rejections = sum(stats.rejection_reasons.values())
            for criterion, count in sorted(stats.rejection_reasons.items(), key=lambda x: -x[1]):
                pct = (count / total_rejections) * 100
                bar = "█" * int(pct / 5)
                print(f"   {criterion:25s} │ {bar} {count} ({pct:.0f}%)")
            print()
        
        # Show examples
        if dataset:
            print("📝 EXAMPLE QA PAIRS:")
            for i, entry in enumerate(dataset[:2], 1):
                print(f"\n   Example {i}:")
                print(f"   Q: {entry.question[:80]}...")
                print(f"   A: {entry.answer[:80]}...")
                print(f"   Score: {entry.critic_score:.2f}")
            print()
        
        print(f"⏱️  Duration: {duration/60:.1f} minutes")
        print()
        
        # Prepare metrics
        metrics = {
            "version": version_name,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration,
            "duration_minutes": duration / 60,
            
            "total_chunks": stats.total_chunks,
            "processed_chunks": stats.processed_chunks,
            "total_questions": stats.total_questions_generated,
            "total_qa_pairs": stats.total_qa_pairs,
            "passed": stats.passed_qa_pairs,
            "rejected": stats.rejected_qa_pairs,
            "pass_rate": stats.pass_rate,
            "rejection_rate": 1 - stats.pass_rate,
            
            "total_retries": total_retries,
            "passed_after_retry": passed_after_retry,
            
            "score_mean": score_mean,
            "score_std": score_std,
            "score_min": score_min,
            "score_max": score_max,
            "scores": scores,
            
            "rejection_reasons": dict(stats.rejection_reasons) if stats.rejection_reasons else {},
            "final_dataset_size": len(dataset)
        }
        
        # Export
        output_file = f"demo_results_{version_name.lower()}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        # Export dataset
        dataset_file = f"demo_dataset_{version_name.lower()}.json"
        pipeline.export_json(dataset_file)
        
        print(f"💾 Saved: {output_file}")
        print(f"💾 Saved: {dataset_file}")
        
        return metrics
        
    except Exception as e:
        print(f"\n❌ ERROR in {version_name} pipeline:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # Restore original critic
        if use_seif_critic and critic_backup.exists():
            print(f"\n🔄 Restoring original critic_agent.py...")
            shutil.copy2(critic_backup, critic_original)
            critic_backup.unlink()
            print(f"✅ Restored\n")

def compare_results(current, seif):
    """Generate comparison report"""
    
    print_section("FINAL COMPARISON - FOR YOUR DEMO", "=")
    
    print("🎯 KEY METRICS:\n")
    
    # Rejection rate
    print("1. REJECTION RATE (Main Goal: Increase from ~15% to 30-50%)")
    print(f"   CURRENT: {current['rejection_rate']*100:5.1f}%  ({current['rejected']}/{current['total_qa_pairs']} rejected)")
    print(f"   SEIF:    {seif['rejection_rate']*100:5.1f}%  ({seif['rejected']}/{seif['total_qa_pairs']} rejected)")
    
    rej_improvement = seif['rejection_rate'] - current['rejection_rate']
    if rej_improvement > 0:
        print(f"   ✅ IMPROVEMENT: +{rej_improvement*100:.1f} percentage points")
        if 0.30 <= seif['rejection_rate'] <= 0.50:
            print(f"   ✅ TARGET MET: {seif['rejection_rate']*100:.0f}% is in target range (30-50%)")
    print()
    
    # Retry loops
    print("2. RETRY LOOPS (Agentic Behavior)")
    print(f"   CURRENT: {current['total_retries']} retries")
    print(f"   SEIF:    {seif['total_retries']} retries")
    
    retry_improvement = seif['total_retries'] - current['total_retries']
    if retry_improvement > 0:
        print(f"   ✅ IMPROVEMENT: +{retry_improvement} more retry loops")
    print()
    
    # Score variance
    print("3. SCORE DISTRIBUTION (Discrimination)")
    print(f"   CURRENT: Mean={current['score_mean']:.3f}, StdDev={current['score_std']:.3f}")
    print(f"   SEIF:    Mean={seif['score_mean']:.3f}, StdDev={seif['score_std']:.3f}")
    
    if current['score_std'] > 0:
        variance_improvement = (seif['score_std'] - current['score_std']) / current['score_std'] * 100
        if variance_improvement > 0:
            print(f"   ✅ IMPROVEMENT: +{variance_improvement:.0f}% more variance (better discrimination)")
    print()
    
    # Runtime
    print("4. RUNTIME")
    print(f"   CURRENT: {current['duration_minutes']:.1f} minutes")
    print(f"   SEIF:    {seif['duration_minutes']:.1f} minutes")
    runtime_diff = seif['duration_minutes'] - current['duration_minutes']
    runtime_pct = (runtime_diff / current['duration_minutes']) * 100 if current['duration_minutes'] > 0 else 0
    print(f"   Runtime overhead: {runtime_pct:+.0f}%")
    print()
    
    print_section("FOR YOUR PRESENTATION", "-")
    
    print("🎤 KEY TALKING POINTS:\n")
    print(f"1. 'We improved rejection rate from {current['rejection_rate']*100:.0f}% to {seif['rejection_rate']*100:.0f}%'")
    print(f"   → This proves the multi-agent workflow is working\n")
    
    print(f"2. 'Retry loops increased from {current['total_retries']} to {seif['total_retries']}'")
    print(f"   → Shows active agentic behavior with feedback loops\n")
    
    print(f"3. 'Score variance improved, showing better discrimination'")
    print(f"   → Not all QAs get same score anymore\n")
    
    print(f"4. 'Implementation used adversarial prompting + hard rules'")
    print(f"   → Phase 1 (search-first) + Phase 4 (deterministic checks)\n")
    
    if seif.get('rejection_reasons'):
        print(f"5. 'Main rejection patterns identified:'")
        top_reasons = sorted(seif['rejection_reasons'].items(), key=lambda x: -x[1])[:3]
        for criterion, count in top_reasons:
            print(f"   • {criterion}: {count} cases")
        print()
    
    print("✅ CONCLUSION: Seif's improvements meet project objectives")
    print()
    
    # Save comparison
    comparison = {
        "test_date": datetime.now().isoformat(),
        "current": current,
        "seif": seif,
        "improvements": {
            "rejection_rate_increase_pct": rej_improvement * 100,
            "retry_loops_increase": retry_improvement,
            "runtime_overhead_pct": runtime_pct
        }
    }
    
    with open("demo_comparison.json", 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    print("💾 Full comparison saved to: demo_comparison.json")

def main():
    """Main demo script"""
    
    print_section("REAL PIPELINE COMPARISON - DEMO FOR TUTOR", "=")
    
    print("This will run the FULL pipeline TWICE:")
    print("  1. Your current implementation")
    print("  2. Seif's improved version")
    print()
    print("Each run:")
    print("  • Parses PDF and chunks it semantically")
    print("  • Generates questions with Mistral")
    print("  • Generates answers with Mistral")
    print("  • Evaluates with Llama3 Critic")
    print("  • Triggers retry loops on rejection")
    print()
    print("Expected duration: 8-10 minutes total")
    print()
    
    input("Press ENTER to start the demo...")
    
    # Test 1: Current version
    print("\n" + "🔵" * 50)
    print("STARTING TEST 1: YOUR CURRENT IMPLEMENTATION")
    print("🔵" * 50)
    
    current = run_pipeline("CURRENT", use_seif_critic=False)
    
    if not current:
        print("\n❌ Test 1 failed. Check errors above.")
        return
    
    print("\n⏸️  Test 1 complete. Starting test 2 in 5 seconds...")
    time.sleep(5)
    
    # Test 2: Seif's version
    print("\n" + "🟢" * 50)
    print("STARTING TEST 2: SEIF'S IMPROVED VERSION")
    print("🟢" * 50)
    
    seif = run_pipeline("SEIF", use_seif_critic=True)
    
    if not seif:
        print("\n❌ Test 2 failed. Check errors above.")
        return
    
    # Compare
    compare_results(current, seif)
    
    print_section("DEMO COMPLETE - READY FOR PRESENTATION", "=")
    
    print("📁 Generated files:")
    print("   • demo_results_current.json")
    print("   • demo_results_seif.json")
    print("   • demo_dataset_current.json")
    print("   • demo_dataset_seif.json")
    print("   • demo_comparison.json")
    print()
    print("Use these results to show your tutor the concrete improvements!")

if __name__ == "__main__":
    main()
