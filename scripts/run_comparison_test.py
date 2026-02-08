"""
REAL COMPARISON TEST - Your Current Code vs Seif's Code
========================================================

This script runs the FULL pipeline with REAL data twice:
1. With YOUR current critic_agent.py
2. With SEIF'S critic_agent_seif.py

Then compares the concrete results for your presentation.
"""

import sys
import shutil
import json
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'chunking'))
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'agents'))
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'orchestrator'))
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'utils'))

from ollama_client import create_ollama_client, OLLAMA_MODELS
from pipeline import DatasetPipeline, PipelineConfig

def print_header(title):
    """Print a nice header"""
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100 + "\n")

def run_pipeline_test(version_name, use_seif_version=False):
    """
    Run the pipeline and capture metrics
    
    Args:
        version_name: "CURRENT" or "SEIF"
        use_seif_version: If True, use Seif's critic_agent
    
    Returns:
        dict with metrics
    """
    print_header(f"TEST {version_name} - Running Pipeline with Real Data")
    
    # Backup and swap if needed
    critic_path = Path("src/agents/critic_agent.py")
    seif_critic_path = Path("seif_changes_review/critic_agent_seif.py")
    backup_path = Path("src/agents/critic_agent_backup.py")
    
    if use_seif_version:
        print(f"📝 Temporarily using Seif's critic_agent.py...")
        shutil.copy2(critic_path, backup_path)
        shutil.copy2(seif_critic_path, critic_path)
        print(f"✅ Swapped to Seif's version\n")
    else:
        print(f"📝 Using YOUR current critic_agent.py...\n")
    
    # Reload modules to pick up changes
    if 'critic_agent' in sys.modules:
        del sys.modules['critic_agent']
    if 'pipeline' in sys.modules:
        del sys.modules['pipeline']
    
    # Re-import
    from pipeline import DatasetPipeline, PipelineConfig
    
    # Configuration - SAME for both tests
    config = PipelineConfig(
        pdf_path="data/pdfs/M2_cours.pdf",
        output_dir=f"output_comparison_{version_name.lower()}",
        max_chunks=10,  # 10 chunks for real test
        questions_per_chunk=2,  # 2 questions per chunk = 20 QA pairs
        generator_model=OLLAMA_MODELS["generator"],
        critic_model=OLLAMA_MODELS["critic"],
        max_retries=2,
        temperature=0.7,
        language="fr"
    )
    
    print("📋 Configuration:")
    print(f"   PDF: {config.pdf_path}")
    print(f"   Chunks: {config.max_chunks}")
    print(f"   Questions/chunk: {config.questions_per_chunk}")
    print(f"   Expected QA pairs: {config.max_chunks * config.questions_per_chunk}")
    print(f"   Max retries: {config.max_retries}")
    print()
    
    # Create client
    print("🔌 Connecting to Ollama...")
    client = create_ollama_client()
    print("✅ Connected!\n")
    
    # Run pipeline
    try:
        print(f"🚀 Running {version_name} pipeline...")
        print("   (This will take 5-10 minutes with real chunking + LLM calls)")
        print()
        
        start_time = datetime.now()
        
        pipeline = DatasetPipeline(
            config=config,
            llm_client=client
        )
        
        dataset = pipeline.run()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Collect metrics
        stats = pipeline.stats
        
        metrics = {
            "version": version_name,
            "duration_seconds": duration,
            "duration_minutes": duration / 60,
            
            # Core metrics
            "total_chunks": stats.total_chunks,
            "processed_chunks": stats.processed_chunks,
            "total_questions_generated": stats.total_questions_generated,
            "total_qa_pairs": stats.total_qa_pairs,
            "passed_qa_pairs": stats.passed_qa_pairs,
            "rejected_qa_pairs": stats.rejected_qa_pairs,
            
            # Key metrics for comparison
            "pass_rate": stats.pass_rate,
            "rejection_rate": 1 - stats.pass_rate,
            "total_retries": getattr(stats, 'total_retries', 0),
            "passed_after_retry": getattr(stats, 'passed_after_retry', 0),
            
            # Rejection reasons
            "rejection_reasons": dict(stats.rejection_reasons) if stats.rejection_reasons else {},
            
            # Dataset
            "final_dataset_size": len(dataset),
            
            # Scores
            "scores": [entry.critic_score for entry in dataset if hasattr(entry, 'critic_score')],
        }
        
        # Calculate score statistics
        if metrics["scores"]:
            metrics["score_mean"] = sum(metrics["scores"]) / len(metrics["scores"])
            metrics["score_min"] = min(metrics["scores"])
            metrics["score_max"] = max(metrics["scores"])
            metrics["score_std"] = (sum((x - metrics["score_mean"])**2 for x in metrics["scores"]) / len(metrics["scores"]))**0.5
        else:
            metrics["score_mean"] = 0
            metrics["score_min"] = 0
            metrics["score_max"] = 0
            metrics["score_std"] = 0
        
        print(f"\n✅ {version_name} pipeline completed in {duration/60:.1f} minutes")
        print()
        
        # Print results
        print_header(f"{version_name} RESULTS")
        
        print("📊 Generation:")
        print(f"   Chunks processed: {metrics['processed_chunks']}/{metrics['total_chunks']}")
        print(f"   Questions generated: {metrics['total_questions_generated']}")
        print(f"   QA pairs evaluated: {metrics['total_qa_pairs']}")
        print()
        
        print("🎯 Critic Evaluation:")
        print(f"   ✅ PASSED: {metrics['passed_qa_pairs']} ({metrics['pass_rate']*100:.1f}%)")
        print(f"   ❌ REJECTED: {metrics['rejected_qa_pairs']} ({metrics['rejection_rate']*100:.1f}%)")
        print()
        
        print("🔄 Retry Loop:")
        print(f"   Total retries triggered: {metrics['total_retries']}")
        print(f"   Passed after retry: {metrics['passed_after_retry']}")
        print()
        
        print("📈 Score Distribution:")
        print(f"   Mean: {metrics['score_mean']:.3f}")
        print(f"   Std Dev: {metrics['score_std']:.3f}")
        print(f"   Min: {metrics['score_min']:.3f}")
        print(f"   Max: {metrics['score_max']:.3f}")
        print()
        
        if metrics["rejection_reasons"]:
            print("❌ Rejection Reasons:")
            total_rejections = sum(metrics["rejection_reasons"].values())
            for criterion, count in sorted(metrics["rejection_reasons"].items(), key=lambda x: -x[1]):
                pct = (count / total_rejections) * 100 if total_rejections > 0 else 0
                bar = "█" * int(pct / 5)
                print(f"   {criterion:25s} │ {bar} {count:2d} ({pct:5.1f}%)")
            print()
        
        # Export results
        output_file = f"comparison_results_{version_name.lower()}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"💾 Metrics saved to: {output_file}")
        
        return metrics
        
    except Exception as e:
        print(f"\n❌ ERROR in {version_name} pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # Restore original if we swapped
        if use_seif_version and backup_path.exists():
            print(f"\n🔄 Restoring original critic_agent.py...")
            shutil.copy2(backup_path, critic_path)
            backup_path.unlink()
            print(f"✅ Restored\n")

def compare_metrics(current_metrics, seif_metrics):
    """Generate comparison report"""
    
    print_header("FINAL COMPARISON - CURRENT vs SEIF")
    
    print("🎯 REJECTION RATE (Main Objective)")
    print("-" * 100)
    print(f"   CURRENT:  {current_metrics['rejection_rate']*100:5.1f}%  {current_metrics['rejected_qa_pairs']}/{current_metrics['total_qa_pairs']} rejected")
    print(f"   SEIF:     {seif_metrics['rejection_rate']*100:5.1f}%  {seif_metrics['rejected_qa_pairs']}/{seif_metrics['total_qa_pairs']} rejected")
    
    improvement = seif_metrics['rejection_rate'] - current_metrics['rejection_rate']
    if improvement > 0:
        print(f"   📈 IMPROVEMENT: +{improvement*100:.1f} percentage points ({improvement/current_metrics['rejection_rate']*100:.0f}% increase)")
    else:
        print(f"   📉 DECREASE: {improvement*100:.1f} percentage points")
    print()
    
    print("🔄 RETRY LOOPS (Agentic Behavior)")
    print("-" * 100)
    print(f"   CURRENT:  {current_metrics['total_retries']} retries triggered")
    print(f"   SEIF:     {seif_metrics['total_retries']} retries triggered")
    
    retry_improvement = seif_metrics['total_retries'] - current_metrics['total_retries']
    if retry_improvement > 0:
        print(f"   📈 IMPROVEMENT: +{retry_improvement} more retries (more agentic!)")
    print()
    
    print("📊 SCORE DISTRIBUTION (Discrimination)")
    print("-" * 100)
    print(f"   CURRENT:  Mean={current_metrics['score_mean']:.3f}  StdDev={current_metrics['score_std']:.3f}  Range=[{current_metrics['score_min']:.2f}, {current_metrics['score_max']:.2f}]")
    print(f"   SEIF:     Mean={seif_metrics['score_mean']:.3f}  StdDev={seif_metrics['score_std']:.3f}  Range=[{seif_metrics['score_min']:.2f}, {seif_metrics['score_max']:.2f}]")
    
    std_improvement = seif_metrics['score_std'] - current_metrics['score_std']
    if std_improvement > 0:
        print(f"   📈 IMPROVEMENT: +{std_improvement:.3f} more variance (better discrimination!)")
    print()
    
    print("⏱️  RUNTIME")
    print("-" * 100)
    print(f"   CURRENT:  {current_metrics['duration_minutes']:.1f} minutes")
    print(f"   SEIF:     {seif_metrics['duration_minutes']:.1f} minutes")
    
    runtime_diff = seif_metrics['duration_minutes'] - current_metrics['duration_minutes']
    runtime_pct = (runtime_diff / current_metrics['duration_minutes']) * 100
    print(f"   {'📈 SLOWER' if runtime_diff > 0 else '📉 FASTER'}: {runtime_diff:+.1f} min ({runtime_pct:+.0f}%)")
    print()
    
    print("📋 SUMMARY FOR PRESENTATION")
    print("-" * 100)
    
    # Determine if objectives met
    rejection_target_met = seif_metrics['rejection_rate'] >= 0.30 and seif_metrics['rejection_rate'] <= 0.50
    variance_improved = seif_metrics['score_std'] > current_metrics['score_std'] * 1.5
    retries_improved = seif_metrics['total_retries'] > current_metrics['total_retries'] * 2
    
    if rejection_target_met:
        print("   ✅ OBJECTIVE 1: Rejection rate in target range (30-50%)")
    else:
        print(f"   ⚠️  OBJECTIVE 1: Rejection rate {seif_metrics['rejection_rate']*100:.1f}% (target: 30-50%)")
    
    if variance_improved:
        print("   ✅ OBJECTIVE 2: Score variance increased significantly")
    else:
        print("   ⚠️  OBJECTIVE 2: Score variance improvement marginal")
    
    if retries_improved:
        print("   ✅ OBJECTIVE 3: Retry loops show strong agentic behavior")
    else:
        print("   ⚠️  OBJECTIVE 3: Retry loop improvement moderate")
    
    print()
    
    # Generate presentation bullets
    print("🎤 KEY POINTS FOR PRESENTATION:")
    print("-" * 100)
    print(f"   1. 'Rejection rate increased from {current_metrics['rejection_rate']*100:.0f}% to {seif_metrics['rejection_rate']*100:.0f}%'")
    print(f"   2. 'Retry loops went from {current_metrics['total_retries']} to {seif_metrics['total_retries']} iterations'")
    print(f"   3. 'Score variance improved by {(std_improvement/current_metrics['score_std']*100):.0f}%, showing better discrimination'")
    print(f"   4. 'Runtime cost: +{runtime_pct:.0f}% but worth it for quality improvement'")
    print()
    
    # Save comparison
    comparison = {
        "test_date": datetime.now().isoformat(),
        "current": current_metrics,
        "seif": seif_metrics,
        "improvements": {
            "rejection_rate_increase_pct": improvement * 100,
            "retry_loops_increase": retry_improvement,
            "score_variance_increase": std_improvement,
            "runtime_increase_pct": runtime_pct,
        }
    }
    
    with open("comparison_final.json", 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    print("💾 Full comparison saved to: comparison_final.json")
    print()

def main():
    """Main comparison workflow"""
    
    print_header("REAL PIPELINE COMPARISON TEST")
    print("This will run the FULL pipeline TWICE with REAL data:")
    print("  1. Your current critic_agent.py")
    print("  2. Seif's critic_agent_seif.py")
    print()
    print("Expected duration: 15-20 minutes total")
    print("Press Ctrl+C to cancel...")
    print()
    
    input("Press ENTER to start...")
    
    # Test 1: Current version
    current_metrics = run_pipeline_test("CURRENT", use_seif_version=False)
    
    if not current_metrics:
        print("\n❌ Current version test failed. Aborting.")
        return
    
    print("\n⏸️  Test 1 complete. Starting test 2 in 5 seconds...")
    import time
    time.sleep(5)
    
    # Test 2: Seif's version
    seif_metrics = run_pipeline_test("SEIF", use_seif_version=True)
    
    if not seif_metrics:
        print("\n❌ Seif's version test failed.")
        return
    
    # Compare
    compare_metrics(current_metrics, seif_metrics)
    
    print_header("TEST COMPLETE")
    print("Results saved:")
    print("  - comparison_results_current.json")
    print("  - comparison_results_seif.json")
    print("  - comparison_final.json")
    print()
    print("Use these concrete numbers for your presentation!")

if __name__ == "__main__":
    main()
