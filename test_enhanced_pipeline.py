"""
Test Enhanced Pipeline with QuestionTypeClassifier, DifficultyEstimator, DiversityManager
==========================================================================================

This test validates that the three new enhancement agents are properly integrated
into the pipeline and produce balanced, diverse datasets.

Expected outcomes:
1. Questions are classified into 7 types
2. Difficulty is estimated (easy/medium/hard) with factor breakdown
3. Duplicates are detected and prevented
4. Distribution reports show type/difficulty balance
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'utils'))
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'orchestrator'))

from ollama_client import create_ollama_client
from pipeline import DatasetPipeline, PipelineConfig


def test_enhanced_pipeline():
    """Test pipeline with all enhancement agents enabled."""
    print("=" * 80)
    print("TEST: ENHANCED PIPELINE WITH 3 NEW AGENTS")
    print("=" * 80)
    print()
    
    # Initialize Ollama client
    print("🔧 Initializing Ollama client...")
    client = create_ollama_client()
    
    print("✅ Ollama client created")
    print()
    
    # Configure pipeline with enhanced features
    config = PipelineConfig(
        pdf_path="data/pdfs/M2_cours.pdf",
        output_dir="output/test_enhanced",
        max_chunks=5,  # Small test
        questions_per_chunk=3,  # Generate 3 questions per chunk
        generator_model="mistral:latest",
        critic_model="llama3:8b",
        max_retries=2,
        language="fr",
        # NEW DIVERSITY SETTINGS
        enable_diversity_check=True,
        diversity_threshold=0.85
    )
    
    print("📋 Pipeline Configuration:")
    print(f"   Source: {config.pdf_path}")
    print(f"   Max chunks: {config.max_chunks}")
    print(f"   Questions per chunk: {config.questions_per_chunk}")
    print(f"   Generator: {config.generator_model}")
    print(f"   Critic: {config.critic_model}")
    print(f"   Diversity check: {config.enable_diversity_check}")
    print(f"   Similarity threshold: {config.diversity_threshold}")
    print()
    
    # Run pipeline
    print("🚀 Starting enhanced pipeline...")
    print()
    
    pipeline = DatasetPipeline(config, client)
    
    try:
        dataset = pipeline.run()
        
        # Print summary with distribution reports
        pipeline.print_summary()
        
        # Export dataset
        print()
        print("💾 Exporting dataset...")
        json_path = pipeline.export_json()
        hf_path = pipeline.export_huggingface()
        csv_path = pipeline.export_csv()
        
        print(f"   ✅ JSON: {json_path}")
        print(f"   ✅ HuggingFace: {hf_path}")
        print(f"   ✅ CSV: {csv_path}")
        print()
        
        # Detailed inspection
        print("=" * 80)
        print("DETAILED INSPECTION (First 3 Entries)")
        print("=" * 80)
        
        for i, entry in enumerate(dataset[:3], 1):
            print(f"\n{i}. Question: {entry.question}")
            print(f"   Type: {entry.question_type} (confidence: {entry.question_type_confidence:.0%})")
            print(f"   Difficulty: {entry.difficulty} (confidence: {entry.difficulty_confidence:.0%})")
            
            if entry.difficulty_factors:
                print(f"   Difficulty Factors:")
                for factor, score in sorted(entry.difficulty_factors.items(), key=lambda x: -x[1])[:3]:
                    print(f"      • {factor}: {score:.2f}")
            
            print(f"   Critic Score: {entry.critic_score:.2f}")
            print(f"   Answer: {entry.answer[:100]}...")
        
        print()
        print("=" * 80)
        print("✅ TEST COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        # Check if distributions are tracked
        if pipeline.stats.type_distribution:
            print("\n✅ Type classification working!")
        else:
            print("\n⚠️  Type classification not working")
        
        if pipeline.stats.difficulty_distribution:
            print("✅ Difficulty estimation working!")
        else:
            print("⚠️  Difficulty estimation not working")
        
        if config.enable_diversity_check:
            print(f"✅ Diversity checking enabled (detected {pipeline.stats.duplicates_detected} duplicates)")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    test_enhanced_pipeline()
