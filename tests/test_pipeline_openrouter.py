"""
Test Pipeline avec OpenRouter
==============================

Test complet du pipeline avec:
- Generator: Mistral Small 3.1 24B 
- Critic: Llama 3.3 70B
- Retry loop activé (max 2)
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add paths
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'chunking'))
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'agents'))
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'orchestrator'))
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'utils'))

from openrouter_client import create_openrouter_client, OPENROUTER_MODELS
from pipeline import DatasetPipeline, PipelineConfig

print("=" * 60)
print("TEST PIPELINE - OPENROUTER + RETRY LOOP")
print("=" * 60)
print()

# Configuration
config = PipelineConfig(
    pdf_path="data/pdfs/M2_cours.pdf",
    output_dir="output_openrouter",
    max_chunks=2,  # Test rapide avec 2 chunks
    questions_per_chunk=2,
    generator_model=OPENROUTER_MODELS["generator"],
    critic_model=OPENROUTER_MODELS["critic"],
    max_retries=2,  # Boucle de retry activée!
    temperature=0.7,
    language="fr"
)

print("📋 CONFIGURATION")
print("-" * 40)
print(f"PDF: {config.pdf_path}")
print(f"Chunks max: {config.max_chunks}")
print(f"Questions/chunk: {config.questions_per_chunk}")
print(f"Max retries: {config.max_retries}")
print()
print(f"🤖 Generator: {config.generator_model}")
print(f"🔍 Critic: {config.critic_model}")
print()

# Create client
print("🔌 Connexion à OpenRouter...")
client = create_openrouter_client()
print("✅ Client créé!")
print()

# Create and run pipeline
try:
    pipeline = DatasetPipeline(
        config=config,
        llm_client=client
    )
    
    print("🚀 Démarrage du pipeline...")
    print()
    
    dataset = pipeline.run()
    
    print()
    print("=" * 60)
    print("RÉSULTATS")
    print("=" * 60)
    print(f"✅ Dataset généré: {len(dataset)} QA pairs")
    print()
    
    # Statistics
    stats = pipeline.stats
    print("📊 STATISTIQUES:")
    print(f"  Chunks traités: {stats.processed_chunks}/{stats.total_chunks}")
    print(f"  Questions générées: {stats.total_questions_generated}")
    print(f"  QA pairs évalués: {stats.total_qa_pairs}")
    print(f"  Acceptés: {stats.passed_qa_pairs}")
    print(f"  Rejetés: {stats.rejected_qa_pairs}")
    print(f"  Taux de passage: {stats.pass_rate * 100:.1f}%")
    
    if stats.total_retries > 0:
        print()
        print(f"🔄 RETRIES (workflow agentic):")
        print(f"  Total retries: {stats.total_retries}")
        print(f"  Succès après retry: {stats.passed_after_retry}")
    
    if stats.rejection_reasons:
        print()
        print("❌ RAISONS DE REJET:")
        for criterion, count in stats.rejection_reasons.items():
            print(f"  - {criterion}: {count}")
    
    # Show first QA pair
    if dataset:
        print()
        print("=" * 60)
        print("EXEMPLE DE QA PAIR VALIDÉ")
        print("=" * 60)
        first = dataset[0]
        print(f"Question: {first.question}")
        print(f"Réponse: {first.answer[:200]}...")
        print(f"Score: {first.critic_score:.2f}")
        print(f"Type: {first.question_type} | Difficulté: {first.difficulty}")
    
    # Export
    print()
    print("💾 Export du dataset...")
    json_path = pipeline.export_json("dataset_openrouter.json")
    print(f"✅ Exporté: {json_path}")
    
    hf_path = pipeline.export_huggingface("dataset_openrouter_hf.jsonl")
    print(f"✅ Format HuggingFace: {hf_path}")
    
except Exception as e:
    print(f"❌ ERREUR: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("✨ Test terminé!")
print("=" * 60)
