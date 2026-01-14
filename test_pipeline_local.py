"""
Test Pipeline Local - Ollama avec Retry Loop Strict
====================================================

Pipeline complet en local:
- Mistral 7B pour génération Q&A
- Phi-3 Mini pour critique stricte
- Retry loop activé pour déclencher le workflow agentic
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src' / 'chunking'))
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'agents'))
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'orchestrator'))
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'utils'))

from ollama_client import create_ollama_client, OLLAMA_MODELS
from pipeline import DatasetPipeline, PipelineConfig

print("=" * 80)
print("TEST PIPELINE LOCAL - OLLAMA + CRITIC STRICT")
print("=" * 80)
print()

# Configuration
config = PipelineConfig(
    pdf_path="data/pdfs/M2_cours.pdf",
    output_dir="output_local",
    max_chunks=5,  # 5 chunks pour test
    questions_per_chunk=3,  # 3 questions par chunk
    generator_model=OLLAMA_MODELS["generator"],  # mistral:latest
    critic_model=OLLAMA_MODELS["critic"],  # phi3:mini
    max_retries=2,  # Retry loop activé
    temperature=0.7,
    language="fr"
)

print("📋 CONFIGURATION - MODÈLES LOCAUX")
print("-" * 80)
print(f"PDF: {config.pdf_path}")
print(f"Chunks: {config.max_chunks}")
print(f"Questions/chunk: {config.questions_per_chunk}")
print(f"Max retries: {config.max_retries}")
print()
print(f"🤖 Generator: {config.generator_model} (Mistral 7B ~4.5GB)")
print(f"🔍 Critic: {config.critic_model} (Phi-3 Mini ~2.3GB)")
print(f"   Total VRAM: ~6.8GB / 7GB")
print()
print("🎯 OBJECTIF: Déclencher le retry loop avec Critic STRICT")
print("   Threshold = 0.90 (très strict)")
print()

# Create client
print("🔌 Connexion à Ollama local...")
client = create_ollama_client()
print("✅ Client créé!")
print()

# Create and run pipeline
try:
    pipeline = DatasetPipeline(
        config=config,
        llm_client=client
    )
    
    print("🚀 Démarrage du pipeline local...")
    print()
    
    dataset = pipeline.run()
    
    print()
    print("=" * 80)
    print("RÉSULTATS - WORKFLOW AGENTIC LOCAL")
    print("=" * 80)
    print()
    
    stats = pipeline.stats
    
    # Statistiques
    print("📊 STATISTIQUES:")
    print(f"  Chunks traités: {stats.processed_chunks}/{stats.total_chunks}")
    print(f"  Questions générées: {stats.total_questions_generated}")
    print(f"  QA pairs évalués: {stats.total_qa_pairs}")
    print(f"  Acceptés: {stats.passed_qa_pairs}")
    print(f"  Rejetés: {stats.rejected_qa_pairs}")
    print(f"  Taux de passage: {stats.pass_rate * 100:.1f}%")
    print()
    
    # WORKFLOW AGENTIC
    print("=" * 80)
    print("🔄 ANALYSE DU WORKFLOW AGENTIC")
    print("=" * 80)
    print()
    
    if hasattr(stats, 'total_retries') and stats.total_retries > 0:
        print(f"✅ RETRY LOOP DÉCLENCHÉ: {stats.total_retries} fois!")
        print()
        print("   Le système multi-agent fonctionne:")
        print(f"   1. Critic rejette ({stats.rejected_qa_pairs} rejets)")
        print(f"   2. Feedback formaté ({stats.total_retries} feedbacks)")
        print(f"   3. Régénération Q+A ({stats.total_retries} retries)")
        
        if hasattr(stats, 'passed_after_retry') and stats.passed_after_retry > 0:
            print(f"   4. Succès après retry: {stats.passed_after_retry} QA")
        
        print()
        print("   🎉 WORKFLOW AGENTIC VALIDÉ EN LOCAL!")
        
    else:
        print("⚠️  Aucun retry déclenché")
        print(f"   Rejetés: {stats.rejected_qa_pairs}")
        
        if stats.rejected_qa_pairs > 0:
            print("   Le Critic rejette mais max retries atteints sans succès")
        else:
            print("   Le Critic accepte tout → augmenter threshold?")
    
    print()
    
    # Raisons de rejet
    if stats.rejection_reasons:
        print("=" * 80)
        print("❌ CRITÈRES DE REJET")
        print("=" * 80)
        print()
        
        total = sum(stats.rejection_reasons.values())
        for criterion, count in sorted(stats.rejection_reasons.items(), key=lambda x: -x[1]):
            pct = (count / total) * 100
            bar = "█" * int(pct / 5)
            print(f"  {criterion:20s} │ {bar} {count:2d} ({pct:5.1f}%)")
        print()
    
    # Exemples
    if len(dataset) > 0:
        print("=" * 80)
        print(f"✅ DATASET FINAL: {len(dataset)} QA pairs")
        print("=" * 80)
        print()
        
        for i, entry in enumerate(dataset[:2], 1):
            print(f"Exemple {i}:")
            print(f"  Q: {entry.question}")
            print(f"  R: {entry.answer[:120]}...")
            print(f"  Score: {entry.critic_score:.2f}")
            print()
        
        # Export
        json_path = pipeline.export_json("dataset_local.json")
        print(f"💾 Exporté: {json_path}")
    
    print()
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()
    
    if hasattr(stats, 'total_retries') and stats.total_retries >= 3:
        print("🎯 SUCCÈS COMPLET!")
        print()
        print("   ✅ Workflow agentic fonctionnel en LOCAL")
        print(f"      • {stats.total_retries} retries")
        print(f"      • {stats.rejected_qa_pairs} rejets")
        print(f"      • {stats.passed_qa_pairs} validations")
        print()
        print("   ✅ Pas de rate limits!")
        print("   ✅ VRAM optimisée (6.8GB / 7GB)")
        
    elif stats.rejected_qa_pairs > 0:
        print("⚠️  Workflow partiel")
        print(f"   Rejets: {stats.rejected_qa_pairs}")
        print(f"   Retries: {getattr(stats, 'total_retries', 0)}")
        
    else:
        print("⚠️  100% PASS - Critic trop laxiste")
        print("   Solution: Augmenter threshold ou renforcer prompts")

except Exception as e:
    print(f"❌ ERREUR: {e}")
    import traceback
    traceback.print_exc()
