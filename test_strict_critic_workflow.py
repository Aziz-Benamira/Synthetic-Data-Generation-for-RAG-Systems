"""
Test du Retry Loop - Workflow Agentic RÉEL
===========================================

Ce test vérifie que le système multi-agent fonctionne vraiment
en déclenchant le retry loop du Critic.

Objectif: Voir des REJECT → FEEDBACK → RETRY → PASS
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / 'src' / 'chunking'))
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'agents'))
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'orchestrator'))
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'utils'))

from openrouter_client import create_openrouter_client, OPENROUTER_MODELS
from pipeline import DatasetPipeline, PipelineConfig

print("=" * 80)
print("TEST RETRY LOOP - WORKFLOW AGENTIC AVEC CRITIC STRICT")
print("=" * 80)
print()

# Configuration avec Critic STRICT
config = PipelineConfig(
    pdf_path="data/pdfs/M2_cours.pdf",
    output_dir="output_strict_critic",
    max_chunks=5,  # Test avec 5 chunks pour plus de variété
    questions_per_chunk=3,  # 3 questions par chunk
    generator_model=OPENROUTER_MODELS["generator"],
    critic_model=OPENROUTER_MODELS["critic"],
    max_retries=2,  # Retry loop activé
    temperature=0.7,
    language="fr"
)

print("📋 CONFIGURATION - MODE STRICT")
print("-" * 80)
print(f"PDF: {config.pdf_path}")
print(f"Chunks: {config.max_chunks}")
print(f"Questions/chunk: {config.questions_per_chunk}")
print(f"Max retries: {config.max_retries}")
print()
print(f"🤖 Generator: {config.generator_model}")
print(f"🔍 Critic: {config.critic_model} (THRESHOLD = 0.85 - STRICT!)")
print()
print("🎯 OBJECTIF: Déclencher le retry loop au moins 3 fois")
print("   Si tout passe → système trop laxiste")
print("   Si retry loop déclenché → workflow agentic fonctionne!")
print()

# Create client
client = create_openrouter_client()

# Create and run pipeline
try:
    pipeline = DatasetPipeline(
        config=config,
        llm_client=client
    )
    
    print("🚀 Démarrage du pipeline avec Critic STRICT...")
    print()
    
    dataset = pipeline.run()
    
    print()
    print("=" * 80)
    print("RÉSULTATS - WORKFLOW AGENTIC")
    print("=" * 80)
    print()
    
    stats = pipeline.stats
    
    # Statistiques globales
    print("📊 STATISTIQUES GLOBALES:")
    print(f"  Chunks traités: {stats.processed_chunks}/{stats.total_chunks}")
    print(f"  Questions générées: {stats.total_questions_generated}")
    print(f"  QA pairs évalués: {stats.total_qa_pairs}")
    print(f"  Acceptés: {stats.passed_qa_pairs}")
    print(f"  Rejetés: {stats.rejected_qa_pairs}")
    print(f"  Taux de passage: {stats.pass_rate * 100:.1f}%")
    print()
    
    # ANALYSE DU WORKFLOW AGENTIC
    print("=" * 80)
    print("🔄 ANALYSE DU WORKFLOW AGENTIC")
    print("=" * 80)
    print()
    
    if hasattr(stats, 'total_retries') and stats.total_retries > 0:
        print(f"✅ RETRY LOOP DÉCLENCHÉ: {stats.total_retries} fois!")
        print()
        print("   Le système multi-agent fonctionne:")
        print(f"   1. Critic rejette des QA ({stats.rejected_qa_pairs} rejets)")
        print(f"   2. Feedback formaté ({stats.total_retries} feedbacks)")
        print(f"   3. Régénération Q+A ({stats.total_retries} retries)")
        
        if hasattr(stats, 'passed_after_retry'):
            print(f"   4. Succès après retry: {stats.passed_after_retry} QA")
        
        print()
        print("   🎉 CE N'EST PAS UNE BASELINE - C'EST UN VRAI WORKFLOW AGENTIC!")
        
    else:
        print("⚠️  AUCUN RETRY DÉCLENCHÉ")
        print()
        print("   Possibilités:")
        print("   1. Les QA générés sont parfaits dès le premier coup")
        print("   2. Le Critic n'est pas assez strict (augmenter threshold?)")
        print("   3. Les chunks sont trop simples")
        print()
        print("   ⚠️  RISQUE: Le système ressemble à une baseline (appels LLM séquentiels)")
    
    print()
    
    # Raisons de rejet
    if stats.rejection_reasons:
        print("=" * 80)
        print("❌ CRITÈRES DE REJET (Preuve de l'évaluation stricte)")
        print("=" * 80)
        print()
        
        total_failures = sum(stats.rejection_reasons.values())
        for criterion, count in sorted(stats.rejection_reasons.items(), key=lambda x: -x[1]):
            percentage = (count / total_failures) * 100
            bar = "█" * int(percentage / 5)
            print(f"  {criterion:20s} │ {bar} {count:2d} ({percentage:5.1f}%)")
        
        print()
        print("   Ces rejets montrent que le Critic évalue selon les 5 critères!")
    
    # Exemples de QA validés
    if len(dataset) > 0:
        print()
        print("=" * 80)
        print(f"✅ DATASET FINAL: {len(dataset)} QA pairs validés")
        print("=" * 80)
        print()
        
        # Show first 2
        for i, entry in enumerate(dataset[:2], 1):
            print(f"Exemple {i}:")
            print(f"  Q: {entry.question}")
            print(f"  R: {entry.answer[:150]}...")
            print(f"  Score: {entry.critic_score:.2f}")
            print(f"  Type: {entry.question_type} | Difficulté: {entry.difficulty}")
            print()
    
    # Export
    if len(dataset) > 0:
        print("💾 Export du dataset...")
        json_path = pipeline.export_json("dataset_strict_critic.json")
        print(f"✅ Exporté: {json_path}")
    
    print()
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()
    
    if hasattr(stats, 'total_retries') and stats.total_retries >= 3:
        print("🎯 SUCCÈS COMPLET!")
        print()
        print("   ✅ Le workflow agentic fonctionne avec:")
        print(f"      • {stats.total_retries} retries déclenchés")
        print(f"      • {stats.rejected_qa_pairs} rejets du Critic")
        print(f"      • {stats.passed_qa_pairs} QA validés après évaluation/retry")
        print()
        print("   Ce n'est PAS une simple baseline d'appels LLM!")
        print("   C'est un vrai système multi-agent avec boucle de feedback.")
        
    elif stats.rejected_qa_pairs > 0:
        print("⚠️  WORKFLOW PARTIEL")
        print()
        print(f"   Le Critic rejette ({stats.rejected_qa_pairs} rejets)")
        print(f"   Mais peu/pas de retries ({getattr(stats, 'total_retries', 0)} retries)")
        print()
        print("   Possible que max_retries soit atteint sans succès.")
        
    else:
        print("❌ PROBLÈME: 100% PASS RATE")
        print()
        print("   Le Critic accepte tout dès le premier coup.")
        print("   Cela ressemble à une baseline, pas à un workflow agentic.")
        print()
        print("   SOLUTIONS:")
        print("   1. Augmenter encore le threshold (0.85 → 0.90)")
        print("   2. Tester avec des chunks plus complexes")
        print("   3. Renforcer les prompts du Critic")

except Exception as e:
    print(f"❌ ERREUR: {e}")
    import traceback
    traceback.print_exc()
