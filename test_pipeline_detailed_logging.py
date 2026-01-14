"""
Test Pipeline avec Logging Détaillé du Critic
==============================================

Capture TOUTES les informations du Critic:
- Score par critère (anchoring, clarity, etc.)
- Raisons exactes des rejets
- Feedback formaté pour les retries
- Timeline complète de chaque QA pair

Sauvegarde dans un fichier JSON détaillé.
"""

import sys
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / 'src' / 'chunking'))
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'agents'))
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'orchestrator'))
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'utils'))

from ollama_client import create_ollama_client, OLLAMA_MODELS
from pipeline import DatasetPipeline, PipelineConfig

print("=" * 100)
print("TEST PIPELINE - LOGGING DÉTAILLÉ DU CRITIC")
print("=" * 100)
print()

# Configuration minimale pour test rapide
config = PipelineConfig(
    pdf_path="data/pdfs/M2_cours.pdf",
    output_dir="output_detailed",
    max_chunks=3,  # Seulement 3 chunks pour voir les détails
    questions_per_chunk=2,  # 2 questions par chunk
    generator_model=OLLAMA_MODELS["generator"],
    critic_model=OLLAMA_MODELS["critic"],
    max_retries=2,
    temperature=0.7,
    language="fr"
)

print("📋 CONFIGURATION:")
print(f"  Chunks: {config.max_chunks}")
print(f"  Questions/chunk: {config.questions_per_chunk}")
print(f"  Max retries: {config.max_retries}")
print(f"  Generator: {config.generator_model}")
print(f"  Critic: {config.critic_model}")
print()

# Client
client = create_ollama_client()

# Stockage des évaluations détaillées
detailed_evaluations = []

# Monkey-patch le Critic pour logger toutes les évaluations
from critic_agent import CriticAgent
original_evaluate = CriticAgent.evaluate

def logged_evaluate(self, qa_pair, chunk):
    """Wrapper qui capture l'évaluation complète"""
    evaluation = original_evaluate(self, qa_pair, chunk)
    
    # Convertir en dict pour logger
    eval_dict = evaluation.to_dict()
    eval_dict['timestamp'] = datetime.now().isoformat()
    eval_dict['chunk_id'] = chunk.chunk_id  # Correction: c'est chunk_id, pas id
    
    detailed_evaluations.append(eval_dict)
    
    return evaluation

CriticAgent.evaluate = logged_evaluate

# Run pipeline
try:
    pipeline = DatasetPipeline(config=config, llm_client=client)
    dataset = pipeline.run()
    
    print()
    print("=" * 100)
    print("PIPELINE TERMINÉ - SAUVEGARDE DES DÉTAILS")
    print("=" * 100)
    print()
    
    # Sauvegarder toutes les évaluations détaillées
    detailed_report = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "config": {
                "chunks": config.max_chunks,
                "questions_per_chunk": config.questions_per_chunk,
                "max_retries": config.max_retries,
                "generator": config.generator_model,
                "critic": config.critic_model,
                "threshold": 0.90
            },
            "stats": {
                "total_evaluations": len(detailed_evaluations),
                "total_retries": pipeline.stats.total_retries,
                "passed_after_retry": pipeline.stats.passed_after_retry,
                "passed": pipeline.stats.passed_qa_pairs,
                "rejected": pipeline.stats.rejected_qa_pairs
            }
        },
        "evaluations": detailed_evaluations,
        "final_dataset": [
            {
                "question": entry.question,
                "answer": entry.answer,
                "chunk_id": entry.chunk_id,
                "score": entry.critic_score
            }
            for entry in dataset
        ]
    }
    
    # Sauvegarder
    output_file = "critic_detailed_log.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_report, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Rapport détaillé sauvegardé: {output_file}")
    print()
    print(f"📊 Statistiques:")
    print(f"   Total évaluations: {len(detailed_evaluations)}")
    print(f"   Retries: {pipeline.stats.total_retries}")
    print(f"   Acceptés: {pipeline.stats.passed_qa_pairs}")
    print(f"   Rejetés: {pipeline.stats.rejected_qa_pairs}")
    print()
    
    # Afficher quelques exemples de rejets
    rejections = [e for e in detailed_evaluations if e['decision'] == 'reject']
    
    if rejections:
        print("=" * 100)
        print(f"❌ DÉTAILS DES REJETS ({len(rejections)} rejets)")
        print("=" * 100)
        print()
        
        for i, rejection in enumerate(rejections[:3], 1):  # Premiers 3 rejets
            print(f"REJET #{i}")
            print("-" * 100)
            print(f"Question: {rejection['question'][:100]}...")
            print(f"Score global: {rejection['overall_score']:.3f}")
            print()
            print("Critères échoués:")
            for criterion in rejection['failed_criteria']:
                details = rejection['criteria_details'].get(criterion, {})
                score = details.get('score', 'N/A')
                explanation = details.get('explanation', 'N/A')
                print(f"  • {criterion}: {score:.2f}")
                print(f"    → {explanation}")
            print()
            print("Raisons de rejet:")
            for reason in rejection['rejection_reasons']:
                print(f"  - {reason}")
            print()
    
    else:
        print("⚠️  Aucun rejet détecté dans ce run")
        print()
        print("Tous les QA ont été acceptés (éventuellement après retry)")
    
    print()
    print("=" * 100)
    print("💡 ANALYSE DU FICHIER")
    print("=" * 100)
    print()
    print(f"Le fichier '{output_file}' contient:")
    print("  • Chaque évaluation du Critic avec:")
    print("    - Score pour les 5 critères")
    print("    - Explication de chaque critère")
    print("    - Décision (pass/reject)")
    print("    - Raisons de rejet si applicable")
    print()
    print("Pour analyser:")
    print(f"  python analyze_critic_log.py")

except Exception as e:
    print(f"❌ ERREUR: {e}")
    import traceback
    traceback.print_exc()
