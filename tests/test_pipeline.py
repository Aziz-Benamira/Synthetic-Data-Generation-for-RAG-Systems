"""
Test du Pipeline Complet
========================

Teste l'orchestrator sur quelques chunks pour valider le flux complet.
"""

import sys
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, 'src/orchestrator')
sys.path.insert(0, 'src/chunking')
sys.path.insert(0, 'src/agents')

from pipeline import DatasetPipeline, PipelineConfig, generate_dataset
from groq import Groq

api_key = os.getenv("GROQ_API_KEY")


def test_pipeline_small():
    """Test le pipeline sur un petit nombre de chunks."""
    
    print("=" * 70)
    print("TEST DU PIPELINE COMPLET")
    print("=" * 70)
    
    # Configuration pour test (petit nombre de chunks)
    config = PipelineConfig(
        pdf_path="data/pdfs/M2_cours.pdf",
        output_dir="output/test_pipeline",
        max_chunks=3,  # Seulement 3 chunks pour le test
        chunk_types=["definition", "theorem", "example"],  # Types intéressants
        min_chunk_length=300,
        questions_per_chunk=2,
        language="fr"
    )
    
    # Client LLM
    client = Groq(api_key=api_key)
    
    # Créer et exécuter le pipeline
    pipeline = DatasetPipeline(config, client)
    
    try:
        dataset = pipeline.run()
        
        # Exporter
        pipeline.export_json()
        pipeline.export_huggingface()
        pipeline.export_csv()
        
        # Résumé
        pipeline.print_summary()
        
        # Afficher quelques exemples
        if dataset:
            print("\n" + "=" * 70)
            print("EXEMPLES DU DATASET GÉNÉRÉ")
            print("=" * 70)
            
            for i, entry in enumerate(dataset[:3], 1):
                print(f"\n{'─'*50}")
                print(f"ENTRÉE #{i}")
                print(f"{'─'*50}")
                print(f"📄 Source: {entry.source_file} | Chunk: {entry.chunk_id}")
                print(f"📝 Type: {entry.question_type} | Difficulté: {entry.difficulty}")
                print(f"⭐ Score Critic: {entry.critic_score:.2f}")
                print(f"\nQ: {entry.question}")
                print(f"\nR: {entry.answer[:300]}..." if len(entry.answer) > 300 else f"\nR: {entry.answer}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_convenience_function():
    """Test la fonction de commodité."""
    
    print("\n" + "=" * 70)
    print("TEST FONCTION generate_dataset()")
    print("=" * 70)
    
    client = Groq(api_key=api_key)
    
    dataset, stats = generate_dataset(
        pdf_path="data/pdfs/M2_cours.pdf",
        llm_client=client,
        output_dir="output/test_convenience",
        max_chunks=2,
        questions_per_chunk=1,
        language="fr"
    )
    
    print(f"\n✅ Dataset généré: {len(dataset)} entrées")
    print(f"✅ Pass rate: {100*stats.pass_rate:.1f}%")


if __name__ == "__main__":
    print("\n🚀 Lancement du test pipeline...")
    print()
    
    success = test_pipeline_small()
    
    if success:
        print("\n" + "=" * 70)
        print("✅ TEST RÉUSSI!")
        print("=" * 70)
        print("\nFichiers générés dans output/test_pipeline/:")
        print("  - dataset.json (format complet)")
        print("  - dataset_hf.jsonl (format HuggingFace)")
        print("  - dataset.csv (format tabulaire)")
    else:
        print("\n❌ TEST ÉCHOUÉ")
