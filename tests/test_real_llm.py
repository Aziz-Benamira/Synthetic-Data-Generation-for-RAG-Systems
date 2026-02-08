"""
Test Question Generator avec VRAI LLM (Groq - Gratuit)
======================================================

Utilise Llama 3.1 70B via Groq pour générer de vraies questions.
"""

import sys
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

sys.path.insert(0, 'src/chunking')
sys.path.insert(0, 'src/agents')

from semantic_chunker import SemanticChunker
from question_generator import (
    QuestionGenerator, 
    QuestionType,
    estimate_question_potential
)
import json

# Vérifier la clé API
api_key = os.getenv("GROQ_API_KEY")
if not api_key or api_key == "ta_cle_api_ici":
    print("❌ ERREUR: Configure ta clé API dans le fichier .env")
    print("   1. Ouvre le fichier .env")
    print("   2. Remplace 'ta_cle_api_ici' par ta vraie clé Groq")
    sys.exit(1)

print("✅ Clé API Groq trouvée")

# Installer groq si nécessaire
try:
    from groq import Groq
except ImportError:
    print("Installation du package groq...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "groq", "-q"])
    from groq import Groq


def test_with_real_llm():
    """Test avec un vrai LLM via Groq."""
    
    print("\n" + "=" * 70)
    print("TEST: Question Generator avec Groq (Llama 3.1 70B)")
    print("=" * 70)
    
    # Créer le client Groq
    client = Groq(api_key=api_key)
    
    # Test simple pour vérifier la connexion
    print("\n[1] Test de connexion à Groq...")
    try:
        test_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Réponds juste 'OK' si tu me reçois."}],
            max_tokens=10
        )
        print(f"    ✅ Connexion OK: {test_response.choices[0].message.content}")
    except Exception as e:
        print(f"    ❌ Erreur de connexion: {e}")
        return
    
    # Charger un chunk du PDF français
    print("\n[2] Chargement d'un chunk du PDF français...")
    chunker = SemanticChunker('data/pdfs/M2_cours.pdf')
    chunks = chunker.chunk_document()
    
    # Trouver un bon chunk de type définition
    test_chunk = None
    for chunk in chunks:
        if chunk.semantic_type == "definition" and 400 < len(chunk.content) < 1000:
            test_chunk = chunk
            break
    
    if not test_chunk:
        test_chunk = chunks[0]
    
    print(f"    Chunk: {test_chunk.chunk_id}")
    print(f"    Type: {test_chunk.semantic_type}")
    print(f"    Taille: {len(test_chunk.content)} caractères")
    print(f"\n    Contenu:")
    print(f"    {'-'*60}")
    print(f"    {test_chunk.content[:500]}...")
    print(f"    {'-'*60}")
    
    # Créer le générateur avec Groq
    print("\n[3] Génération de questions avec Llama 3.3 70B...")
    
    generator = QuestionGenerator(
        llm_client=client,
        model_name="llama-3.3-70b-versatile",
        language="fr",
        default_num_questions=3,
        temperature=0.7
    )
    
    # Générer les questions
    try:
        questions = generator.generate_from_chunk(test_chunk, num_questions=3)
        
        print(f"\n    ✅ {len(questions)} questions générées!\n")
        
        for i, q in enumerate(questions, 1):
            print(f"    {'='*60}")
            print(f"    Q{i}: {q.question}")
            print(f"    {'='*60}")
            print(f"    Type: {q.question_type.value}")
            print(f"    Difficulté: {q.difficulty.value}")
            print(f"    Concepts: {', '.join(q.key_concepts)}")
            print(f"    Indices: {', '.join(q.expected_answer_hints[:2])}")
            print()
        
        # Sauvegarder en JSON
        output = {
            "chunk_id": test_chunk.chunk_id,
            "chunk_content": test_chunk.content,
            "questions": [q.to_dict() for q in questions]
        }
        
        with open("test_output_questions.json", "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"    📁 Résultat sauvegardé dans: test_output_questions.json")
        
    except Exception as e:
        print(f"    ❌ Erreur lors de la génération: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("TEST TERMINÉ")
    print("=" * 70)


if __name__ == "__main__":
    test_with_real_llm()
