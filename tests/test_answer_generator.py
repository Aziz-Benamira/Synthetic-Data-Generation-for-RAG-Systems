"""
Test Answer Generator avec VRAI LLM (Groq)
==========================================

Génère des questions puis des réponses ancrées dans le chunk source.
"""

import sys
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, 'src/chunking')
sys.path.insert(0, 'src/agents')

from semantic_chunker import SemanticChunker
from question_generator import QuestionGenerator
from answer_generator import AnswerGenerator, QAPair
import json

# Vérifier la clé API
api_key = os.getenv("GROQ_API_KEY")
if not api_key or api_key == "ta_cle_api_ici":
    print("❌ ERREUR: Configure ta clé API dans le fichier .env")
    sys.exit(1)

from groq import Groq


def test_full_pipeline():
    """Test complet: Chunk → Questions → Réponses → QA Pairs."""
    
    print("=" * 70)
    print("TEST: Pipeline Question → Answer Generator")
    print("=" * 70)
    
    # Créer le client Groq
    client = Groq(api_key=api_key)
    print("✅ Client Groq initialisé")
    
    # Charger un chunk
    print("\n[1] Chargement d'un chunk...")
    chunker = SemanticChunker('data/pdfs/M2_cours.pdf')
    chunks = chunker.chunk_document()
    
    # Trouver un bon chunk
    test_chunk = None
    for chunk in chunks:
        if chunk.semantic_type == "definition" and 500 < len(chunk.content) < 1200:
            test_chunk = chunk
            break
    
    if not test_chunk:
        test_chunk = chunks[5]  # Fallback
    
    print(f"    Chunk: {test_chunk.chunk_id}")
    print(f"    Type: {test_chunk.semantic_type}")
    print(f"    Pages: {test_chunk.page_range}")
    print(f"\n    Contenu ({len(test_chunk.content)} chars):")
    print(f"    {'-'*60}")
    print(f"    {test_chunk.content[:600]}...")
    print(f"    {'-'*60}")
    
    # Générer des questions
    print("\n[2] Génération de 2 questions...")
    q_generator = QuestionGenerator(
        llm_client=client,
        model_name="llama-3.3-70b-versatile",
        language="fr",
        default_num_questions=2,
        temperature=0.7
    )
    
    questions = q_generator.generate_from_chunk(test_chunk, num_questions=2)
    print(f"    ✅ {len(questions)} questions générées")
    
    for i, q in enumerate(questions, 1):
        print(f"\n    Q{i}: {q.question}")
        print(f"        Type: {q.question_type.value}, Difficulté: {q.difficulty.value}")
    
    # Générer des réponses
    print("\n[3] Génération des réponses...")
    a_generator = AnswerGenerator(
        llm_client=client,
        model_name="llama-3.3-70b-versatile",
        language="fr",
        temperature=0.3  # Plus bas pour des réponses factuelles
    )
    
    qa_pairs = []
    for i, question in enumerate(questions, 1):
        print(f"\n    Génération réponse pour Q{i}...")
        answer = a_generator.generate_answer(question, test_chunk)
        qa_pair = QAPair.from_question_and_answer(question, answer)
        qa_pairs.append(qa_pair)
        
        print(f"    ✅ Réponse générée (confiance: {answer.confidence:.2f})")
    
    # Afficher les résultats
    print("\n" + "=" * 70)
    print("RÉSULTATS: Paires Question-Réponse")
    print("=" * 70)
    
    for i, qa in enumerate(qa_pairs, 1):
        print(f"\n{'='*60}")
        print(f"QA PAIR #{i}")
        print(f"{'='*60}")
        print(f"\n📝 QUESTION [{qa.question_type}, {qa.difficulty}]:")
        print(f"   {qa.question}")
        print(f"\n✅ RÉPONSE (confiance: {qa.confidence:.0%}):")
        print(f"   {qa.answer}")
        if qa.supporting_quotes:
            print(f"\n📌 CITATIONS DU TEXTE:")
            for quote in qa.supporting_quotes[:2]:
                print(f"   \"{quote[:100]}...\"" if len(quote) > 100 else f"   \"{quote}\"")
    
    # Sauvegarder
    output = {
        "source_chunk": {
            "chunk_id": test_chunk.chunk_id,
            "chapter": test_chunk.chapter_title,
            "section": test_chunk.section_title,
            "pages": list(test_chunk.page_range),
            "content": test_chunk.content
        },
        "qa_pairs": [qa.to_dict() for qa in qa_pairs]
    }
    
    with open("test_output_qa_pairs.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n\n📁 Résultats sauvegardés dans: test_output_qa_pairs.json")
    
    # Afficher le format final
    print("\n" + "=" * 70)
    print("FORMAT DE SORTIE (pour dataset HuggingFace)")
    print("=" * 70)
    print(json.dumps(qa_pairs[0].to_dict(), indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 70)
    print("TEST TERMINÉ")
    print("=" * 70)


if __name__ == "__main__":
    test_full_pipeline()
