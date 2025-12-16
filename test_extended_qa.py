"""
Test Answer Generator - Version Étendue
=======================================

Génère plusieurs QA pairs à partir de différents chunks.
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


def test_extended_pipeline():
    """Génère plusieurs QA pairs à partir de différents types de chunks."""
    
    print("=" * 70)
    print("TEST ÉTENDU: Génération de multiples QA Pairs")
    print("=" * 70)
    
    # Créer le client Groq
    client = Groq(api_key=api_key)
    print("✅ Client Groq initialisé")
    
    # Charger les chunks
    print("\n[1] Chargement des chunks du PDF...")
    chunker = SemanticChunker('data/pdfs/M2_cours.pdf')
    chunks = chunker.chunk_document()
    print(f"    {len(chunks)} chunks disponibles")
    
    # Sélectionner des chunks variés (différents types)
    selected_chunks = []
    
    # 1 définition
    for c in chunks:
        if c.semantic_type == "definition" and 400 < len(c.content) < 1000:
            selected_chunks.append(c)
            break
    
    # 1 exemple
    for c in chunks:
        if c.semantic_type == "example" and 400 < len(c.content) < 1000:
            selected_chunks.append(c)
            break
    
    # 1 texte (d'un autre chapitre si possible)
    for c in chunks:
        if c.semantic_type == "text" and 500 < len(c.content) < 1200:
            if not selected_chunks or c.chapter_title != selected_chunks[0].chapter_title:
                selected_chunks.append(c)
                break
    
    print(f"    {len(selected_chunks)} chunks sélectionnés pour le test")
    
    # Initialiser les générateurs
    q_generator = QuestionGenerator(
        llm_client=client,
        model_name="llama-3.3-70b-versatile",
        language="fr",
        temperature=0.7
    )
    
    a_generator = AnswerGenerator(
        llm_client=client,
        model_name="llama-3.3-70b-versatile",
        language="fr",
        temperature=0.3
    )
    
    all_qa_pairs = []
    
    # Traiter chaque chunk
    for chunk_idx, chunk in enumerate(selected_chunks, 1):
        print(f"\n{'='*70}")
        print(f"CHUNK #{chunk_idx}: {chunk.chunk_id}")
        print(f"{'='*70}")
        print(f"Type: {chunk.semantic_type}")
        print(f"Chapitre: {chunk.chapter_title}")
        print(f"Section: {chunk.section_title}")
        print(f"Pages: {chunk.page_range}")
        print(f"\nContenu ({len(chunk.content)} chars):")
        print("-" * 50)
        print(chunk.content[:400] + "..." if len(chunk.content) > 400 else chunk.content)
        print("-" * 50)
        
        # Générer 2 questions par chunk
        print(f"\n📝 Génération de 2 questions...")
        questions = q_generator.generate_from_chunk(chunk, num_questions=2)
        print(f"   ✅ {len(questions)} questions générées")
        
        # Générer les réponses
        print(f"\n✍️  Génération des réponses...")
        for q_idx, question in enumerate(questions, 1):
            answer = a_generator.generate_answer(question, chunk)
            qa_pair = QAPair.from_question_and_answer(question, answer)
            all_qa_pairs.append(qa_pair)
            
            print(f"\n   --- QA Pair {len(all_qa_pairs)} ---")
            print(f"   Q: {question.question}")
            print(f"   Type: {qa_pair.question_type} | Difficulté: {qa_pair.difficulty}")
            print(f"   R: {qa_pair.answer[:150]}..." if len(qa_pair.answer) > 150 else f"   R: {qa_pair.answer}")
            print(f"   Confiance: {qa_pair.confidence:.0%}")
    
    # Résumé final
    print("\n" + "=" * 70)
    print(f"RÉSUMÉ: {len(all_qa_pairs)} QA PAIRS GÉNÉRÉS")
    print("=" * 70)
    
    # Stats par type
    types = {}
    difficulties = {}
    for qa in all_qa_pairs:
        types[qa.question_type] = types.get(qa.question_type, 0) + 1
        difficulties[qa.difficulty] = difficulties.get(qa.difficulty, 0) + 1
    
    print("\nDistribution par type:")
    for t, count in types.items():
        print(f"  - {t}: {count}")
    
    print("\nDistribution par difficulté:")
    for d, count in difficulties.items():
        print(f"  - {d}: {count}")
    
    avg_confidence = sum(qa.confidence for qa in all_qa_pairs) / len(all_qa_pairs)
    print(f"\nConfiance moyenne: {avg_confidence:.0%}")
    
    # Afficher tous les QA pairs
    print("\n" + "=" * 70)
    print("TOUTES LES QA PAIRS")
    print("=" * 70)
    
    for i, qa in enumerate(all_qa_pairs, 1):
        print(f"\n{'─'*60}")
        print(f"#{i} [{qa.question_type.upper()}] [{qa.difficulty}] (confiance: {qa.confidence:.0%})")
        print(f"{'─'*60}")
        print(f"📌 Source: {qa.chapter} > {qa.section} (p.{qa.page_range[0]}-{qa.page_range[1]})")
        print(f"\n❓ QUESTION:")
        print(f"   {qa.question}")
        print(f"\n✅ RÉPONSE:")
        # Formater la réponse avec indentation
        answer_lines = qa.answer.split('\n')
        for line in answer_lines:
            print(f"   {line}")
        if qa.supporting_quotes:
            print(f"\n📝 CITATION:")
            for quote in qa.supporting_quotes[:1]:
                q_preview = quote[:120] + "..." if len(quote) > 120 else quote
                print(f"   \"{q_preview}\"")
    
    # Sauvegarder
    output = {
        "summary": {
            "total_qa_pairs": len(all_qa_pairs),
            "types": types,
            "difficulties": difficulties,
            "avg_confidence": avg_confidence
        },
        "qa_pairs": [qa.to_dict() for qa in all_qa_pairs]
    }
    
    with open("test_output_extended_qa.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n\n📁 Tous les résultats sauvegardés dans: test_output_extended_qa.json")
    
    print("\n" + "=" * 70)
    print("TEST TERMINÉ")
    print("=" * 70)


if __name__ == "__main__":
    test_extended_pipeline()
