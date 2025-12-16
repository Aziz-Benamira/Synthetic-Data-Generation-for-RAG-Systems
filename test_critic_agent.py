"""
Test Critic Agent - Pipeline Complet avec Évaluation Qualité
============================================================

Pipeline: Chunk → Questions → Réponses → Critic → Dataset Final
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
from critic_agent import CriticAgent, FinalDecision, get_evaluation_summary
import json
import time

# Vérifier la clé API
api_key = os.getenv("GROQ_API_KEY")
if not api_key or api_key == "ta_cle_api_ici":
    print("❌ ERREUR: Configure ta clé API dans le fichier .env")
    sys.exit(1)

from groq import Groq


def test_full_pipeline_with_critic():
    """Pipeline complet: Chunk → Q → A → Critic → Dataset."""
    
    print("=" * 70)
    print("PIPELINE COMPLET AVEC CRITIC AGENT")
    print("=" * 70)
    print()
    
    # Initialiser le client
    client = Groq(api_key=api_key)
    print("✅ Client Groq initialisé")
    
    # Charger les chunks
    print("\n" + "─" * 70)
    print("ÉTAPE 1: CHARGEMENT DES CHUNKS")
    print("─" * 70)
    
    chunker = SemanticChunker('data/pdfs/M2_cours.pdf')
    all_chunks = chunker.chunk_document()
    print(f"📚 {len(all_chunks)} chunks disponibles")
    
    # Sélectionner 3 chunks variés pour le test
    test_chunks = []
    
    # Chercher un bon chunk définition
    for c in all_chunks:
        if c.semantic_type == "definition" and 500 < len(c.content) < 1200:
            test_chunks.append(c)
            break
    
    # Chercher un chunk exemple
    for c in all_chunks:
        if c.semantic_type == "example" and 400 < len(c.content) < 1000:
            test_chunks.append(c)
            break
    
    # Chercher un chunk texte d'un autre chapitre
    for c in all_chunks:
        if c.semantic_type == "text" and 500 < len(c.content) < 1000:
            if len(test_chunks) < 2 or c.chapter_title != test_chunks[0].chapter_title:
                test_chunks.append(c)
                break
    
    print(f"🎯 {len(test_chunks)} chunks sélectionnés pour le test")
    for i, c in enumerate(test_chunks, 1):
        print(f"   {i}. [{c.semantic_type}] {c.chunk_id} - {c.section_title[:40]}...")
    
    # Initialiser les agents
    print("\n" + "─" * 70)
    print("INITIALISATION DES AGENTS")
    print("─" * 70)
    
    q_generator = QuestionGenerator(
        llm_client=client,
        model_name="llama-3.3-70b-versatile",
        language="fr",
        temperature=0.7
    )
    print("✅ Question Generator prêt")
    
    a_generator = AnswerGenerator(
        llm_client=client,
        model_name="llama-3.3-70b-versatile",
        language="fr",
        temperature=0.3
    )
    print("✅ Answer Generator prêt")
    
    critic = CriticAgent(
        llm_client=client,
        model_name="llama-3.3-70b-versatile",
        language="fr",
        temperature=0.2,
        strict_mode=True  # Tous les critères doivent passer
    )
    print("✅ Critic Agent prêt (mode strict)")
    
    # Pipeline principal
    print("\n" + "─" * 70)
    print("ÉTAPE 2: GÉNÉRATION DES QA PAIRS")
    print("─" * 70)
    
    all_qa_pairs = []
    qa_chunk_pairs = []  # Pour garder le lien qa → chunk
    
    for chunk_idx, chunk in enumerate(test_chunks, 1):
        print(f"\n📄 Chunk {chunk_idx}/{len(test_chunks)}: {chunk.chunk_id}")
        print(f"   Type: {chunk.semantic_type}, {len(chunk.content)} chars")
        
        # Générer 2 questions
        print("   🔹 Génération de 2 questions...")
        questions = q_generator.generate_from_chunk(chunk, num_questions=2)
        time.sleep(0.5)  # Rate limiting
        
        # Générer les réponses
        for q in questions:
            print(f"   🔹 Génération réponse pour: {q.question[:50]}...")
            answer = a_generator.generate_answer(q, chunk)
            qa_pair = QAPair.from_question_and_answer(q, answer)
            all_qa_pairs.append(qa_pair)
            qa_chunk_pairs.append((qa_pair, chunk))
            time.sleep(0.5)  # Rate limiting
    
    print(f"\n✅ {len(all_qa_pairs)} QA pairs générés")
    
    # Évaluation par le Critic
    print("\n" + "─" * 70)
    print("ÉTAPE 3: ÉVALUATION PAR LE CRITIC")
    print("─" * 70)
    
    evaluations = []
    
    for i, (qa_pair, chunk) in enumerate(qa_chunk_pairs, 1):
        print(f"\n🔍 Évaluation {i}/{len(qa_chunk_pairs)}:")
        print(f"   Q: {qa_pair.question[:60]}...")
        
        evaluation = critic.evaluate(qa_pair, chunk)
        evaluations.append(evaluation)
        
        # Afficher le résultat
        decision_icon = "✅" if evaluation.decision == FinalDecision.PASS else "❌"
        print(f"   {decision_icon} Décision: {evaluation.decision.value.upper()}")
        print(f"   📊 Score global: {evaluation.overall_score:.2f}")
        
        # Détails des critères
        for crit_name, crit_eval in evaluation.criteria_evaluations.items():
            icon = "✓" if crit_eval.score >= 0.7 else "✗"
            print(f"      {icon} {crit_name}: {crit_eval.score:.2f}")
        
        if evaluation.rejection_reasons:
            print(f"   ⚠️  Raisons rejet: {', '.join(evaluation.rejection_reasons[:2])}")
        
        time.sleep(0.5)  # Rate limiting
    
    # Résumé
    print("\n" + "=" * 70)
    print("RÉSUMÉ FINAL")
    print("=" * 70)
    
    passed = [e for e in evaluations if e.decision == FinalDecision.PASS]
    rejected = [e for e in evaluations if e.decision == FinalDecision.REJECT]
    
    print(f"\n📊 STATISTIQUES:")
    print(f"   Total QA pairs: {len(evaluations)}")
    print(f"   ✅ Acceptés: {len(passed)} ({100*len(passed)/len(evaluations):.1f}%)")
    print(f"   ❌ Rejetés: {len(rejected)} ({100*len(rejected)/len(evaluations):.1f}%)")
    
    avg_score = sum(e.overall_score for e in evaluations) / len(evaluations)
    print(f"   📈 Score moyen: {avg_score:.2f}")
    
    # Scores par critère
    print(f"\n📋 SCORES MOYENS PAR CRITÈRE:")
    for crit in ["anchoring", "local_answerability", "factual_accuracy", "completeness", "clarity"]:
        avg = sum(e.criteria_evaluations[crit].score for e in evaluations) / len(evaluations)
        bar = "█" * int(avg * 10) + "░" * (10 - int(avg * 10))
        print(f"   {crit:25} [{bar}] {avg:.2f}")
    
    # Dataset final (seulement les PASS)
    print("\n" + "─" * 70)
    print("DATASET FINAL (QA PAIRS ACCEPTÉS)")
    print("─" * 70)
    
    final_dataset = []
    for eval, (qa, chunk) in zip(evaluations, qa_chunk_pairs):
        if eval.decision == FinalDecision.PASS:
            entry = {
                "question": qa.question,
                "answer": qa.answer,
                "question_type": qa.question_type,
                "difficulty": qa.difficulty,
                "metadata": {
                    "chunk_id": qa.chunk_id,
                    "chapter": qa.chapter,
                    "section": qa.section,
                    "page_range": list(qa.page_range),
                    "source_file": qa.source_file
                },
                "quality_scores": {
                    crit: eval.criteria_evaluations[crit].score
                    for crit in eval.criteria_evaluations
                },
                "overall_quality_score": eval.overall_score
            }
            final_dataset.append(entry)
            
            print(f"\n✅ QA #{len(final_dataset)}:")
            print(f"   Q: {qa.question}")
            print(f"   R: {qa.answer[:150]}..." if len(qa.answer) > 150 else f"   R: {qa.answer}")
            print(f"   Score: {eval.overall_score:.2f}")
    
    # Afficher aussi les rejetés pour comprendre
    if rejected:
        print("\n" + "─" * 70)
        print("QA PAIRS REJETÉS (pour analyse)")
        print("─" * 70)
        
        for eval, (qa, chunk) in zip(evaluations, qa_chunk_pairs):
            if eval.decision == FinalDecision.REJECT:
                print(f"\n❌ REJETÉ:")
                print(f"   Q: {qa.question}")
                print(f"   R: {qa.answer[:100]}...")
                print(f"   Raisons: {', '.join(eval.rejection_reasons)}")
    
    # Sauvegarder
    output = {
        "summary": {
            "total_generated": len(all_qa_pairs),
            "total_accepted": len(final_dataset),
            "total_rejected": len(rejected),
            "acceptance_rate": len(final_dataset) / len(all_qa_pairs) if all_qa_pairs else 0,
            "average_quality_score": avg_score
        },
        "dataset": final_dataset,
        "rejected_for_analysis": [
            {
                "question": qa.question,
                "answer": qa.answer,
                "rejection_reasons": eval.rejection_reasons,
                "scores": {c: eval.criteria_evaluations[c].score for c in eval.criteria_evaluations}
            }
            for eval, (qa, _) in zip(evaluations, qa_chunk_pairs)
            if eval.decision == FinalDecision.REJECT
        ]
    }
    
    with open("dataset_with_critic_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n\n📁 Dataset sauvegardé: dataset_with_critic_evaluation.json")
    print(f"   - {len(final_dataset)} QA pairs de haute qualité")
    
    print("\n" + "=" * 70)
    print("PIPELINE TERMINÉ")
    print("=" * 70)


if __name__ == "__main__":
    test_full_pipeline_with_critic()
