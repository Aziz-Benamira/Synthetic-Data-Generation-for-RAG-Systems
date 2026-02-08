"""
Test Critic Agent avec QA INTENTIONNELLEMENT MAUVAIS
====================================================

Vérifie que le Critic rejette bien les QA de mauvaise qualité.
"""

import sys
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, 'src/chunking')
sys.path.insert(0, 'src/agents')

from semantic_chunker import SemanticChunker
from answer_generator import QAPair
from critic_agent import CriticAgent, FinalDecision
import json

api_key = os.getenv("GROQ_API_KEY")
from groq import Groq


def create_bad_qa_pairs(chunk):
    """Créer des QA pairs intentionnellement mauvais pour tester le Critic."""
    
    bad_pairs = []
    
    # 1. HALLUCINATION - Réponse avec information inventée
    bad_pairs.append(QAPair(
        question="Qu'est-ce qu'une tribu en théorie des probabilités ?",
        answer="Une tribu est une collection de sous-ensembles qui a été inventée par le mathématicien français Pierre de Fermat en 1654 lors de sa correspondance avec Blaise Pascal sur les jeux de hasard.",
        question_type="factual",
        difficulty="easy",
        supporting_quotes=[],  # Pas de citation !
        chunk_id=chunk.chunk_id,
        source_file="test",
        page_range=chunk.page_range,
        chapter=chunk.chapter_title,
        section=chunk.section_title,
        confidence=0.9
    ))
    
    # 2. NON-ANCRÉ - Réponse correcte mais pas dans le chunk
    bad_pairs.append(QAPair(
        question="Quelle est la mesure de Lebesgue sur les boréliens ?",
        answer="La mesure de Lebesgue est l'unique mesure sur les boréliens de R qui est invariante par translation et qui attribue la valeur 1 à l'intervalle [0,1].",
        question_type="conceptual",
        difficulty="medium",
        supporting_quotes=[],
        chunk_id=chunk.chunk_id,
        source_file="test",
        page_range=chunk.page_range,
        chapter=chunk.chapter_title,
        section=chunk.section_title,
        confidence=0.8
    ))
    
    # 3. QUESTION TROP VAGUE
    bad_pairs.append(QAPair(
        question="C'est quoi le truc avec les ensembles ?",
        answer="Les ensembles sont des collections d'objets mathématiques.",
        question_type="conceptual",
        difficulty="easy",
        supporting_quotes=[],
        chunk_id=chunk.chunk_id,
        source_file="test",
        page_range=chunk.page_range,
        chapter=chunk.chapter_title,
        section=chunk.section_title,
        confidence=0.5
    ))
    
    # 4. RÉPONSE INCOMPLÈTE
    bad_pairs.append(QAPair(
        question="Qu'est-ce qu'une sous-tribu de F et quelles sont ses propriétés ?",
        answer="Une sous-tribu.",  # Réponse trop courte/incomplète
        question_type="conceptual",
        difficulty="medium",
        supporting_quotes=[],
        chunk_id=chunk.chunk_id,
        source_file="test",
        page_range=chunk.page_range,
        chapter=chunk.chapter_title,
        section=chunk.section_title,
        confidence=0.3
    ))
    
    # 5. ERREUR FACTUELLE - Contredit le chunk
    bad_pairs.append(QAPair(
        question="Qu'est-ce qu'une intersection de tribus ?",
        answer="Une intersection de tribus n'est PAS une tribu, contrairement à une réunion de tribus qui est toujours une tribu.",
        question_type="factual",
        difficulty="easy",
        supporting_quotes=["Une intersection de tribus est une tribu"],  # Citation qui contredit la réponse !
        chunk_id=chunk.chunk_id,
        source_file="test",
        page_range=chunk.page_range,
        chapter=chunk.chapter_title,
        section=chunk.section_title,
        confidence=0.9
    ))
    
    # 6. QUESTION HORS CONTEXTE - Nécessite info externe
    bad_pairs.append(QAPair(
        question="En utilisant le théorème de Radon-Nikodym et la définition de tribu, démontrez que toute mesure finie est régulière.",
        answer="La démonstration utilise le fait qu'une tribu est fermée par union dénombrable et complémentation.",
        question_type="procedural",
        difficulty="hard",
        supporting_quotes=[],
        chunk_id=chunk.chunk_id,
        source_file="test",
        page_range=chunk.page_range,
        chapter=chunk.chapter_title,
        section=chunk.section_title,
        confidence=0.4
    ))
    
    return bad_pairs


def test_critic_with_bad_qa():
    """Tester le Critic avec des QA intentionnellement mauvais."""
    
    print("=" * 70)
    print("TEST: CRITIC AVEC QA INTENTIONNELLEMENT MAUVAIS")
    print("=" * 70)
    print()
    print("Ce test vérifie que le Critic rejette bien les QA de mauvaise qualité.")
    print()
    
    client = Groq(api_key=api_key)
    
    # Charger un chunk
    chunker = SemanticChunker('data/pdfs/M2_cours.pdf')
    chunks = chunker.chunk_document()
    
    test_chunk = None
    for c in chunks:
        if c.semantic_type == "definition" and len(c.content) > 500:
            test_chunk = c
            break
    
    print(f"Chunk de test: {test_chunk.chunk_id}")
    print(f"Contenu ({len(test_chunk.content)} chars):")
    print("-" * 50)
    print(test_chunk.content[:400] + "...")
    print("-" * 50)
    
    # Créer les mauvais QA
    bad_qa_pairs = create_bad_qa_pairs(test_chunk)
    
    # Labels des problèmes
    problem_labels = [
        "HALLUCINATION - Info inventée (Fermat/Pascal)",
        "NON-ANCRÉ - Réponse hors chunk (mesure de Lebesgue)",
        "QUESTION VAGUE - 'C'est quoi le truc'",
        "RÉPONSE INCOMPLÈTE - Juste 'Une sous-tribu.'",
        "ERREUR FACTUELLE - Contredit le chunk",
        "HORS CONTEXTE - Nécessite Radon-Nikodym"
    ]
    
    # Initialiser le Critic
    critic = CriticAgent(
        llm_client=client,
        model_name="llama-3.3-70b-versatile",
        language="fr",
        temperature=0.2,
        strict_mode=True
    )
    
    print(f"\n{'='*70}")
    print("ÉVALUATION DES {len(bad_qa_pairs)} QA MAUVAIS")
    print("="*70)
    
    results = []
    
    for i, (qa, label) in enumerate(zip(bad_qa_pairs, problem_labels), 1):
        print(f"\n{'─'*70}")
        print(f"TEST #{i}: {label}")
        print(f"{'─'*70}")
        print(f"Q: {qa.question}")
        print(f"R: {qa.answer[:100]}..." if len(qa.answer) > 100 else f"R: {qa.answer}")
        
        evaluation = critic.evaluate(qa, test_chunk)
        results.append((label, evaluation))
        
        # Afficher résultat
        if evaluation.decision == FinalDecision.REJECT:
            print(f"\n✅ CORRECTEMENT REJETÉ!")
        else:
            print(f"\n❌ FAUX POSITIF - Aurait dû être rejeté!")
        
        print(f"   Score global: {evaluation.overall_score:.2f}")
        print(f"   Critères échoués: {evaluation.failed_criteria}")
        if evaluation.rejection_reasons:
            print(f"   Raisons: {evaluation.rejection_reasons[:2]}")
    
    # Résumé
    print("\n" + "=" * 70)
    print("RÉSUMÉ DU TEST")
    print("=" * 70)
    
    correctly_rejected = sum(1 for _, e in results if e.decision == FinalDecision.REJECT)
    false_positives = sum(1 for _, e in results if e.decision == FinalDecision.PASS)
    
    print(f"\nTotal QA mauvais testés: {len(results)}")
    print(f"✅ Correctement rejetés: {correctly_rejected}")
    print(f"❌ Faux positifs (auraient dû être rejetés): {false_positives}")
    print(f"\nTaux de détection: {100*correctly_rejected/len(results):.1f}%")
    
    if false_positives > 0:
        print("\n⚠️  ATTENTION: Le Critic laisse passer des QA de mauvaise qualité!")
        print("   Il faut renforcer les critères d'évaluation.")
    
    # Détails par type de problème
    print("\n" + "-" * 70)
    print("DÉTAIL PAR TYPE DE PROBLÈME")
    print("-" * 70)
    
    for label, evaluation in results:
        status = "✅ REJETÉ" if evaluation.decision == FinalDecision.REJECT else "❌ ACCEPTÉ (ERREUR)"
        print(f"\n{label}:")
        print(f"   {status} - Score: {evaluation.overall_score:.2f}")
        for crit, eval in evaluation.criteria_evaluations.items():
            icon = "✓" if eval.score >= 0.7 else "✗"
            print(f"   {icon} {crit}: {eval.score:.2f}")
    
    # Sauvegarder
    output = {
        "test_type": "bad_qa_detection",
        "total_tested": len(results),
        "correctly_rejected": correctly_rejected,
        "false_positives": false_positives,
        "detection_rate": correctly_rejected / len(results),
        "details": [
            {
                "problem": label,
                "question": qa.question,
                "answer": qa.answer,
                "decision": evaluation.decision.value,
                "scores": {c: e.score for c, e in evaluation.criteria_evaluations.items()}
            }
            for (label, evaluation), qa in zip(results, bad_qa_pairs)
        ]
    }
    
    with open("test_bad_qa_detection.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 Résultats sauvegardés: test_bad_qa_detection.json")
    
    print("\n" + "=" * 70)
    print("TEST TERMINÉ")
    print("=" * 70)


if __name__ == "__main__":
    test_critic_with_bad_qa()
