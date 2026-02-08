"""
Test Critic Agent avec QA BORDERLINE / CAS LIMITES
===================================================

Vérifie que le Critic produit des scores variés (pas juste 0% ou 100%)
et discrimine les cas subtils.
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


def create_borderline_qa_pairs(chunk):
    """
    Créer des QA pairs BORDERLINE - ni parfaits ni catastrophiques.
    Ces cas doivent produire des scores variés (0.4-0.8).
    """
    
    borderline_pairs = []
    
    # Le chunk parle de : intersection de tribus, sous-tribus, tribu engendrée
    # Contenu clé: "Une intersection de tribus est une tribu"
    #              "une réunion de tribus n'est pas une tribu"
    #              "Une sous-tribu de F est une tribu G telle que G ⊂ F"
    
    # 1. LÉGÈREMENT IMPRÉCIS - Réponse correcte mais approximative
    borderline_pairs.append({
        "label": "LÉGÈREMENT IMPRÉCIS - Bonne idée, formulation floue",
        "expected_issues": ["completeness ou clarity faible"],
        "qa": QAPair(
            question="Qu'est-ce qu'une sous-tribu ?",
            answer="Une sous-tribu c'est quand on a une tribu qui est contenue dans une autre tribu plus grande.",
            question_type="factual",
            difficulty="easy",
            supporting_quotes=["Une sous-tribu de F est une tribu G telle que G ⊂F"],
            chunk_id=chunk.chunk_id,
            source_file="test",
            page_range=chunk.page_range,
            chapter=chunk.chapter_title,
            section=chunk.section_title,
            confidence=0.7
        )
    })
    
    # 2. PARTIELLEMENT ANCRÉ - Citation présente mais réponse ajoute un peu
    borderline_pairs.append({
        "label": "PARTIELLEMENT ANCRÉ - Réponse va légèrement au-delà",
        "expected_issues": ["anchoring moyen"],
        "qa": QAPair(
            question="Pourquoi une intersection de tribus est-elle une tribu ?",
            answer="Une intersection de tribus est une tribu car elle hérite des propriétés de fermeture par complémentation et union dénombrable de chaque tribu composante. C'est une conséquence directe de la définition axiomatique.",
            question_type="conceptual",
            difficulty="medium",
            supporting_quotes=["Une intersection de tribus est une tribu"],
            chunk_id=chunk.chunk_id,
            source_file="test",
            page_range=chunk.page_range,
            chapter=chunk.chapter_title,
            section=chunk.section_title,
            confidence=0.75
        )
    })
    
    # 3. QUESTION LÉGITIME, RÉPONSE TROP COURTE
    borderline_pairs.append({
        "label": "RÉPONSE TROP SUCCINCTE - Correcte mais manque de détail",
        "expected_issues": ["completeness faible"],
        "qa": QAPair(
            question="Quelle est la différence entre une intersection et une réunion de tribus ?",
            answer="L'intersection de tribus est une tribu, mais pas la réunion.",
            question_type="comparative",
            difficulty="medium",
            supporting_quotes=[
                "Une intersection de tribus est une tribu",
                "une réunion de tribus n'est pas une tribu"
            ],
            chunk_id=chunk.chunk_id,
            source_file="test",
            page_range=chunk.page_range,
            chapter=chunk.chapter_title,
            section=chunk.section_title,
            confidence=0.8
        )
    })
    
    # 4. FORMULATION MALADROITE - Correct sur le fond, confus dans la forme
    borderline_pairs.append({
        "label": "FORMULATION CONFUSE - Fond correct, forme maladroite",
        "expected_issues": ["clarity faible"],
        "qa": QAPair(
            question="Comment définit-on la plus petite tribu contenant une famille d'ensembles ?",
            answer="Eh bien, pour avoir la plus petite tribu, on prend toutes les tribus qui contiennent cette famille, et puis on fait l'intersection de toutes ces tribus-là, et ça donne la plus petite tribu parce que l'intersection de tribus c'est une tribu.",
            question_type="procedural",
            difficulty="medium",
            supporting_quotes=["La plus petite tribu contenant une famille d'ensembles est l'intersection de toutes les tribus"],
            chunk_id=chunk.chunk_id,
            source_file="test",
            page_range=chunk.page_range,
            chapter=chunk.chapter_title,
            section=chunk.section_title,
            confidence=0.7
        )
    })
    
    # 5. LÉGÈRE EXTRAPOLATION - Part du chunk mais extrapole un peu
    borderline_pairs.append({
        "label": "LÉGÈRE EXTRAPOLATION - Déduit quelque chose non explicite",
        "expected_issues": ["local_answerability ou anchoring moyen"],
        "qa": QAPair(
            question="Que peut-on déduire sur les opérations ensemblistes appliquées aux tribus ?",
            answer="On peut déduire que les tribus sont stables par intersection mais pas par réunion. Cela suggère que l'intersection préserve les propriétés de fermeture tandis que la réunion peut les détruire.",
            question_type="causal",
            difficulty="hard",
            supporting_quotes=[
                "Une intersection de tribus est une tribu",
                "une réunion de tribus n'est pas une tribu"
            ],
            chunk_id=chunk.chunk_id,
            source_file="test",
            page_range=chunk.page_range,
            chapter=chunk.chapter_title,
            section=chunk.section_title,
            confidence=0.65
        )
    })
    
    # 6. CITATION DÉCONTEXTUALISÉE - Utilise une citation hors de son sens
    borderline_pairs.append({
        "label": "CITATION MAL UTILISÉE - Citation correcte, interprétation douteuse",
        "expected_issues": ["factual_accuracy moyen"],
        "qa": QAPair(
            question="Pourquoi dit-on qu'une sous-tribu est 'plus petite' ?",
            answer="Une sous-tribu G de F est dite plus petite car elle est contenue dans F. Donc tous les éléments de G sont aussi dans F, ce qui fait que G a moins d'éléments que F en général.",
            question_type="conceptual",
            difficulty="easy",
            supporting_quotes=["Une sous-tribu de F est une tribu G telle que G ⊂F"],
            chunk_id=chunk.chunk_id,
            source_file="test",
            page_range=chunk.page_range,
            chapter=chunk.chapter_title,
            section=chunk.section_title,
            confidence=0.6
        )
    })
    
    # 7. QUESTION AMBIGUË - Peut être interprétée de plusieurs façons
    borderline_pairs.append({
        "label": "QUESTION AMBIGUË - Interprétation discutable",
        "expected_issues": ["local_answerability ou clarity"],
        "qa": QAPair(
            question="Qu'est-ce qui se passe avec les tribus quand on les combine ?",
            answer="Quand on combine des tribus, le résultat dépend de l'opération : une intersection donne toujours une tribu, mais une réunion ne donne pas forcément une tribu.",
            question_type="conceptual",
            difficulty="easy",
            supporting_quotes=[
                "Une intersection de tribus est une tribu",
                "une réunion de tribus n'est pas une tribu"
            ],
            chunk_id=chunk.chunk_id,
            source_file="test",
            page_range=chunk.page_range,
            chapter=chunk.chapter_title,
            section=chunk.section_title,
            confidence=0.7
        )
    })
    
    # 8. RÉPONSE VRAIE MAIS TRIVIALE
    borderline_pairs.append({
        "label": "TROP TRIVIAL - Réponse correcte mais n'apporte rien",
        "expected_issues": ["completeness très faible"],
        "qa": QAPair(
            question="Qu'est-ce qu'une tribu engendré par une famille d'ensembles ?",
            answer="C'est la plus petite tribu qui contient cette famille.",
            question_type="factual",
            difficulty="easy",
            supporting_quotes=["La plus petite tribu contenant une famille d'ensembles"],
            chunk_id=chunk.chunk_id,
            source_file="test",
            page_range=chunk.page_range,
            chapter=chunk.chapter_title,
            section=chunk.section_title,
            confidence=0.8
        )
    })
    
    # 9. AJOUT D'EXEMPLE NON PRÉSENT
    borderline_pairs.append({
        "label": "EXEMPLE INVENTÉ - Concept correct, exemple non dans le chunk",
        "expected_issues": ["anchoring faible"],
        "qa": QAPair(
            question="Comment fonctionne une intersection de tribus ?",
            answer="Une intersection de tribus est une tribu. Par exemple, si on prend la tribu des boréliens et la tribu de Lebesgue sur R, leur intersection est aussi une tribu.",
            question_type="application",
            difficulty="medium",
            supporting_quotes=["Une intersection de tribus est une tribu"],
            chunk_id=chunk.chunk_id,
            source_file="test",
            page_range=chunk.page_range,
            chapter=chunk.chapter_title,
            section=chunk.section_title,
            confidence=0.6
        )
    })
    
    # 10. TERMINOLOGIE LÉGÈREMENT INCORRECTE
    borderline_pairs.append({
        "label": "TERMINOLOGIE APPROXIMATIVE - Sens correct, mots imprécis",
        "expected_issues": ["factual_accuracy ou clarity"],
        "qa": QAPair(
            question="Qu'est-ce qui caractérise une sous-tribu ?",
            answer="Une sous-tribu est un sous-ensemble d'une tribu qui garde les mêmes propriétés de tribu. C'est quand une tribu G est incluse dans une tribu F.",
            question_type="conceptual",
            difficulty="easy",
            supporting_quotes=["Une sous-tribu de F est une tribu G telle que G ⊂F"],
            chunk_id=chunk.chunk_id,
            source_file="test",
            page_range=chunk.page_range,
            chapter=chunk.chapter_title,
            section=chunk.section_title,
            confidence=0.75
        )
    })
    
    return borderline_pairs


def test_borderline_qa():
    """Tester le Critic avec des QA borderline."""
    
    print("=" * 70)
    print("TEST: CRITIC AVEC QA BORDERLINE / CAS LIMITES")
    print("=" * 70)
    print()
    print("Ce test vérifie que le Critic produit des scores VARIÉS (pas 0 ou 1)")
    print("et discrimine les cas subtils avec des scores comme 3/5, 4/5, etc.")
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
    
    print(f"📄 Chunk de test: {test_chunk.chunk_id}")
    print(f"📝 Contenu ({len(test_chunk.content)} chars):")
    print("-" * 50)
    print(test_chunk.content[:600])
    print("-" * 50)
    
    # Créer les QA borderline
    borderline_items = create_borderline_qa_pairs(test_chunk)
    
    # Initialiser le Critic
    critic = CriticAgent(
        llm_client=client,
        model_name="llama-3.3-70b-versatile",
        language="fr",
        temperature=0.2,
        strict_mode=True
    )
    
    print(f"\n{'='*70}")
    print(f"ÉVALUATION DE {len(borderline_items)} QA BORDERLINE")
    print("="*70)
    
    results = []
    score_distribution = {"0-0.3": 0, "0.3-0.5": 0, "0.5-0.7": 0, "0.7-0.9": 0, "0.9-1.0": 0}
    criteria_scores = {c: [] for c in ["anchoring", "local_answerability", "factual_accuracy", "completeness", "clarity"]}
    
    for i, item in enumerate(borderline_items, 1):
        qa = item["qa"]
        label = item["label"]
        expected = item["expected_issues"]
        
        print(f"\n{'─'*70}")
        print(f"TEST #{i}: {label}")
        print(f"Problèmes attendus: {expected}")
        print(f"{'─'*70}")
        print(f"Q: {qa.question}")
        print(f"R: {qa.answer[:120]}..." if len(qa.answer) > 120 else f"R: {qa.answer}")
        
        evaluation = critic.evaluate(qa, test_chunk)
        results.append((item, evaluation))
        
        # Classifier le score
        score = evaluation.overall_score
        if score < 0.3:
            score_distribution["0-0.3"] += 1
        elif score < 0.5:
            score_distribution["0.3-0.5"] += 1
        elif score < 0.7:
            score_distribution["0.5-0.7"] += 1
        elif score < 0.9:
            score_distribution["0.7-0.9"] += 1
        else:
            score_distribution["0.9-1.0"] += 1
        
        # Collecter les scores par critère
        for crit, ev in evaluation.criteria_evaluations.items():
            criteria_scores[crit].append(ev.score)
        
        # Afficher résultat
        decision_icon = "✅ PASS" if evaluation.decision == FinalDecision.PASS else "❌ REJECT"
        print(f"\n{decision_icon} | Score: {evaluation.overall_score:.2f}")
        
        # Afficher critères avec codes couleur
        passed = 0
        failed = 0
        for crit, ev in evaluation.criteria_evaluations.items():
            if ev.score >= 0.7:
                icon = "✓"
                passed += 1
            else:
                icon = "✗"
                failed += 1
            print(f"   {icon} {crit}: {ev.score:.2f}")
        
        print(f"   → Critères: {passed}/5 passés, {failed}/5 échoués")
        
        if evaluation.rejection_reasons:
            print(f"   Raisons: {evaluation.rejection_reasons[:2]}")
    
    # ================== ANALYSE DE LA VARIABILITÉ ==================
    print("\n" + "=" * 70)
    print("ANALYSE DE LA VARIABILITÉ DES SCORES")
    print("=" * 70)
    
    # Distribution des scores globaux
    print("\n📊 DISTRIBUTION DES SCORES GLOBAUX:")
    print("-" * 40)
    total = len(results)
    for bucket, count in score_distribution.items():
        bar = "█" * (count * 3) if count > 0 else ""
        pct = 100 * count / total
        print(f"   {bucket}: {bar} {count} ({pct:.0f}%)")
    
    # Variété des scores par critère
    print("\n📈 SCORES PAR CRITÈRE (min - moy - max):")
    print("-" * 40)
    for crit, scores in criteria_scores.items():
        min_s = min(scores)
        max_s = max(scores)
        avg_s = sum(scores) / len(scores)
        variance = sum((s - avg_s)**2 for s in scores) / len(scores)
        
        # Indicateur de variété
        variety = "🔴 AUCUNE" if variance < 0.01 else "🟡 FAIBLE" if variance < 0.05 else "🟢 BONNE" if variance < 0.1 else "🟢 EXCELLENTE"
        
        print(f"   {crit:22}: {min_s:.2f} - {avg_s:.2f} - {max_s:.2f} (var={variance:.3f}) {variety}")
    
    # Scores uniques observés
    all_scores = [e.overall_score for _, e in results]
    unique_scores = len(set(f"{s:.2f}" for s in all_scores))
    
    print(f"\n🎯 SCORES UNIQUES OBSERVÉS: {unique_scores}/{len(results)}")
    if unique_scores == 1:
        print("   ⚠️  PROBLÈME: Tous les scores sont identiques!")
    elif unique_scores < 3:
        print("   ⚠️  PROBLÈME: Très peu de variété dans les scores")
    elif unique_scores < 5:
        print("   🟡 OK: Variété modérée")
    else:
        print("   ✅ BIEN: Bonne variété de scores")
    
    # Résumé Pass/Reject
    print("\n" + "-" * 70)
    print("RÉSUMÉ DES DÉCISIONS")
    print("-" * 70)
    
    passed_count = sum(1 for _, e in results if e.decision == FinalDecision.PASS)
    rejected_count = sum(1 for _, e in results if e.decision == FinalDecision.REJECT)
    
    print(f"   ✅ PASS: {passed_count} ({100*passed_count/total:.0f}%)")
    print(f"   ❌ REJECT: {rejected_count} ({100*rejected_count/total:.0f}%)")
    
    # Tableau récapitulatif
    print("\n" + "=" * 70)
    print("TABLEAU RÉCAPITULATIF")
    print("=" * 70)
    print(f"{'#':<3} {'Label':<45} {'Score':<6} {'Critères':<10} {'Décision':<8}")
    print("-" * 70)
    
    for i, (item, evaluation) in enumerate(results, 1):
        label = item["label"][:42] + "..." if len(item["label"]) > 45 else item["label"]
        score = f"{evaluation.overall_score:.2f}"
        
        passed_criteria = sum(1 for ev in evaluation.criteria_evaluations.values() if ev.score >= 0.7)
        criteria_str = f"{passed_criteria}/5"
        
        decision = "PASS" if evaluation.decision == FinalDecision.PASS else "REJECT"
        
        print(f"{i:<3} {label:<45} {score:<6} {criteria_str:<10} {decision:<8}")
    
    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    # Critères de succès du test
    has_variety = unique_scores >= 4
    has_partial_passes = any(
        0.5 <= e.overall_score < 0.9 
        for _, e in results
    )
    has_mixed_criteria = any(
        1 <= sum(1 for ev in e.criteria_evaluations.values() if ev.score >= 0.7) <= 4
        for _, e in results
    )
    
    if has_variety and has_partial_passes and has_mixed_criteria:
        print("✅ Le Critic produit des scores VARIÉS et discrimine les cas limites!")
        print("   → Prêt pour passer à l'étape suivante.")
    else:
        print("⚠️  PROBLÈMES DÉTECTÉS:")
        if not has_variety:
            print("   - Pas assez de variété dans les scores globaux")
        if not has_partial_passes:
            print("   - Pas de scores intermédiaires (tout est 0 ou 1)")
        if not has_mixed_criteria:
            print("   - Les critères passent tous ou échouent tous ensemble")
        print("\n   → Le Critic pourrait être trop binaire.")
    
    # Sauvegarder
    output = {
        "test_type": "borderline_qa_discrimination",
        "total_tested": len(results),
        "score_distribution": score_distribution,
        "unique_scores": unique_scores,
        "passed": passed_count,
        "rejected": rejected_count,
        "criteria_stats": {
            crit: {
                "min": min(scores),
                "max": max(scores),
                "avg": sum(scores)/len(scores),
                "variance": sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)
            }
            for crit, scores in criteria_scores.items()
        },
        "details": [
            {
                "label": item["label"],
                "expected_issues": item["expected_issues"],
                "question": item["qa"].question,
                "answer": item["qa"].answer,
                "decision": evaluation.decision.value,
                "overall_score": evaluation.overall_score,
                "criteria_scores": {c: e.score for c, e in evaluation.criteria_evaluations.items()},
                "criteria_passed": sum(1 for e in evaluation.criteria_evaluations.values() if e.score >= 0.7)
            }
            for item, evaluation in results
        ]
    }
    
    with open("test_borderline_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 Résultats sauvegardés: test_borderline_results.json")


if __name__ == "__main__":
    test_borderline_qa()
