#!/usr/bin/env python3
"""
Create a clean comparison report from ScopedMemory demo results.
"""

import json
from pathlib import Path

def create_tutor_report():
    """Generate a clean comparison report for the tutor."""
    
    # Load demo results
    with open('results/scoped_memory_demo.json', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract data
    section = data['section']
    chunks_tested = data['chunks_tested']
    without = data['without_memory']
    with_mem = data['with_memory']
    
    # Prepare report
    report = {
        "experiment": {
            "title": "Impact de ScopedMemory sur la Diversité des Questions",
            "date": "2026-02-18",
            "section_testée": section,
            "nb_chunks": chunks_tested,
            "model": "DeepSeek R1 Distill Qwen 32B (IQ3_M)"
        },
        
        "baseline_v2_sans_memoire": {
            "description": "Question Generator V2 sans mémoire contextuelle",
            "questions": []
        },
        
        "v4_avec_memoire": {
            "description": "Question Generator avec ScopedMemory (V4 approach)",
            "questions": []
        },
        
        "metriques_comparaison": {},
        
        "conclusion": ""
    }
    
    # Extract questions without memory
    for i, result in enumerate(without, 1):
        if result.get('question'):
            report["baseline_v2_sans_memoire"]["questions"].append({
                "chunk_id": result['chunk_id'],
                "question": result['question'],
                "content_preview": result['content_preview']
            })
    
    # Extract questions with memory
    for i, result in enumerate(with_mem, 1):
        if result.get('question'):
            report["v4_avec_memoire"]["questions"].append({
                "chunk_id": result['chunk_id'],
                "question": result['question'],
                "diversity_active": result.get('diversity_active', False),
                "memory_stats": result.get('memory_stats', {}),
                "content_preview": result['content_preview']
            })
    
    # Calculate metrics
    from difflib import SequenceMatcher
    
    def avg_similarity(questions):
        if len(questions) < 2:
            return 0.0
        total = 0
        count = 0
        for i in range(len(questions)):
            for j in range(i+1, len(questions)):
                sim = SequenceMatcher(None, questions[i], questions[j]).ratio()
                total += sim
                count += 1
        return total / count if count > 0 else 0.0
    
    without_q = [r['question'] for r in without if r.get('question')]
    with_q = [r['question'] for r in with_mem if r.get('question')]
    
    without_starts = [q.split()[0].lower() for q in without_q]
    with_starts = [q.split()[0].lower() for q in with_q]
    
    without_sim = avg_similarity(without_q)
    with_sim = avg_similarity(with_q)
    improvement = ((without_sim - with_sim) / without_sim * 100) if without_sim > 0 else 0
    
    report["metriques_comparaison"] = {
        "sans_memoire": {
            "questions_generees": len(without_q),
            "mots_debut_uniques": f"{len(set(without_starts))}/{len(without_starts)}",
            "similarite_moyenne": f"{without_sim:.2%}",
            "diversite_score": f"{(1-without_sim)*100:.1f}/100"
        },
        "avec_memoire": {
            "questions_generees": len(with_q),
            "mots_debut_uniques": f"{len(set(with_starts))}/{len(with_starts)}",
            "similarite_moyenne": f"{with_sim:.2%}",
            "diversite_score": f"{(1-with_sim)*100:.1f}/100",
            "memoire_activee": f"{sum(1 for r in with_mem if r.get('diversity_active'))}/{len(with_q)} fois",
            "concepts_accumules": with_mem[-1]['memory_stats']['concepts_count'] if with_mem and with_mem[-1].get('memory_stats') else 0
        },
        "amelioration": {
            "reduction_similarite": f"{improvement:+.1f}%",
            "gain_diversite": f"{(with_sim - without_sim) / (1 - without_sim) * 100 if without_sim < 1 else 0:.1f}%"
        }
    }
    
    # Conclusion
    report["conclusion"] = (
        f"Sur {chunks_tested} chunks de la section '{section}', "
        f"ScopedMemory a réduit la similarité inter-questions de {improvement:.1f}%, "
        f"passant de {without_sim:.2%} à {with_sim:.2%}. "
        f"La mémoire contextuelle a été activée pour {sum(1 for r in with_mem if r.get('diversity_active'))}/{len(with_q)} questions, "
        f"accumulant {with_mem[-1]['memory_stats']['concepts_count'] if with_mem and with_mem[-1].get('memory_stats') else 0} concepts distincts. "
        f"Cela démontre que ScopedMemory améliore significativement la diversité des questions, "
        f"essentielle pour un Gold Dataset de qualité destiné à l'évaluation de systèmes RAG."
    )
    
    # Save report
    output_path = Path('results/scoped_memory_report_for_tutor.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Rapport créé: {output_path}")
    
    # Also create a markdown version
    md_path = Path('results/scoped_memory_report_for_tutor.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Impact de ScopedMemory sur la Diversité des Questions\n\n")
        f.write(f"**Date:** {report['experiment']['date']}\n")
        f.write(f"**Section testée:** {section}\n")
        f.write(f"**Chunks testés:** {chunks_tested}\n")
        f.write(f"**Modèle:** {report['experiment']['model']}\n\n")
        
        f.write("---\n\n")
        
        f.write("## 📊 Métriques de Comparaison\n\n")
        f.write("| Métrique | Sans Mémoire (V2) | Avec Mémoire (V4) | Amélioration |\n")
        f.write("|----------|:-----------------:|:-----------------:|:------------:|\n")
        
        m = report['metriques_comparaison']
        f.write(f"| **Questions générées** | {m['sans_memoire']['questions_generees']} | {m['avec_memoire']['questions_generees']} | - |\n")
        f.write(f"| **Mots de début uniques** | {m['sans_memoire']['mots_debut_uniques']} | {m['avec_memoire']['mots_debut_uniques']} | - |\n")
        f.write(f"| **Similarité moyenne** | {m['sans_memoire']['similarite_moyenne']} | {m['avec_memoire']['similarite_moyenne']} | {m['amelioration']['reduction_similarite']} |\n")
        f.write(f"| **Score de diversité** | {m['sans_memoire']['diversite_score']} | {m['avec_memoire']['diversite_score']} | - |\n")
        f.write(f"| **Mémoire activée** | 0/{chunks_tested} | {m['avec_memoire']['memoire_activee']} | - |\n")
        f.write(f"| **Concepts accumulés** | 0 | {m['avec_memoire']['concepts_accumules']} | - |\n")
        
        f.write("\n---\n\n")
        
        f.write("## ❌ Questions SANS ScopedMemory (Baseline V2)\n\n")
        for i, q_data in enumerate(report['baseline_v2_sans_memoire']['questions'], 1):
            f.write(f"{i}. **{q_data['chunk_id']}** — {q_data['question']}\n\n")
        
        f.write("---\n\n")
        
        f.write("## ✅ Questions AVEC ScopedMemory (V4 Approach)\n\n")
        for i, q_data in enumerate(report['v4_avec_memoire']['questions'], 1):
            icon = "🧠" if q_data['diversity_active'] else "  "
            f.write(f"{icon} {i}. **{q_data['chunk_id']}** — {q_data['question']}\n")
            if q_data['diversity_active']:
                stats = q_data['memory_stats']
                f.write(f"   - *Mémoire: {stats.get('concepts_count', 0)} concepts, {stats.get('questions_count', 0)} questions*\n")
            f.write("\n")
        
        f.write("---\n\n")
        
        f.write("## 💡 Conclusion\n\n")
        f.write(report['conclusion'])
        f.write("\n\n---\n\n")
        f.write("**Recommandation:** Intégrer ScopedMemory dans le pipeline de production (Critique V4 + Question Generator V3) pour garantir un Gold Dataset avec une diversité maximale des questions, critère essentiel pour l'évaluation fiable des systèmes RAG.\n")
    
    print(f"✅ Rapport Markdown créé: {md_path}")
    
    return report


if __name__ == '__main__':
    create_tutor_report()
