"""
Analyseur de Logs Détaillés du Critic
======================================

Lit critic_detailed_log.json et affiche:
- Tous les QA avec leur score complet
- Détails de chaque critère (5 critères)
- Raisons exactes des retries
- Comparaison avant/après retry
"""

import json
import sys

# Fix encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# Load log
log_file = "critic_detailed_log.json"

try:
    with open(log_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"ERREUR: Fichier '{log_file}' introuvable.")
    print("Lancez d'abord: python test_pipeline_detailed_logging.py")
    sys.exit(1)

print("=" * 120)
print("ANALYSE DÉTAILLÉE DES ÉVALUATIONS DU CRITIC")
print("=" * 120)
print()

metadata = data['metadata']
evaluations = data['evaluations']
final_dataset = data['final_dataset']

# Stats
print("STATISTIQUES GLOBALES")
print("-" * 120)
stats = metadata['stats']
config = metadata['config']

print(f"Date: {metadata['date']}")
print(f"Configuration: {config['chunks']} chunks, {config['questions_per_chunk']} Q/chunk, {config['max_retries']} max retries")
print(f"Modèles: Generator={config['generator']}, Critic={config['critic']}")
print(f"Threshold: {config['threshold']}")
print()
print(f"Total évaluations: {stats['total_evaluations']}")
print(f"Retries déclenchés: {stats['total_retries']}")
print(f"Succès après retry: {stats['passed_after_retry']}")
print(f"Acceptés: {stats['passed']}")
print(f"Rejetés: {stats['rejected']}")
print()

# Analyser chaque évaluation
print("=" * 120)
print("DÉTAIL DE CHAQUE ÉVALUATION")
print("=" * 120)
print()

passed_evals = [e for e in evaluations if e['decision'] == 'pass']
rejected_evals = [e for e in evaluations if e['decision'] == 'reject']

print(f"Total: {len(evaluations)} évaluations")
print(f"  - PASS: {len(passed_evals)}")
print(f"  - REJECT: {len(rejected_evals)}")
print()

# Grouper par question (même question peut avoir plusieurs évaluations si retry)
from collections import defaultdict
qa_evaluations = defaultdict(list)

for eval in evaluations:
    question = eval['question']
    qa_evaluations[question].append(eval)

print(f"Nombre de QA pairs uniques: {len(qa_evaluations)}")
print()

# Afficher chaque QA avec ses évaluations
for i, (question, evals) in enumerate(qa_evaluations.items(), 1):
    print()
    print("=" * 120)
    print(f"QA PAIR #{i}")
    print("=" * 120)
    print()
    
    print("QUESTION:")
    print("-" * 120)
    print(question)
    print()
    
    # Si plusieurs évaluations → retry
    if len(evals) > 1:
        print(f"⚠️  {len(evals)} évaluations (RETRY LOOP détecté)")
        print()
    
    # Afficher chaque évaluation
    for eval_num, eval in enumerate(evals, 1):
        if len(evals) > 1:
            print(f"--- ÉVALUATION {eval_num}/{len(evals)} ---")
            print()
        
        print(f"Réponse: {eval['answer'][:200]}...")
        print()
        
        print(f"DÉCISION: {eval['decision'].upper()}")
        print(f"Score global: {eval['overall_score']:.3f}")
        print()
        
        # Détails des 5 critères
        print("DÉTAILS DES 5 CRITÈRES:")
        print("-" * 120)
        
        criteria_details = eval['criteria_details']
        
        for criterion_name in ['anchoring', 'local_answerability', 'factual_accuracy', 'completeness', 'clarity']:
            if criterion_name in criteria_details:
                crit = criteria_details[criterion_name]
                result = crit['result']
                score = crit['score']
                explanation = crit['explanation']
                
                status = "✅" if result == 'pass' else "❌"
                
                print(f"{status} {criterion_name.upper()}: {score:.2f} ({result})")
                print(f"   → {explanation}")
                print()
        
        # Si rejet, afficher les raisons
        if eval['decision'] == 'reject':
            print("❌ RAISONS DU REJET:")
            print("-" * 120)
            
            if eval['failed_criteria']:
                print("Critères échoués:")
                for crit in eval['failed_criteria']:
                    print(f"  • {crit}")
            
            if eval['rejection_reasons']:
                print()
                print("Explications:")
                for reason in eval['rejection_reasons']:
                    print(f"  - {reason}")
            print()
        
        if len(evals) > 1 and eval_num < len(evals):
            print()
            print("🔄 → RETRY DÉCLENCHÉ → Régénération Q+A...")
            print()

print()
print("=" * 120)
print("ANALYSE DES SCORES PARFAITS")
print("=" * 120)
print()

perfect_scores = [e for e in evaluations if e['overall_score'] == 1.0]
near_perfect = [e for e in evaluations if 0.95 <= e['overall_score'] < 1.0]
good_scores = [e for e in evaluations if 0.90 <= e['overall_score'] < 0.95]
below_threshold = [e for e in evaluations if e['overall_score'] < 0.90]

print(f"Score = 1.00 (parfait):     {len(perfect_scores)}/{len(evaluations)} ({len(perfect_scores)/len(evaluations)*100:.1f}%)")
print(f"Score 0.95-0.99:            {len(near_perfect)}/{len(evaluations)} ({len(near_perfect)/len(evaluations)*100:.1f}%)")
print(f"Score 0.90-0.94:            {len(good_scores)}/{len(evaluations)} ({len(good_scores)/len(evaluations)*100:.1f}%)")
print(f"Score < 0.90 (sous seuil):  {len(below_threshold)}/{len(evaluations)} ({len(below_threshold)/len(evaluations)*100:.1f}%)")
print()

if len(perfect_scores) / len(evaluations) > 0.5:
    print("⚠️  ATTENTION: Plus de 50% de scores parfaits!")
    print()
    print("Cela peut signifier:")
    print("  1. Les QA générés sont vraiment excellents (peu probable)")
    print("  2. Le Critic (Phi-3 Mini) est trop laxiste")
    print("  3. Les prompts du Critic ne sont pas assez stricts")
    print()
    print("Solutions:")
    print("  - Augmenter threshold: 0.90 → 0.95")
    print("  - Ajouter des pénalités automatiques dans le scoring")
    print("  - Utiliser un modèle Critic plus puissant")

print()
print("=" * 120)
print("ANALYSE DES CRITÈRES ÉCHOUÉS")
print("=" * 120)
print()

# Compter les échecs par critère
criterion_failures = defaultdict(int)
for eval in rejected_evals:
    for crit in eval['failed_criteria']:
        criterion_failures[crit] += 1

if criterion_failures:
    print("Critères les plus souvent échoués:")
    print()
    for criterion, count in sorted(criterion_failures.items(), key=lambda x: -x[1]):
        pct = (count / len(rejected_evals)) * 100 if rejected_evals else 0
        print(f"  {criterion:25s}: {count:2d} rejets ({pct:5.1f}% des rejets)")
else:
    print("Aucun critère échoué (aucun rejet)")

print()
print("=" * 120)
print("DATASET FINAL")
print("=" * 120)
print()

print(f"{len(final_dataset)} QA pairs dans le dataset final:")
print()

for i, entry in enumerate(final_dataset, 1):
    print(f"{i}. Score {entry['score']:.3f} - {entry['question'][:80]}...")

print()
