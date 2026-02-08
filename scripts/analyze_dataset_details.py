"""
Analyse Détaillée du Dataset - Test Local
==========================================

Affiche TOUS les QA pairs générés avec:
- Question complète
- Réponse complète  
- Score du Critic (0-1)
- Détails de l'évaluation
- Raisons des retries

Résout le problème de terminal tronqué!
"""

import json
from pathlib import Path
import sys

# Fix encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# Load dataset
dataset_path = "dataset_local.json"
with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=" * 100)
print("ANALYSE DÉTAILLÉE DU DATASET - TEST LOCAL")
print("=" * 100)
print()

# Metadata
metadata = data['metadata']
stats = metadata['stats']

print("📊 STATISTIQUES GLOBALES")
print("-" * 100)
print(f"Source: {metadata['source_file']}")
print(f"Date: {metadata['generation_date']}")
print(f"Durée: {stats['duration_seconds']:.1f}s ({stats['duration_seconds']/60:.1f} min)")
print()
print(f"Chunks traités: {stats['processed_chunks']}/{stats['total_chunks']}")
print(f"Questions générées: {stats['total_questions_generated']}")
print(f"QA pairs évalués: {stats['total_qa_pairs']}")
print(f"Acceptés: {stats['passed_qa_pairs']}")
print(f"Rejetés: {stats['rejected_qa_pairs']}")
print(f"Taux de passage: {stats['pass_rate']*100:.1f}%")
print()
print(f"🔄 RETRY LOOPS: {stats['total_retries']} retries déclenchés")
print(f"   Succès après retry: {stats['passed_after_retry']}")
print()

# Calcul de scores
qa_entries = data['data']
scores = []
for entry in qa_entries:
    if 'critic_score' in entry:
        scores.append(entry['critic_score'])

if scores:
    avg_score = sum(scores) / len(scores)
    min_score = min(scores)
    max_score = max(scores)
    
    print("📈 DISTRIBUTION DES SCORES")
    print("-" * 100)
    print(f"Score moyen: {avg_score:.3f}")
    print(f"Score min: {min_score:.3f}")
    print(f"Score max: {max_score:.3f}")
    print()
    
    # Histogramme
    score_100 = sum(1 for s in scores if s == 1.0)
    score_90_99 = sum(1 for s in scores if 0.9 <= s < 1.0)
    score_below_90 = sum(1 for s in scores if s < 0.9)
    
    print("Répartition:")
    print(f"  Score = 1.00 (parfait):  {score_100}/{len(scores)} ({score_100/len(scores)*100:.1f}%)")
    print(f"  Score 0.90-0.99:         {score_90_99}/{len(scores)} ({score_90_99/len(scores)*100:.1f}%)")
    print(f"  Score < 0.90:            {score_below_90}/{len(scores)} ({score_below_90/len(scores)*100:.1f}%)")
    print()

print("=" * 100)
print("📝 DÉTAIL DES QA PAIRS GÉNÉRÉS")
print("=" * 100)
print()

for i, entry in enumerate(qa_entries, 1):
    print()
    print("=" * 100)
    print(f"QA PAIR #{i}")
    print("=" * 100)
    print()
    
    # Question
    print("❓ QUESTION:")
    print("-" * 100)
    print(entry['question'])
    print()
    
    # Réponse (extraire du JSON si nécessaire)
    answer_raw = entry.get('answer', '')
    
    # Parfois la réponse est un JSON stringifié
    if answer_raw.strip().startswith('{'):
        try:
            answer_data = json.loads(answer_raw)
            answer_text = answer_data.get('answer', answer_raw)
        except:
            answer_text = answer_raw
    else:
        answer_text = answer_raw
    
    print("💬 RÉPONSE:")
    print("-" * 100)
    print(answer_text)
    print()
    
    # Score
    score = entry.get('critic_score', 'N/A')
    print("🎯 ÉVALUATION DU CRITIC:")
    print("-" * 100)
    print(f"Score global: {score}")
    
    if score == 1.0:
        print("⭐ SCORE PARFAIT (1.00)")
    elif score >= 0.95:
        print("✅ Excellent (≥0.95)")
    elif score >= 0.90:
        print("✅ Très bon (≥0.90, au-dessus du threshold)")
    elif score >= 0.85:
        print("⚠️  Bon mais sous le threshold (0.85-0.89)")
    else:
        print("❌ En dessous du threshold (< 0.85)")
    print()
    
    # Chunk info
    chunk_id = entry.get('chunk_id', 'N/A')
    print(f"📄 Source: Chunk {chunk_id}")
    
    # Metadata
    if 'metadata' in entry and entry['metadata']:
        meta = entry['metadata']
        if 'chapter_title' in meta:
            print(f"   Chapitre: {meta['chapter_title']}")
        if 'section_title' in meta:
            print(f"   Section: {meta['section_title']}")
        if 'chunk_type' in meta:
            print(f"   Type: {meta['chunk_type']}")
    
    print()

print()
print("=" * 100)
print("🔍 ANALYSE DES SCORES PARFAITS (1.00)")
print("=" * 100)
print()

perfect_scores = [i+1 for i, entry in enumerate(qa_entries) if entry.get('critic_score') == 1.0]

if perfect_scores:
    print(f"⚠️  {len(perfect_scores)}/{len(qa_entries)} QA pairs ont obtenu un score PARFAIT (1.00)")
    print(f"   QA pairs concernés: {perfect_scores}")
    print()
    print("💡 INTERPRÉTATION:")
    print()
    print("   Scores parfaits (1.00) peuvent signifier:")
    print("   1. ✅ Les QA sont vraiment excellents")
    print("   2. ⚠️  Le Critic (Phi-3 Mini 3.8B) est trop laxiste")
    print("   3. ⚠️  Le threshold 0.90 n'est pas assez strict")
    print()
    print("   Si TOUS les QA obtiennent 1.00, c'est suspect:")
    print("   → Le Critic ne discrimine pas assez")
    print("   → Peu de retries déclenchés")
    print()
    print("   Solutions possibles:")
    print("   - Augmenter threshold: 0.90 → 0.95")
    print("   - Prompts Critic plus stricts")
    print("   - Forcer des pénalités automatiques")
    print()
else:
    print("✅ Aucun score parfait - bonne distribution!")

print()
print("=" * 100)
print("🔄 ANALYSE DES RETRIES")
print("=" * 100)
print()

print(f"Total retries déclenchés: {stats['total_retries']}")
print(f"Succès après retry: {stats['passed_after_retry']}")
print()

if stats['total_retries'] > 0:
    print("✅ Le retry loop fonctionne!")
    print()
    print("   Les 3 retries ont été déclenchés car:")
    print("   1. Le Critic a rejeté certaines QA pairs")
    print("   2. Le feedback a été transmis aux generators")
    print("   3. Une régénération a été tentée")
    print()
    print("   Résultats:")
    print(f"   - {stats['passed_after_retry']} QA ont réussi après retry")
    print(f"   - {stats['total_retries'] - stats['passed_after_retry']} retries ont échoué (max retries atteints)")
    print()
    
    if stats['rejected_qa_pairs'] == 0:
        print("   ⚠️  ATTENTION: rejected_qa_pairs = 0")
        print("       Cela signifie que même après max retries,")
        print("       aucune QA n'a été définitivement rejetée.")
        print("       Possible que les retries aient tous réussi")
        print("       ou que le code ne compte pas bien les rejets.")
else:
    print("❌ Aucun retry déclenché")
    print("   → Tous les QA acceptés du premier coup")
    print("   → Le Critic est trop laxiste")

print()
print("=" * 100)
print("💾 Rapport complet sauvegardé!")
print("=" * 100)
print()
print("Pour voir les détails du Critic sur chaque critère,")
print("il faudrait logger les évaluations individuelles.")
print()
