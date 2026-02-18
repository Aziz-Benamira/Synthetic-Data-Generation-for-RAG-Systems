# Architecture Decision Records (ADR)

Ce fichier documente chaque décision architecturale majeure du projet.
Chaque entrée explique le contexte, la décision prise, les alternatives considérées et le raisonnement.

---

## ADR-001 : Suppression de Answer Accuracy (2026-02-17)

**Contexte :**
Le Critic V3 évalue les réponses avec 4 métriques : Anchoring, Answer Accuracy, Clarity, Completeness.

**Décision :**
Supprimer Answer Accuracy du Critic V4.

**Raisonnement :**
- Answer Accuracy est redondant avec Anchoring + Completeness.
- Preuve empirique sur le batch MI201 (85 QA) :
  - Quand anchoring ✅ ET completeness ✅ → answer_accuracy est TOUJOURS ✅
  - Quand answer_accuracy ❌ → c'est TOUJOURS parce que anchoring OU completeness échoue
- answer_accuracy ≈ f(anchoring, completeness)
- Gain : ~30s/QA (2 judges × ~15s), soit ~42 min sur 85 QA

**Alternatives rejetées :**
- Garder les 4 métriques : Redondant, coûteux en tokens et en temps
- Fusionner answer_accuracy dans anchoring : Changerait la sémantique d'anchoring

---

## ADR-002 : Critic V4 en 2 Phases (2026-02-17)

**Contexte :**
Le Critic V3 évalue uniquement la réponse. Des questions de mauvaise qualité (non-answerables, triviales) passent quand même si la réponse est bien formulée.

**Décision :**
Architecture 2-phases :
- Phase 1 : Évaluation de la Question (avant génération de réponse)
- Phase 2 : Évaluation de la Réponse (3 métriques, feedback loop)

**Raisonnement :**
- Si la question est mauvaise, inutile de générer une réponse (économie de tokens)
- Un Gold Dataset doit avoir des questions de qualité (SQuAD 2.0, Natural Questions font du filtrage)
- Phase 1 filtre ~30% des questions (non-answerables + triviales)

**Impact :**
- Pipeline : Chunk → QGen → Phase 1 → AGen → Phase 2 → Output
- Si Phase 1 rejette : Régénération de la question (pas de la réponse)

---

## ADR-003 : Métriques Phase 1 — Question (2026-02-17)

**Contexte :**
Besoin de 2 métriques maximum pour évaluer la qualité de la question.

**Décision :**
1. **Contextual Answerability** : La question est-elle répondable uniquement avec le chunk ?
   - Le LLM doit extraire les evidence spans (phrases exactes du chunk)
   - Scoring : 0 (aucune info) → 3 (info complète)
   - Seuil : ≥ 2 pour PASS

2. **Pedagogical Value** : La question a-t-elle une valeur d'apprentissage ?
   - 3 critères binaires : tests_understanding, non_trivial, educational_utility
   - Score = nb_yes / 3
   - Seuil : ≥ 0.67 pour PASS

**Alternatives rejetées pour la 2ème métrique :**
- Complexity (Bloom's Taxonomy) : Subjectif avec un LLM, difficile à calibrer
- Clarté de la question : Déjà couverte par Clarity en Phase 2
- Self-containment : Sous-ensemble de Contextual Answerability

**Raisonnement pour Pedagogical Value :**
- Problème détecté : Questions circulaires qui obtiennent score 0.93 en V3
  (ex: "Pourquoi X est essentiel?" → "X est essentiel car X est essentiel")
- Pedagogical Value détecte ce pattern (non_trivial = False)
- Indispensable pour un dataset destiné à l'évaluation d'un système RAG

---

## ADR-004 : ScopedMemory pour la Diversité (2026-02-17)

**Contexte :**
Sans mémoire, le Question Generator produit des questions redondantes au sein d'une même section (~40% de redondance observée).

**Décision :**
Implémenter une mémoire contextuelle (ScopedMemory) avec les choix suivants :
1. Reset par section ET chapitre (si l'un des deux change → mémoire vidée)
2. Maximum 5 questions dans le prompt de diversité (limite tokens, n_ctx=4096)
3. Concepts fournis optionnellement par Phase 1 (QuestionEvaluator)
4. Pas d'embeddings en V1 (liste textuelle simple)
5. Emplacement : `src/utils/scoped_memory.py`

**Raisonnement :**
- Reset par section : Les concepts de sections différentes sont non-liés
- 5 questions max : n_ctx=4096, chaque question ~100 chars = ~500 chars total
- Pas d'embeddings V1 : Simplicité d'abord, efficace à 80% des cas
- Le prompt de diversité est injecté dans le Question Generator, pas dans le Critic

**Alternatives rejetées :**
- Vector DB globale : Overhead disproportionné (compare tout avec tout)
- Embedding-Based Similarity : Reporté à V2 (besoin d'un modèle supplémentaire)
- Micro-clustering : Complexité inutile, les chunks sont déjà ordonnés par TOC

**Évolution prévue (V2) :**
- Ajout d'un Embedding Check post-génération (filet de sécurité)
- Modèle léger type all-MiniLM-L6-v2 (pas Qwen 32B)
- Seuil de similarité cosinus > 0.85 → rejet + régénération

---
