# Génération Automatique de Gold Datasets pour RAG
### Présentation — Pipeline V4

---

## 1. Contexte et objectif

Un système RAG (*Retrieval-Augmented Generation*) a besoin d'un **Gold Dataset** : des paires (question, réponse_de_référence) pour évaluer ses performances. Les créer manuellement prend des semaines.

**Notre système génère ce dataset automatiquement à partir d'un PDF de cours.**

```
PDF de cours  →  Chunks  →  Questions  →  Réponses  →  Gold Dataset
                              (auto)       (auto)       (validé)
```

---

## 2. Bloc 1 — Chunker sémantique (pas juste `myPDF`)

### Pourquoi pas Docling ?

| Critère | Docling | Notre SemanticChunker |
|---|---|---|
| Découpe | Par page / taille fixe | Par frontières sémantiques (TOC) |
| Contexte | Perdu entre chunks | Préservé (chapitre → section → sous-section) |
| Mathématiques | Coupe les équations | Détecte et préserve les blocs LaTeX |
| Théorèmes/Def. | Non reconnus | Classifiés : `definition`, `theorem`, `example` |
| Métadonnées | Minimales | chunk_id, chapter, section, page_range, semantic_type |

### Architecture du SemanticChunker

```
          PDF
           │
    ┌──────▼───────┐
    │  Extract TOC  │   → hiérarchie : Chapitre > Section > Sous-section
    └──────┬───────┘
           │
    ┌──────▼───────────────┐
    │  Détection sémantique │
    │  ┌─────────────────┐  │
    │  │ definition      │  │   regex : "Définition X.X", "Theorem", ...
    │  │ theorem / lemma │  │
    │  │ equation (LaTeX)│  │   regex : $$...$$ , \[...\]
    │  │ example         │  │
    │  │ text            │  │
    │  └─────────────────┘  │
    └──────┬───────────────┘
           │
    ┌──────▼──────────────────┐
    │  Adaptive chunking       │
    │  - Respecte les frontières│
    │  - Overlap si section >   │
    │    taille max             │
    └──────┬──────────────────┘
           │
    SemanticChunk { content, chunk_id,
                    chapter, section, semantic_type,
                    page_range }
```

**Force clé** : un chunk de type `theorem` ne sera jamais coupé au milieu d'une démonstration. Le retriever récupère des unités logiquement cohérentes.

---

## 3. Évolution du pipeline

### V1 — Baseline (Critic V2)
```
Chunk → QuestionGenerator → AnswerGenerator → CriticAgent → Dataset
         (LLM OpenRouter)    (LLM OpenRouter)   (règles fixes)
```
Problème : critic basé sur règles, pas de feedback loop, questions redondantes.

---

### V2 — Critic V3 (LLM local)
```
Chunk → Generate Q → Generate A → Critic V3 ──REJECT──┐
                                      │                 │
                                    PASS             feedback
                                      │                 │
                                   Dataset    ←── Regenerate (max 2x)
```
Amélioration : retry loop avec feedback. Mais critic monolithique, un seul score global.

---

### V4 — Architecture actuelle (2 phases)
```
Chunk
  │
  ├─ Phase 1 : Filtrage QUESTION
  │    ├─ ContextualAnswerability  (chunk contient-il l'info ?)
  │    └─ PedagogicalValue         (question pédagogiquement valide ?)
  │         REJECT → feedback → retry (max 3x)
  │         PASS ↓
  │
  ├─ ScopedMemory → enregistre les concepts clés
  │                 (évite les questions redondantes)
  │
  ├─ Phase 2 : Validation RÉPONSE
  │    ├─ AnswerCompleteness  (réponse couvre tous les aspects ?)
  │    └─ AnswerAnchoring     (aucune hallucination hors-chunk ?)
  │         REJECT → feedback → retry (max 2x)
  │         PASS ↓
  │
  └─ Gold Entry { question, answer, scores, concepts, ... }
```

---

## 4. Détail du Critic V4

### 4.1 Score global

$$\text{Score} = 0.4 \times \underbrace{(0.6 \cdot \text{Contextual} + 0.4 \cdot \text{Pedagogical})}_{\text{Phase 1 — question}} + 0.6 \times \underbrace{(0.6 \cdot \text{Completeness} + 0.4 \cdot \text{Anchoring})}_{\text{Phase 2 — réponse}}$$

### 4.2 Les 4 métriques

#### ContextualAnswerability (Phase 1)
> *"Le chunk contient-il assez d'information pour répondre à cette question ?"*

- Score 0–3 (0 = hors-sujet, 3 = parfaitement ancré)
- Seuil : ≥ 2.0 pour PASS
- Prompt : demande au LLM de lister les passages pertinents + ce qui manque
- **Court-circuit** : si REJECT → PedagogicalValue skippée (économise ~8s)

#### PedagogicalValue (Phase 1)
> *"Cette question est-elle pédagogiquement valide ?"*

3 critères binaires :
- `tests_understanding` : nécessite une vraie compréhension ?
- `non_trivial` : pas une évidence ou trop vague ?
- `educational_utility` : apporte quelque chose au lecteur ?

Score = critères vrais / 3 — Seuil : ≥ 0.67 (2/3 minimum)

#### AnswerCompleteness (Phase 2)
> *"La réponse couvre-t-elle tous les aspects requis ?"*

- Identifie les aspects attendus, couverts, manquants
- Score 0–3 — Seuil : ≥ 2.0 pour PASS
- **Court-circuit** : si REJECT → AnswerAnchoring skippé

#### AnswerAnchoring (Phase 2)
> *"Chaque affirmation de la réponse est-elle dans le chunk ?"*

- Classifie : ancrée / non-ancrée / extrapolation
- Score 0–3 (0 = majoritairement halluciné, 3 = parfaitement ancré)
- Seuil : ≥ 2.0 pour PASS

### 4.3 Technique : prompt avec sortie JSON forcée

Chaque métrique utilise un prompt en deux parties :

```
SYSTEM : Rôle + critères + format JSON attendu + exemples
USER   : chunk + question (+ réponse pour Phase 2)
```

Exemple de sortie pour ContextualAnswerability :
```json
{
  "passages_pertinents": ["la loi s'écrit P + ½ρv² + ρgh = cte"],
  "manquements": [],
  "score": 3,
  "justification": "Le chunk contient exactement les 4 conditions..."
}
```

Le parsing est robuste : JSON → fallback extraction regex → score 0 (fail-safe).

---

## 5. ScopedMemory — Diversité des questions

```
Section en cours : "Loi de Bernoulli"
                         │
              Questions déjà posées :
              • "Quelles sont les conditions d'application ?"
              • "Expliquez la signification de ρv²/2..."
                         │
              Hint injecté dans le prochain prompt :
              ┌─────────────────────────────────────────┐
              │ Évite les questions sur :               │
              │   conditions, application, signification│
              │ Préfère : démonstration, comparaison,   │
              │           application numérique         │
              └─────────────────────────────────────────┘
```

Résultat mesuré : **+51.6% de diversité** vs sans ScopedMemory.

---

## 6. Résultats

### Test CriticV4 (job 13123 — 4 scénarios)

| Scénario | Attendu | Obtenu | Score | Rejeté par |
|---|---|---|---|---|
| Bonne paire QA | PASS | ✅ PASS | 1.000 | — |
| Question vague | REJECT | ✅ REJECT | 0.240 | Phase 1 (Pedagogical) |
| Question hors-contexte | REJECT | ✅ REJECT | 0.000 | Phase 1 (Contextual) |
| Réponse avec hallucinations | REJECT | ✅ REJECT | 0.840 | Phase 2 (Anchoring) |

**4/4 — 0 erreur**

### Pipeline V4 complet (job 13127 — 5 chunks réels MI201)

| Chunks traités | Gold entries | Score moyen | Temps/chunk |
|---|---|---|---|
| 5 | **5 (100%)** | **0.848 / 1.0** | ~41s |

Pipeline sur les 100 chunks en cours (job 13133).

---

## 7. Stack technique

- **Modèle** : DeepSeek R1 Distill Qwen 32B (IQ3_M, 14.8 GB) — local, GPU L40S
- **Inférence** : `llama.cpp` via `llama-cpp-python`
- **Dataset source** : cours MI201 (100 chunks sémantiques)
- **Format sortie** : JSONL compatible HuggingFace Datasets

---

*Pipeline implémenté en 7 étapes sur la branche `Aziz_branch`.*
