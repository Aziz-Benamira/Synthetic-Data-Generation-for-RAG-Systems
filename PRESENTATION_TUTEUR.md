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

**Justification des poids** : la réponse (Phase 2) est pondérée plus fortement (60%) car une mauvaise réponse est plus dangereuse pour un système RAG qu'une question imparfaite. Dans la Phase 1, la Contextual Answerability est prioritaire (60%) car si le chunk ne contient pas l'info, tout le reste est inutile.

---

### 4.2 Métrique 1 — ContextualAnswerability

**Question posée** : *"Le chunk contient-il assez d'informations pour répondre à cette question ?"*

**Pourquoi c'est important** : sans cette vérification, on génère des questions auxquelles le système RAG ne pourra jamais répondre correctement — le chunk ne contient tout simplement pas l'information.

#### Prompt Système (extrait exact)

```
Tu es un expert en évaluation pédagogique. Ta mission est de déterminer
si un extrait de cours (chunk) contient suffisamment d'informations pour
permettre de répondre à une question donnée.

Échelle de notation (0-3) :
  0 → Aucune information : chunk hors-sujet total
  1 → Information partielle et insuffisante : le sujet est mentionné
      mais manque de détails essentiels
  2 → Information suffisante mais incomplète : éléments principaux
      présents, manque de nuances ou d'exemples
  3 → Information complète : tout ce qu'il faut pour une réponse
      précise et complète

Seuil de rejet : score < 2.0

Format de réponse JSON attendu :
{
  "passages_pertinents": ["citation exacte du chunk", ...],
  "score": <0|1|2|3>,
  "justification": "Explication du score",
  "manquements": ["ce qui empêche un score plus élevé", ...]
}
```

#### Prompt Utilisateur

```
**Chunk de cours :**
{chunk_content}

**Question à évaluer :**
{question}

Consignes :
1. Identifie les passages pertinents pour répondre à cette question
2. Évalue si ces passages sont suffisants pour une réponse complète
3. Attribue un score 0–3 et justifie
4. Liste les manquements s'il y en a

Réponds en JSON uniquement.
```

#### Exemple de sortie LLM réelle

```json
{
  "passages_pertinents": [
    "la loi s'écrit : P + ½ρv² + ρgh = constante",
    "Conditions d'application : 1. Fluide parfait (non visqueux)
     2. Écoulement incompressible  3. Régime permanent  4. Le long d'une ligne de courant"
  ],
  "score": 3,
  "justification": "Le chunk contient exactement les 4 conditions demandées,
                    formulées de manière explicite avec des explications.",
  "manquements": []
}
```

#### Feedback généré en cas de REJECT

```
"Question rejetée: le chunk ne contient pas assez d'informations (score=1/3).
 Manquements: définition formelle de la viscosité, exemple numérique.
 Génère une question plus ancrée dans le contenu disponible."
```

Ce feedback est injecté mot pour mot dans le prompt de régénération du QuestionGeneratorV3.

**Court-circuit** : si REJECT → PedagogicalValue **skippée** → économie de ~8s par question rejetée.

---

### 4.3 Métrique 2 — PedagogicalValue

**Question posée** : *"Cette question a-t-elle une vraie valeur pédagogique ?"*

**Pourquoi c'est important** : un LLM peut facilement générer des questions "correctes" mais creuses — trop vagues, triviales, ou dont la réponse se trouve littéralement dans la question. Ce filtre empêche ça.

#### Architecture : 3 critères binaires indépendants

```
Question
   │
   ├─ tests_understanding ?   → Nécessite une vraie compréhension,
   │                            pas juste copier-coller ?
   │
   ├─ non_trivial ?           → Pas une évidence absolue ?
   │                            Pas trop vague ?
   │
   └─ educational_utility ?   → Apporte quelque chose à l'étudiant ?
                                 Aide à maîtriser un concept clé ?

Score = nb(True) / 3    Seuil : ≥ 0.67 (= 2/3 minimum)
```

#### Prompt Système (points clés)

```
Critère 1 — tests_understanding :
  ✅ OUI si : demande d'expliquer, d'analyser, de relier des idées,
              d'énumérer les conditions/prérequis d'un concept
  ❌ NON si : répéter une définition courte mot-à-mot,
              question circulaire (réponse dans la question)

Critère 2 — non_trivial :
  ✅ OUI si : nécessite réflexion, liste de conditions précises,
              aspects importants et non-évidents
  ❌ NON si : évidence absolue ("le ML utilise des données ?"),
              trop vague ("parlez de X")

Critère 3 — educational_utility :
  ✅ OUI si : aide à maîtriser un concept clé, clarifie les limites
              d'une méthode, approfondit la compréhension
  ❌ NON si : détail anecdotique, date de publication, hors-essentiel

Format JSON attendu :
{
  "tests_understanding": true/false,
  "non_trivial": true/false,
  "educational_utility": true/false,
  "justification": "explication critère par critère",
  "suggestions": "comment améliorer la question"
}
```

#### Robustesse : normalisation des typos LLM

Un problème découvert en production (job 13122) : le LLM écrivait parfois `"educational_utily"` au lieu de `"educational_utility"`. Solution mise en place :

```python
# Normalise tous les préfixes "educational_util*" → "educational_utility"
for k, v in evaluation.items():
    if k.startswith("educational_util"):
        normalized["educational_utility"] = v
```

#### Feedback généré en cas de REJECT

```
"Question rejetée en Phase 1 :
  • Qualité pédagogique insuffisante (score=0.33, critères échoués:
    non_trivial, educational_utility)
  Suggestions : reformuler pour demander une analyse comparative
  plutôt qu'une simple définition."
```

---

### 4.4 Métrique 3 — AnswerCompleteness

**Question posée** : *"La réponse couvre-t-elle tous les aspects demandés par la question ?"*

**Pourquoi c'est important** : un LLM peut répondre "à côté", ne traiter qu'une partie de la question, ou donner une réponse trop superficielle. Cette métrique vérifie la couverture.

#### Mécanisme en 3 étapes

```
Question : "Quels sont les avantages ET les limites de X ?"
                          │
              ┌───────────▼────────────┐
              │  Extraction des aspects │
              │  requis par la question │
              │  → ["avantages", "limites"]
              └───────────┬────────────┘
                          │
              ┌───────────▼─────────────────────────┐
              │  Vérification de couverture          │
              │  • "avantages" → couvert ✓           │
              │  • "limites"   → absent ✗            │
              └───────────┬─────────────────────────┘
                          │
              ┌───────────▼──────────────┐
              │  Score + feedback         │
              │  score=1/3 → REJECT       │
              │  aspects_manquants=["limites"]
              └──────────────────────────┘
```

#### Prompt Système

```
Échelle 0–3 :
  0 → Réponse hors-sujet ou vide
  1 → Réponse partielle (omet des aspects importants)
  2 → Réponse suffisante (aspects principaux couverts, manque détails)
  3 → Réponse complète (tous les aspects, profondeur suffisante)

Seuil de rejet : score < 2.0

Format JSON :
{
  "aspects_requis":   ["aspect 1", "aspect 2", ...],
  "aspects_couverts": ["aspect bien traité", ...],
  "aspects_manquants":["aspect absent ou superficiel", ...],
  "score": <0|1|2|3>,
  "justification": "..."
}
```

#### Feedback injecté dans le prompt de régénération

```
"Réponse incomplète (score=1/3).
 Aspects manquants : ['les limites de la méthode', 'exemple numérique'].
 La réponse doit également aborder ces points pour être acceptée."
```

**Court-circuit** : si REJECT → AnswerAnchoring **skippé** → économie ~8s.

---

### 4.5 Métrique 4 — AnswerAnchoring

**Question posée** : *"Est-ce que la réponse invente des choses absentes du chunk ?"*

**Pourquoi c'est important** : c'est le problème d'hallucination. Le LLM connaît beaucoup de choses par son entraînement et peut "compléter" une réponse avec des informations vraies dans l'absolu, mais absentes du cours — ce qui fausse l'évaluation RAG.

#### Mécanisme de détection

```
Réponse
   │
   ├─ Affirmation 1 : "la loi s'écrit P + ½ρv²..."
   │     → vérification dans le chunk → ANCRÉE ✓
   │
   ├─ Affirmation 2 : "publiée en 1738 par Bernoulli"
   │     → absent du chunk → NON-ANCRÉE ✗  (hallucination !)
   │
   └─ Affirmation 3 : "applicable aux gaz compressibles"
         → contredit le chunk → NON-ANCRÉE ✗  (hallucination !)

Score = f(proportion d'affirmations ancrées)
```

#### Prompt Système

```
Tu es un expert en détection d'hallucinations dans les systèmes RAG.
Pour chaque affirmation de la réponse, vérifie si elle est :
  - Ancrée        : directement présente ou clairement déductible du chunk
  - Non-ancrée    : absente du chunk (hallucination potentielle)
  - Extrapolation : raisonnement logique allant légèrement au-delà

Échelle 0–3 :
  0 → > 50% des affirmations hors-chunk (majoritairement halluciné)
  1 → 25–50% hors-chunk (partiellement ancré)
  2 → < 25% hors-chunk (bien ancré avec extrapolations mineures)
  3 → 0% hors-chunk (parfaitement ancré)

Seuil de rejet : score < 2.0

Format JSON :
{
  "affirmations_ancrees":       ["affirmation supportée par le chunk", ...],
  "affirmations_non_ancrees":   ["affirmation inventée", ...],
  "affirmations_extrapolations":["déduction acceptable", ...],
  "score": <0|1|2|3>,
  "justification": "..."
}
```

#### Exemple réel — test HALLUCINATE (job 13123)

Réponse testée contenait : *"publiée en 1738 dans 'Hydrodynamica'"*, *"applicable à des vitesses supersoniques"*, *"viscosité < 0.001 Pa·s"* — aucun de ces éléments n'était dans le chunk.

**Résultat** : score = 1/3 → REJECT, feedback :
```
"Réponse rejetée: ancrage insuffisant (score=1/3, seuil=2.0).
 Affirmations non-ancrées détectées :
  - 'publiée en 1738' (absent du chunk)
  - 'vitesses supersoniques' (contredit le chunk)
  - 'viscosité < 0.001' (inventé)
 Réécris la réponse en utilisant UNIQUEMENT le contenu du chunk."
```

---

### 4.6 La boucle de feedback — comment ça s'articule

```
                    ┌─────────────────────────┐
                    │   QuestionGeneratorV3    │
                    │                          │
  chunk ──────────► │  1. Prompt initial       │
                    │     + hint ScopedMemory  │
                    │          ↓               │
                    │     LLM → question       │
                    │          ↓               │
                    │  2. CriticV4 Phase 1    │
                    │     ContextualAns. (0-3) │
                    │     PedagogicalVal. (0-1)│
                    │          ↓               │
                    │    PASS ──────────────────────────────────────────┐
                    │    REJECT → feedback                              │
                    │          ↓                                        │
                    │  3. Prompt de régénération                        │
                    │     = question rejetée + feedback + chunk         │
                    │     (max 3 tentatives)                            │
                    └─────────────────────────┘                        │
                                                                        │
                    ┌─────────────────────────┐                        │
                    │   AnswerGeneratorV3      │◄───────────────────────┘
                    │                          │
  question ────────► │  1. Prompt initial       │
                    │          ↓               │
                    │     LLM → réponse        │
                    │          ↓               │
                    │  2. CriticV4 Phase 2    │
                    │     Completeness (0-3)   │
                    │     Anchoring    (0-3)   │
                    │          ↓               │
                    │    PASS ──────────────────────────── Gold Entry
                    │    REJECT → feedback     │
                    │          ↓               │
                    │  3. Prompt de régénération│
                    │     = réponse + feedback  │
                    │     + chunk (max 2x)      │
                    └─────────────────────────┘
```

**Prompt de régénération** (commun aux deux générateurs) :
```
=== QUESTION/RÉPONSE REJETÉE (tentative N) ===
{previous_output}

=== FEEDBACK DU CRITIC ===
{feedback}   ← texte exact généré par la métrique

=== CONTENU SOURCE ===
{chunk_content}

=== INSTRUCTIONS ===
- Corrige PRÉCISÉMENT les problèmes du feedback
- Garde les éléments corrects de la version précédente
- JSON uniquement en sortie
```

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
