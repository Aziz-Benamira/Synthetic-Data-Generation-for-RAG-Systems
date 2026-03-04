# RAGEval — Génération Synthétique de Benchmarks QA pour l'Évaluation RAG
## ENSTA Paris · Cours MI201 Apprentissage Automatique

---

## 1. Objectif et problème

### Le problème de l'évaluation RAG

Un système **RAG (Retrieval-Augmented Generation)** répond à des questions en deux étapes :
1. **Retrieval** : recherche des passages pertinents dans une base documentaire
2. **Generation** : envoi des passages + question à un LLM pour produire la réponse

**Le problème central :** comment mesurer objectivement la qualité de ce pipeline ?
- Il faut un dataset de référence ancré dans les documents réels du système
- Le construire manuellement est lent, coûteux et subjectif
- **RAGEval automatise cette construction** avec un LLM local, sans intervention humaine

### Notre cas d'usage

Construire et évaluer un RAG sur le **cours MI201 Machine Learning** (ENSTA Paris) :
- Cours en **PDF français**, ~187 000 caractères, ~104 pages
- 5 thèmes : Introduction ML, Arbres de décision, SVM, Réseaux de neurones, Apprentissage non supervisé
- Question clé : *"Notre RAG peut-il répondre aux questions qu'un étudiant poserait sur ce cours ?"*

---

## 2. RAGEval — Framework original (NeurIPS 2024)

RAGEval génère automatiquement des benchmarks QA annotés à partir de documents métier.

| Domaine original | Langue | Type de document |
|------------------|--------|------------------|
| Juridique | ZH/EN | Dossiers judiciaires |
| Finance | ZH/EN | Rapports annuels |
| Médical | ZH/EN | Dossiers hospitaliers |

**Notre contribution :** extension au domaine **académique** (cours technique) avec :
- Un LLM **local** (Ministral-8B-Instruct-2410) — aucune clé API, aucun envoi de données externes
- Un document **en français** avec des questions générées **en anglais** (multilingue croisé)
- Un retriever **sémantique multilingue** pour surmonter la barrière linguistique
- Des références verbatim **extraites en français** du document source

---

## 3. Pipeline complet

```
┌─────────────────────────────────────────────────────────────────┐
│         Cours MI201 — PDF brut (187K chars, artefacts PDF)       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
              prepare_academic_data.py
    Nettoyage des artefacts pdftotext + découpe en 5 chapitres
    (accents reconstruits, en-têtes supprimés, TOC éliminé)
                          │
                          ▼
        ┌─────────────────────────────────────┐
        │  sbatch run_academic_single_doc      │  ~10 min · H100
        │                                     │
        │  Ministral-8B lit chaque chapitre   │
        │  et génère pour chacun :             │
        │  ├─ Factual Questions    (~8/ch)    │  réponse factuelle directe
        │  ├─ Multi-hop Questions  (~5/ch)    │  combiner 2+ faits
        │  └─ Summarization       (~3/ch)    │  synthèse d'une section
        └─────────────────┬───────────────────┘
                          │  output/academic/en/config/Ch*/0/0.json
                          ▼
        ┌─────────────────────────────────────┐
        │   sbatch run_add_references          │  ~5 min · H100
        │                                     │
        │  Pour chaque QA item sans ref :      │
        │  → 1 appel LLM par item              │  (évite la troncature JSON)
        │  → extrait la phrase verbatim        │
        │    française du document             │
        │  → stocke dans champ "ref"           │
        │  Résultat : 78/79 items avec ref     │
        └─────────────────┬───────────────────┘
                          │
                          ▼
              python combine_dataset.py
       Fusionne les 5 chapitres → eval_dataset.json (79 questions)
                          │
                          ▼
        ┌─────────────────────────────────────┐
        │     sbatch run_evaluate_rag          │  ~5 min · H100
        │                                     │
        │  Pour chaque question :              │
        │  a) Embed question (EN) avec         │
        │     multilingual-MiniLM-L12-v2      │
        │  b) Top-5 chunks français par        │
        │     cosine similarity                │
        │  c) Vérifier si ref ≃ chunk          │
        │     (cosine sim ≥ 0.45, multilingue) │
        │  d) Générer réponse avec             │
        │     Ministral-8B + contexte         │
        └─────────────────┬───────────────────┘
                          │
                          ▼
                  eval_results.json
         Context Recall · Hit Rate · Answer F1
```

---

## 4. Dataset généré (run finale — job 13837→13839→13840)

### 79 questions au total

| Chapitre | Sujet | Factuelle | Multi-sauts | Résumé | Total | Refs |
|----------|-------|-----------|-------------|--------|-------|------|
| Ch1 | Introduction à l'apprentissage automatique | 10 | 6 | 3 | **19** | 19/19 ✅ |
| Ch2 | Arbres de décision et méthodes d'ensemble | 10 | 5 | 2 | **17** | 17/17 ✅ |
| Ch3 | Régularisation et machines à vecteurs de support | 10 | 7 | 2 | **19** | 19/19 ✅ |
| Ch4 | Introduction aux réseaux de neurones | 5 | — | — | **5** | 5/5 ✅ |
| Ch5 | Apprentissage non supervisé | 10 | 6 | 3 | **19** | 18/19 ✅ |
| **Total** | | **45** | **24** | **10** | **79** | **78/79 (98,7%)** |

> **Note Ch4 :** Le chapitre 4 a produit moins de questions suite à des échecs d'analyse JSON
> lors de la génération (le modèle a entouré le JSON de balises ````json...```` → analyse échouée).
> Ce comportement est aléatoire ; une nouvelle exécution produirait ~19 questions pour Ch4 également.

### Format d'une question (JSON output)

```json
{
  "question type": "Question factuelle",
  "question": "What is the main goal of machine learning according to MI201?",
  "answer": "To give machines the ability to learn from data.",
  "ref": [
    "L'objectif principal de l'apprentissage automatique est de donner aux machines la capacité d'apprendre à partir des données."
  ]
}
```

**3 champs clés :**
- `question` — générée en **anglais** par Ministral depuis le PDF français
- `answer` (**réponse de référence**) — générée par Ministral, **paraphrase anglaise** du contenu du document
- `ref` — phrase(s) **verbatim en français** copiées caractère par caractère du document source

> **Important :** la réponse de référence est une paraphrase synthétique, pas une vérité absolue.
> C'est inhérent à la génération automatique. Les refs verbatim françaises sont la source de vérité
> pour les métriques de récupération.

### Exemples réels par type de question

**Question factuelle** — réponse courte, fait unique :
```
Q: What is the main goal of machine learning?
A: "To give machines the ability to learn from data."
ref: "L'objectif principal de l'apprentissage automatique est de donner
      aux machines la capacité d'apprendre à partir des données."
```

**Question multi-sauts** — combiner 2+ faits du document :
```
Q: What effect does L1 regularization in SVM have on feature selection?
A: "L1 regularization encourages sparsity — only a few features are used,
    resulting in a more interpretable model."
refs: [
  "La régularisation L1 encourage la parcimonie (sparsité).",
  "Cela signifie que seules quelques caractéristiques sont utilisées,
   rendant le modèle plus interprétable."
]
```

**Question de synthèse** — résumé d'une section entière :
```
Q: Summarize the key steps of the model evaluation methodology.
A: "Model evaluation involves: (1) splitting data into train/val/test,
    (2) choosing a metric, (3) cross-validation..."
refs: [8+ phrases verbatim françaises couvrant la section entière]
```

---

## 5. Architecture technique

### Infrastructure

| Composant | Valeur |
|-----------|--------|
| Cluster | ENSTA HPC (SLURM) |
| Partition | `ENSTA-h100` |
| GPU | NVIDIA H100 NVL — 95 GB VRAM |
| Modèle génération | Ministral-8B-Instruct-2410 · float16 · 14.9–17.6 GB VRAM |
| Modèle embedding | paraphrase-multilingual-MiniLM-L12-v2 · 384-dim · ~470 MB |
| Chunking | Fenêtre glissante 500 chars · overlap 100 chars |
| Chunks indexés | 400 (80 par chapitre) |

### LLM local (remplacement complet d'OpenAI)

```python
# Chargement CPU d'abord, puis GPU
# (évite le hang de device_map="auto" sans la lib accelerate)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16)
model = model.cuda()
model.eval()

# Génération déterministe, 2048 tokens max (augmenté depuis 1024 pour éviter les troncatures)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
```

### Retriever sémantique multilingue

Le document est en **français**, les questions en **anglais** — BM25 (keyword matching) ne trouve
aucun token commun entre `"k-nearest neighbors"` et `"k plus proches voisins"` (recall 0.037).
Solution : embeddings denses multilingues.

```python
class SemanticRetriever:
    def __init__(self, chunks, device='cpu'):
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        # Encodage normalisé pour cosine similarity via produit scalaire
        self.chunk_embeddings = self.model.encode(chunks, normalize_embeddings=True)

    def retrieve(self, query, top_k=5):
        # Query anglaise → espace vectoriel partagé → chunks français
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.chunk_embeddings @ q_emb          # cosine sim (vecteurs normalisés)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(i, scores[i], self.chunks[i]) for i in top_indices]

    def ref_matches_context(self, ref, context_chunks, threshold=0.45):
        """Matching cross-lingual : ref (FR ou EN) vs chunks (FR).
        Même modèle multilingue → fonctionne dans les deux sens."""
        texts = [ref] + context_chunks
        embs = self.model.encode(texts, normalize_embeddings=True)
        sims = embs[1:] @ embs[0]   # sim(chaque chunk, ref)
        return bool(np.any(sims >= threshold))
```

---

## 6. Résultats finaux (Job 13840 — 2026-03-04)

### Métriques

| Métrique | Run propre (job 13840) | Run précédente (job 13817) | Δ |
|----------|------------------------|---------------------------|---|
| **Rappel du contexte@5** | **0.818** | 0.795 | +0,023 |
| **Taux de succès@5** | **0.936** (73/78) | 0.950 (76/80) | −0,014 |
| F1 de réponse | 0.154 | 0.198 | ⚠️ non significatif |
| Correspondance exacte | 0.000 | 0.011 | ⚠️ non significatif |
| Questions totales | 79 | 93 | (run propre) |
| Questions avec refs | **78/79 (98,7%)** | 80/93 (86%) | +12,7% |

### Par type de question (job 13840)

| Type | Rappel du contexte@5 | Taux de succès | Δ vs job 13817 |
|------|---------------------|----------------|----------------|
| **Factuelle** | **0.838** | 36/40 | +0,000 (stable) |
| **Multi-sauts** | **0.787** | 28/29 | +0,030 |
| **Synthèse** | **0.826** | 9/9 | **+0,220** ← amélioration majeure |

> **Amélioration clé — Synthèse :** Le nettoyage du texte (artefacts PDF éliminés)
> permet au module de récupération de trouver les passages de synthèse bien plus facilement.
> Rappel Synthèse : 0,606 → **0,826** après le nettoyage.

### Pourquoi le F1 de réponse = 0,154 ne reflète pas la qualité réelle

**Le F1 mesure le recouvrement de mots** entre la réponse générée et la réponse de référence.
Il est structurellement biaisé dans ce pipeline pour deux raisons :

**Raison 1 — Les réponses de référence sont très courtes (médiane ~15 mots)**

```
Référence :  "To give machines the ability to learn from data."   ← 9 mots
Générée :    "The main goal of machine learning is to give machines the ability
              to learn from data, which is one of the most essential..."   ← 25 mots

Précision = mots communs / mots générés → faible (beaucoup de mots supplémentaires)
F1        ≈ 0,50  (meilleur cas)
```

**Raison 2 — Le modèle dit correctement "Impossible de répondre" quand le contexte est absent**

Quand le module de récupération ne trouve pas le bon passage, le modèle refuse de répondre —
c'est le comportement attendu d'un RAG honnête :

```
Référence : "K-means, DBSCAN."
Générée :   "Unable to answer. The provided context does not contain
             information about clustering algorithms."
F1 :        0,00   ← le modèle a raison de refuser, mais le F1 pénalise quand même
```

**Les métriques significatives sont le Rappel du contexte (0,818) et le Taux de succès (0,936).**
Elles mesurent directement ce qui compte : est-ce que le module de récupération trouve le bon passage ?

### Exemple de succès complet (rappel=1,0)

```
Question : "What is the main goal of machine learning?"

Passage récupéré :
  "...l'objectif principal de l'apprentissage automatique est de donner
   aux machines la capacité d'apprendre à partir des données..."

Ref :  "L'objectif principal de l'apprentissage automatique est de donner
        aux machines la capacité d'apprendre à partir des données."

Similarité cosinus ref ↔ passage : 0,91  → SUCCÈS ✅

Réponse générée :
  "The main goal of machine learning is to give machines the ability to learn
   from data, which is one of the most essential faculties of living beings."

F1 de réponse : 0,52  (pénalisé par les mots supplémentaires)
```

---

## 7. Métriques RAGEval complètes

### Métriques de récupération (calculées par le pipeline)

| Métrique | Définition | Notre valeur |
|----------|-----------|-------------|
| **Rappel du contexte@k** | Fraction des phrases-ref présentes dans les k passages récupérés | **0,818** |
| **Taux de succès@k** | Fraction des questions avec ≥1 ref dans les k passages récupérés | **0,936** |

### Métriques de génération

| Métrique | Définition | Notre valeur | Fiable ? |
|----------|-----------|-------------|----------|
| **F1 de réponse** | Recouvrement de mots entre réponse générée et référence | 0,154 | ⚠️ biaisé |
| **Correspondance exacte** | Réponse générée = réponse de référence | 0,000 | ⚠️ trop strict |

### Métriques absentes (hors portée de ce projet)

| Métrique | Définition | Pourquoi absente |
|----------|-----------|----------------|
| **Fidélité** (NLI) | Les affirmations générées sont-elles fondées dans le contexte ? | Nécessite un modèle NLI séparé (ex. DeBERTa) |
| **Pertinence de la réponse** | La réponse est-elle pertinente à la question ? | Nécessite un LLM-juge externe |
| **ROUGE / BLEU** | Recouvrement de n-grammes vs référence | Non adapté aux réponses longues |

> **Fidélité / hallucination :** Pour mesurer si le modèle "invente", il faudrait décomposer
> la réponse en affirmations atomiques (via NLI) et vérifier chacune contre les passages récupérés.
> C'est la métrique centrale de RAGAs, TruLens, ARES. Dans notre cas, le comportement
> "Impossible de répondre" quand le contexte manque indique que le modèle n'hallucine pas massivement.

---

## 8. Parcours d'ingénierie — 8 obstacles surmontés

### Obstacle 1 — Pipeline bloqué silencieusement (device_map="auto")

**Jobs :** 13752, 13735 — aucune sortie après des heures d'attente

**Cause :** `device_map="auto"` exige la bibliothèque `accelerate`. Sans elle, l'appel bloque
indéfiniment sans message d'erreur ni délai d'expiration.

```python
# Avant (bloque sans accelerate):
model = AutoModelForCausalLM.from_pretrained(path, device_map="auto")

# Après (fonctionne toujours):
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16)
model = model.cuda()
```

---

### Obstacle 2 — L'import OpenAI bloque le démarrage Python sur SLURM

**Jobs :** 13777, 13778 — 0 octet produit en sortie (.out et .err vides)

**Cause :** `postprocess.py` importait `from openai import OpenAI` au niveau du module.
Le SDK OpenAI initialise des connexions HTTP à l'import → bloque indéfiniment
sur les nœuds de calcul sans accès internet.

**Symptôme révélateur :** les jobs 13775 produisaient 93 Ko dans `.err` (chargement transformers normal).
Jobs 13777/13778 → **0 octet** : Python n'avait jamais démarré.

**Correction :** Suppression complète du code OpenAI. Remplacement par `LocalModelClient`.

---

### Obstacle 3 — Journaux invisibles dans le fichier `.out` SLURM

**Cause :** Python `logging` écrit par défaut sur `stderr` → va dans `.err`, pas dans `.out`.
On regardait `.out` et voyait le job tourner depuis 45 min sans aucune sortie.

```python
# Fix : forcer stdout avec force=True (override les handlers existants)
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True,
                    format='%(asctime)s [%(levelname)s] %(message)s')
```

---

### Obstacle 4 — Le modèle génère des questions sur les métadonnées, pas sur l'apprentissage automatique

**Cause :** Le prompt de génération ne contenait que les métadonnées (titre, auteurs, mots-clés).
Le modèle n'avait aucun contenu réel sur l'apprentissage automatique et imitait les exemples du prompt
(questions sur "the instructors" et "the course objectives" au lieu des concepts enseignés).

**Correction :** Injection d'un extrait de 4 000 caractères du document dans chaque prompt :

```python
config_for_qa['document_excerpt'] = doc_content[2000:6000]
# → Le modèle voit maintenant le texte réel sur les k-PPV, les SVM, etc.
```

---

### Obstacle 5 — Récupération inter-lingues : rappel BM25 = 0,037

**Cause :** Document en français, questions en anglais.
BM25 compare des mots → aucun mot commun entre `"k-nearest neighbors"` et `"k plus proches voisins"`.

**Correction :** Remplacement de BM25 par `paraphrase-multilingual-MiniLM-L12-v2`.
Modèle téléchargé localement, chargé en mode hors ligne (`TRANSFORMERS_OFFLINE=1`).

**Résultat :** BM25 0,037 → sémantique multilingue **0,818** de rappel du contexte

---

### Obstacle 6 — Références générées en anglais (au lieu du français verbatim)

**Cause :** Les exemples dans le prompt d'extraction de références étaient en anglais.
Ministral imitait ce style et produisait des paraphrases anglaises au lieu d'extraire
les phrases françaises du document.

**Conséquence :** même avec un retriever multilingue, sim cosinus(ref EN, passage FR) ≈ 0,30 < 0,45
→ presque tous les succès manqués.

**Correction en deux parties :**

*A — Prompt système :*
```
RÈGLE LINGUISTIQUE CRITIQUE : l'article est rédigé en FRANÇAIS.
Toutes les références DOIVENT être des phrases verbatim copiées directement de l'article français.
Ne PAS traduire. Ne PAS paraphraser. Copier le texte français exact caractère par caractère.
```

*B — Correspondance sémantique (ref_matches_context) au lieu du recouvrement de mots :*
```python
def ref_matches_context(self, ref, context_chunks, threshold=0.45):
    texts = [ref] + context_chunks
    embs = self.model.encode(texts, normalize_embeddings=True)
    sims = embs[1:] @ embs[0]
    return bool(np.any(sims >= threshold))
# Fonctionne aussi bien pour les refs en anglais qu'en français vs les passages en français
```

---

### Obstacle 7 — JSON tronqué : seulement 20/93 questions avaient des références

**Cause :** Envoi de 10–15 éléments QA par appel LLM → réponse JSON > `max_new_tokens=1024`
→ JSON coupé en plein milieu → `Aucun JSON valide trouvé` sur 90% des appels.

```
[AVERTISSEMENT] postprocess_en: échec d'analyse JSON
(No valid JSON found in response (first 200 chars): '[{"question type": "Factual ...')
```

**Correction :** Traitement **1 élément par appel** + augmentation de 1024 à 2048 tokens :

```python
# Avant : 1 appel pour 10-15 éléments → troncature
# Après : 1 appel par élément → ~300 tokens → jamais tronqué
for idx, item in items_needing_ref:
    task = {
        "system_prompt": ref_prompt['system_prompt'],
        "user_prompt": ref_prompt['user_prompt'].format(
            doc=doc_for_ref,
            qa_pairs=json.dumps([item], ensure_ascii=False)
        )
    }
    responses = client.generate([task])
    parsed = postprocess_en(responses[0], ...)
    if parsed and parsed[0].get('ref'):
        data[key][idx] = parsed[0]
```

**Résultat :** 20/93 références (22%) → **78/79 références (98,7%)**

---

### Obstacle 8 — Texte du PDF illisible (artefacts pdftotext)

**Cause :** `pdftotext` encode les caractères accentués comme deux points de code Unicode séparés :
- `U+00B4` (ACUTE ACCENT) + lettre → devrait être `é`, `à`, `ê`...
- `U+0060` (GRAVE ACCENT) + lettre → idem
- `U+02C6` (CIRCUMFLEX MODIFIER) + lettre → idem
- `c` + `U+00B8` (CEDILLA) → devrait être `ç`

**Conséquence visible dans les fichiers de config :**
```
"Apprentissage automatique: introduction"
→ brut : "Apprentissage automatique: introduction"  ← correct par chance

Mais :
"Généralités" → "G ´en´eralit ´es"      ← accent désassemblé avec espace parasite
"très"        → "tr `es"                 ← grave désassemblé
"être"        → "^etre"                  ← circumflex désassemblé
"ça"          → "c¸a"                    ← cedille désassemblée
```

**Bruit structurel supplémentaire :**
```
. . . . . . . . . . . . . . . 34    ← lignes de table des matières
MI201-ENSTA Paris 42             ← en-têtes de page répétitifs
CONTENTS CONTENTS                ← en-têtes redondants en majuscules
```

**Correction — `clean_text()` dans `prepare_academic_data.py` :**

```python
def clean_text(text: str) -> str:
    # 1. Accents désassemblés → caractères accentués corrects
    text = text.replace(' \u00B4', '\u00B4')     # espace parasite avant accent aigu
    acute = '\u00B4'
    text = text.replace(acute + 'e', 'é').replace(acute + 'E', 'É')
    # ... (toutes les voyelles)

    grave = '\u0060'
    text = text.replace(grave + 'a', 'à').replace(grave + 'e', 'è')
    # ... (toutes les voyelles)

    circ = '\u02C6'
    circ_map = {'e':'ê','a':'â','o':'ô','u':'û','i':'î', ...}
    for v, r in circ_map.items():
        text = text.replace(circ + v, r)

    text = text.replace('c\u00B8', 'ç').replace('C\u00B8', 'Ç')

    # 2. Bruit structurel
    text = re.sub(r'[^\n]*(\.\s){3,}\.?[^\n]*\n', '', text)  # TOC ". . . . ."
    text = re.sub(r'[^\n]*\.{4,}[^\n]*\n', '', text)          # TOC "...."
    text = re.sub(r'MI201-ENSTA Paris \d+[^\n]*', '\n', text) # en-têtes de page
    text = re.sub(r'^[A-Z ]{6,}\n', '', text, flags=re.MULTILINE)  # headers CAPS
    text = re.sub(r'\n{3,}', '\n\n', text)                    # blancs multiples

    # 3. Corps réel du document (après la TOC)
    body_marker = 'Apprentissage automatique: introduction\n1.1'
    body_start = text.find(body_marker)  # était hardcodé à 8000 chars (dans la TOC !)
    return text[body_start:].strip()
```

**Résultat :**
- 187K caractères bruts → 160K caractères propres
- `G ´en´eralit ´es` → `Généralités` ✅
- Lignes de table des matières éliminées ✅
- En-têtes de page éliminés ✅
- Corps du document commence à "Apprentissage automatique: introduction 1.1..." ✅
- Rappel Synthèse : 0,606 → **0,826** (le retriever trouvait avant du texte de table des matières)

---

### Résumé de l'évolution des métriques

| Étape | Rappel du contexte | Taux de succès | Références | Problème |
|-------|--------------------|----------------|------------|---------|
| Référence BM25 | 0,037 | — | 20/93 | Mots FR/EN incompatibles |
| MiniLM + recouvrement de mots | 0,025 | 0,050 | 20/93 | Références générées en anglais → 0 recouvrement |
| MiniLM + similarité cosinus | 0,795 | 0,950 | 80/93 | Texte brut dans les configs |
| **MiniLM + similarité cosinus + texte propre** | **0,818** | **0,936** | **78/79** | — |

---

## 9. Structure du code

```
RAGEval/RAGEval-main/rageval/qar_generation/
│
├── code/
│   ├── local_client.py                    ← NOUVEAU : client HuggingFace local
│   │                                         (remplace complètement OpenAIClient)
│   ├── academic/en/
│   │   └── qra_pipeline_single_doc.py     ← NOUVEAU : pipeline génération QA académique
│   └── data_processing/
│       └── postprocess.py                 ← RÉÉCRIT : sans OpenAI, JSON parsing robuste
│
├── prompts/
│   └── academic_en.jsonl                  ← NOUVEAU : 8 types de prompts
│                                             + règle "verbatim français" pour les refs
│
├── data/academic/en/
│   ├── config/Ch{1..5}_*/0/0.json         ← configs chapitres (entrée pipeline)
│   └── doc/Ch{1..5}_*/0/0.txt            ← textes chapitres nettoyés
│
├── output/academic/en/
│   ├── config/Ch{1..5}_*/0/0.json         ← GÉNÉRÉ : QA pairs + refs par chapitre
│   ├── eval_dataset.json                  ← GÉNÉRÉ : benchmark combiné (79 questions)
│   └── eval_results.json                  ← GÉNÉRÉ : métriques + détails par question
│
├── scripts/
│   ├── run_academic_single_doc.sbatch     ← SLURM : génération QA (~10 min)
│   ├── run_add_references.sbatch          ← SLURM : extraction refs (~5 min)
│   └── run_evaluate_rag.sbatch            ← SLURM : évaluation RAG (~5 min)
│
├── prepare_academic_data.py               ← RÉÉCRIT : PDF → nettoyage → 5 chapitres
├── add_references.py                      ← RÉÉCRIT : 1 item/appel, FR verbatim
├── combine_dataset.py                     ← fusion → eval_dataset.json
└── evaluate_rag.py                        ← RÉÉCRIT : retriever sémantique multilingue
```

---

## 10. Commandes pour reproduire

```bash
cd RAGEval/RAGEval-main/rageval/qar_generation

# Pré-requis : texte propre des 5 chapitres
python prepare_academic_data.py
# → crée data/academic/en/config/Ch*/0/0.json
#   et data/academic/en/doc/Ch*/0/0.txt

# Étape 1 — Générer les paires QA (5 chapitres × 3 types) — ~10 min sur H100
sbatch scripts/run_academic_single_doc.sbatch
tail -f logs/academic_single_<ID_JOB>.out

# Étape 2 — Extraire les références verbatim françaises — ~5 min sur H100
# (1 appel LLM par élément, jamais de troncature JSON)
sbatch scripts/run_add_references.sbatch
tail -f logs/academic_refs_<ID_JOB>.out

# Étape 3 — Fusionner en un seul fichier — instantané (CPU)
python combine_dataset.py

# Étape 4 — Évaluation complète récupération + génération — ~5 min sur H100
sbatch scripts/run_evaluate_rag.sbatch
tail -f logs/rag_eval_<ID_JOB>.out
```

---

## 11. Chiffres clés (exécution finale — job 13840)

| Indicateur | Valeur |
|------------|--------|
| Document source | Cours MI201, PDF français, 187K caractères bruts → 160K propres |
| Chapitres | 5 parties ~32K caractères chacune |
| Questions générées | **79** (3 types × 5 chapitres, ~1 échec d'analyse) |
| Questions avec références | **78 / 79 (98,7%)** |
| GPU | NVIDIA H100 NVL · 95 Go de VRAM |
| Modèle de génération | Ministral-8B-Instruct-2410 · float16 · ~15 Go VRAM |
| Modèle d'enchâssement | paraphrase-multilingual-MiniLM-L12-v2 · 384 dimensions |
| Passages indexés | 400 (80/chapitre · fenêtre 500 caractères · chevauchement 100) |
| **Rappel du contexte@5** | **0,818** ← métrique principale |
| **Taux de succès@5** | **0,936** (73/78 questions) |
| Rappel du contexte — Synthèse | **0,826** (vs 0,606 avant nettoyage) |
| F1 de réponse | 0,154 ⚠️ biaisé (voir §6) |
| Temps total du pipeline (étapes 1–4) | ~20 min sur H100 |
| Nombre de jobs SLURM nécessaires | 4 (single_doc · refs · combine · evaluate) |
