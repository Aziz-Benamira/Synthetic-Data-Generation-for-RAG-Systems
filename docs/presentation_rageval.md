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

## 3. Flux de données — description narrative

### Pourquoi ce flux existe

Le document source est un PDF académique français converti en texte brut par `pdftotext`. Ce format pose trois problèmes qui rendent le texte inexploitable directement :

1. **Artefacts d'accents** : `pdftotext` décompose les caractères accentués en deux points Unicode (`U+00B4` + `e` au lieu de `é`). Le résultat brut donne `G ´en´eralit ´es` au lieu de `Généralités`. Le LLM ne reconnaît pas ces mots.

2. **Table des matières (TOC)** : les premières pages du PDF contiennent la TOC avec des lignes du type `"1.1 Introduction . . . . . . . . . 7"`. `pdftotext` les inclut en début de texte. Si on découpait le document sans les supprimer, les premiers "chapitres" ne contiendraient que des lignes de TOC, pas de contenu réel.

3. **En-têtes de page** : chaque page imprimée porte le header `"MI201-ENSTA Paris 42"` (avec le numéro de page). Ces lignes apparaissent des centaines de fois dans le texte extrait et polluent tous les passages récupérés par le retriever.

### Le flux étape par étape

**Entrée brute** — `taxonomy/data/real_doc.txt` : 187 000 caractères avec artefacts, TOC, en-têtes.

**Étape 0 — Nettoyage et découpe** (`prepare_academic_data.py`) :
- `clean_text()` reconstruit les accents, supprime les lignes TOC (pattern `. . . . .`), supprime les en-têtes `MI201-ENSTA Paris N`, supprime les lignes tout-en-majuscules, saute au début du corps réel du document (marqueur `"Apprentissage automatique: introduction\n1.1"`)
- `split_document()` coupe le corps propre en 5 parties égales alignées sur une fin de ligne (~32 000 chars chacune)
- Pour chaque chapitre, deux fichiers sont créés :
  - `data/academic/en/config/Ch1_Intro_ML/0/0.json` — **fichier config** : JSON avec les métadonnées (titre, mots-clés, pages, extrait de contenu). C'est le "passeport" du chapitre que le pipeline lit pour savoir de quoi il parle.
  - `data/academic/en/doc/Ch1_Intro_ML/0/0.txt` — texte nettoyé du chapitre (~32 000 chars de français propre)

**Étape 1 — Génération QA** (`qra_pipeline_single_doc.py` via SLURM) :
- Pour chaque chapitre, le script lit le config JSON + le doc TXT
- Il injecte les 8 000 premiers caractères du chapitre dans les prompts
- Ministral-8B reçoit 3 appels successifs (un par type de question) et produit un tableau JSON
- Le résultat (les QA pairs) est ajouté directement dans le config JSON et sauvé dans `output/academic/en/config/Ch1_Intro_ML/0/0.json` — **le fichier config d'entrée devient le fichier de sortie enrichi**

**Étape 2 — Extraction des références verbatim** (`add_references_academic.py` via SLURM) :
- Pour chaque question sans champ `"ref"`, le script envoie **1 appel LLM séparé** (évite la troncature JSON)
- Le LLM reçoit le texte français complet du chapitre + la question/réponse et doit retrouver la phrase exacte en français
- La phrase verbatim est stockée dans le champ `"ref"` du même config JSON

**Étape 3 — Fusion** (`combine_dataset.py`, CPU) :
- Lit les 5 configs de sortie, aplatit toutes les QA pairs dans une liste unique
- Ajoute un champ `"chapter"` pour savoir d'où vient chaque question
- Sauve `output/academic/en/eval_dataset.json` (79 questions, format plat)

**Étape 4 — Évaluation RAG** (`evaluate_rag.py` via SLURM) :
- Charge les 5 textes de chapitres, les découpe en chunks (500 chars, overlap 100)
- Encode tous les chunks en français avec MiniLM-L12-v2 → 400 vecteurs de 384 dims
- Pour chaque question anglaise : encode la question, calcule la similarité cosinus avec tous les chunks, prend le top-5
- Vérifie si la référence verbatim française est sémantiquement présente dans les 5 chunks récupérés (seuil cosinus 0,45)
- Génère une réponse avec Ministral-8B en passant les 5 chunks comme contexte
- Calcule Context Recall, Hit Rate, Answer F1

---

## 4. Étape 0 — Nettoyage PDF (prepare_academic_data.py)

### Code : clean_text()

```python
def clean_text(text: str) -> str:
    # 1. Accents désassemblés → caractères accentués corrects
    text = text.replace(' \u00B4', '\u00B4')     # espace parasite avant accent aigu
    acute = '\u00B4'
    for base, accented in [('e','é'),('E','É'),('a','á'),('o','ó'),('u','ú'),('i','í')]:
        text = text.replace(acute + base, accented)

    grave = '\u0060'
    for base, accented in [('e','è'),('E','È'),('a','à'),('A','À'),('u','ù')]:
        text = text.replace(grave + base, accented)

    circ = '\u02C6'
    for base, accented in [('e','ê'),('a','â'),('i','î'),('o','ô'),('u','û')]:
        text = text.replace(circ + base, accented)

    text = text.replace('c\u00B8', 'ç').replace('C\u00B8', 'Ç')   # cédille

    # 2. Suppression de la table des matières (". . . . .")
    text = re.sub(r'[^\n]*\.{4,}[^\n]*\n', '', text)

    # 3. Suppression des en-têtes de page répétitifs ("MI201-ENSTA Paris 42")
    text = re.sub(r'MI201-ENSTA Paris \d+[A-Z]?[^\n]*\n?', '', text)

    # 4. Suppression des titres tout-en-majuscules ("CONTENTS CONTENTS")
    text = re.sub(r'^[A-Z ]{6,}\n', '', text, flags=re.MULTILINE)

    # 5. Corps du document = tout ce qui vient après la TOC
    body_marker = 'Apprentissage automatique: introduction\n1.1'
    body_start = text.find(body_marker)   # sauter la TOC entière
    return text[body_start:].strip()
```

**Résultat :** 187K chars bruts → 160K chars propres · `Généralités` au lieu de `G ´en´eralit ´es`

### Format du fichier config (entrée — data/)

```json
{
  "course_code": "MI201",
  "course_title": "Introduction to Machine Learning",
  "institution": "ENSTA Paris",
  "chapter": "Chapter 1: Introduction to Machine Learning",
  "keywords": ["machine learning", "k-NN", "Bayesian", "supervised learning", ...],
  "pages": "pages 7-32",
  "content_summary": "Apprentissage automatique: introduction 1.1 Qu'est-ce..."
}
```

Après l'étape 1, ce même fichier (dans `output/`) devient :

```json
{
  "course_code": "MI201",
  ... (mêmes métadonnées) ...
  "qa_fact_based": [
    {"question type": "Factual Question", "question": "What is the main goal of ML?", "answer": "..."}
  ],
  "qa_multi_hop": [ ... ],
  "qa_summary": [ ... ]
}
```

---

## 5. Étape 1 — Génération QA (qra_pipeline_single_doc.py)

### Prompts donnés au LLM

**Prompt système (tous types) :**
```
You are an expert in generating [factual / multi-hop reasoning / summarization]
questions about academic course material.
```

**Prompt utilisateur — Questions factuelles :**
```
Factual questions have a single, precise answer. Based on the course document
excerpt below, generate factual questions about concepts, definitions,
algorithms, and formulas.

Course: MI201 - Introduction to Machine Learning (ENSTA Paris)
Chapter: Chapter 1: Introduction to Machine Learning
Keywords: machine learning, k-NN, Bayesian, supervised learning, ...

Document excerpt:
Apprentissage automatique: introduction
1.1 Qu'est-ce que l'apprentissage automatique ?
L'objectif principal de l'apprentissage automatique est de donner aux machines
la capacité d'apprendre à partir des données...
[8 000 premiers caractères du chapitre]

Instructions:
1. Generate exactly 10 factual questions.
2. Each question must have a single correct answer grounded in the document.
3. Questions should cover different concepts (not all about the same topic).
4. Use English for both questions and answers.
5. Format as a JSON array:
[{"question type": "Factual Question", "question": "...", "answer": "..."}, ...]
Output only the JSON array, no other text.
```

**Prompt utilisateur — Questions multi-sauts :**
```
Multi-hop questions require combining 2 or more facts from the document
to reach an answer through reasoning.

[mêmes métadonnées + extrait]

Instructions:
1. Generate exactly 5 multi-hop reasoning questions.
2. Each question must require connecting at least 2 separate facts from the document.
3. The answer should be derived through logical steps, not found in a single sentence.
4. Use English for both questions and answers.
5. Format as a JSON array:
[{"question type": "Multi-hop Reasoning Question", "question": "...", "answer": "..."}, ...]
Output only the JSON array, no other text.
```

**Prompt utilisateur — Questions de synthèse :**
```
Summarization questions require a comprehensive answer covering multiple
aspects of a topic or section.

[mêmes métadonnées + extrait]

Instructions:
1. Generate exactly 2 summarization questions.
2. Each question should require synthesizing information from across the section.
3. Answers should be multi-sentence, covering key points.
4. Use English for both questions and answers.
5. Format as a JSON array:
[{"question type": "Summarization Question", "question": "...", "answer": "..."}, ...]
Output only the JSON array, no other text.
```

### Code : envoi des prompts au LLM

```python
def build_user_prompt(prompt_type: str) -> str:
    """Construit le prompt avec f-strings (jamais .format() — le doc contient des { })."""
    excerpt = doc_content[:8000]
    if prompt_type == 'Factual Question':
        return (
            f"Factual questions have a single, precise answer...\n\n"
            f"Course: {course_code} - {course_title} ({institution})\n"
            f"Chapter: {chapter}\n"
            f"Keywords: {keywords_str}\n\n"
            f"Document excerpt:\n{excerpt}\n\n"
            f"Instructions:\n1. Generate exactly 10 factual questions.\n..."
        )
    # idem pour Multi-hop et Summarization

# 3 appels LLM par chapitre (un par type de question)
qa_tasks = [
    {'system_prompt': p['system_prompt'], 'user_prompt': build_user_prompt(pt)}
    for pt in ['Factual Question', 'Multi-hop Reasoning Question', 'Summarization Question']
]
responses = client.generate(qa_tasks)   # → liste de 3 réponses JSON

for key, resp in zip(['qa_fact_based', 'qa_multi_hop', 'qa_summary'], responses):
    config[key] = parse_json_response(resp)   # parse + stocke dans le config
```

```python
def generate(self, tasks, max_new_tokens=2048):
    for task in tasks:
        messages = [
            {"role": "system", "content": task["system_prompt"]},
            {"role": "user",   "content": task["user_prompt"]},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)
        out = model.generate(inputs, max_new_tokens=2048, do_sample=False)
        new_tokens = out[0][inputs.shape[1]:]   # retirer les tokens d'entrée
        yield tokenizer.decode(new_tokens, skip_special_tokens=True)
```

---

## 6. Étape 2 — Extraction des références verbatim (add_references_academic.py)

### Prompt donné au LLM

**Prompt système :**
```
You are an expert in finding verbatim reference sentences in French academic documents.
```

**Prompt utilisateur (1 item QA à la fois) :**
```
CRITICAL LANGUAGE RULE: The source document is written in FRENCH.
All references MUST be verbatim sentences copied DIRECTLY from the French document.
Do NOT translate. Do NOT paraphrase. Copy the exact French text character by character.

Source document (French):
Apprentissage automatique: introduction
1.1 Qu'est-ce que l'apprentissage automatique ?
L'objectif principal de l'apprentissage automatique est de donner aux machines...
[12 000 premiers caractères du chapitre]

For each question-answer pair below, find the verbatim French sentence(s) from
the document that support the answer. Copy them exactly.

QA pairs:
[{"question type": "Factual Question",
  "question": "What is the main goal of machine learning?",
  "answer": "To give machines the ability to learn from data."}]

Return a JSON array with the same items, each with a 'ref' field containing
a list of verbatim French sentences:
[{"question type": "...", "question": "...", "answer": "...",
  "ref": ["verbatim French sentence 1", ...]}]
Output only the JSON array, no other text.
```

### Code : 1 appel par item (évite la troncature JSON)

```python
for key in ['qa_fact_based', 'qa_multi_hop', 'qa_summary']:
    for idx, item in enumerate(config[key]):
        if item.get('ref'):          # déjà une ref → on passe
            continue

        # Prompt construit par concaténation directe (pas .format() sur le doc)
        user_prompt = (
            "CRITICAL LANGUAGE RULE: The source document is written in FRENCH.\n"
            "All references MUST be verbatim sentences...\n\n"
            f"Source document (French):\n{doc_content[:12000]}\n\n"
            f"QA pairs:\n{json.dumps([item], ensure_ascii=False)}\n\n"
            "Return a JSON array..."
        )

        response = client.generate_one(ref_prompt['system_prompt'], user_prompt)
        parsed = parse_json_response(response)

        if parsed and parsed[0].get('ref'):
            config[key][idx] = parsed[0]    # mise à jour in-place
```

**Pourquoi 1 item par appel ?** Avec 10–15 items par appel, la réponse JSON dépasse `max_new_tokens=1024` et est tronquée en plein milieu → `json.JSONDecodeError`. Avec 1 item, la réponse fait ~300 tokens, jamais tronquée.

---

## 7. Étape 3 — Fusion (combine_dataset.py)

```python
all_questions = []
for config_path in sorted(glob.glob('output/academic/en/config/**/*.json')):
    config = json.load(open(config_path))
    chapter_id = config_path.split(os.sep)[-3]   # "Ch1_Intro_ML"
    for key in ['qa_fact_based', 'qa_multi_hop', 'qa_summary']:
        for item in config.get(key, []):
            item['chapter'] = chapter_id
            item['question_type'] = item.pop('question type', key)
            all_questions.append(item)

json.dump(all_questions, open('output/academic/en/eval_dataset.json', 'w'))
# → 79 questions, toutes avec "chapter" et "question_type"
```

---

## 8. Étape 4 — Évaluation RAG (evaluate_rag.py)

### Prompt donné au LLM pour la génération de réponse

**Prompt système :**
```
You are a helpful assistant. Answer the question based ONLY on the provided context.
If the context does not contain the answer, say 'Unable to answer. The provided
context does not contain information about this topic.' Do not use prior knowledge.
```

**Prompt utilisateur (construit à la volée pour chaque question) :**
```
Context:
...l'objectif principal de l'apprentissage automatique est de donner aux machines
la capacité d'apprendre à partir des données...

---

...Le k-NN (k plus proches voisins) est un algorithme non paramétrique qui...

---

[3 autres chunks français]

Question: What is the main goal of machine learning?

Answer:
```

### Code : retrieval sémantique multilingue

```python
class SemanticIndex:
    def __init__(self, embed_model):
        self.model = embed_model   # paraphrase-multilingual-MiniLM-L12-v2

    def build(self, chunks):
        # Encode tous les chunks français une seule fois
        self.chunks = chunks
        self.embeddings = self.model.encode(chunks, normalize_embeddings=True)

    def retrieve(self, query, top_k=5):
        # Query en anglais → même espace vectoriel que les chunks français
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ q_emb        # cosine sim (vecteurs normalisés)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.chunks[i] for i in top_indices]

    def ref_matches_context(self, ref, context_chunks, threshold=0.45):
        """Matching cross-lingual : ref (FR verbatim) vs chunks (FR).
        Seuil 0.45 : assez bas pour autoriser légères variations, assez haut
        pour éviter les faux positifs."""
        texts = [ref] + context_chunks
        embs = self.model.encode(texts, normalize_embeddings=True)
        sims = embs[1:] @ embs[0]   # sim(chaque chunk, ref)
        return bool(np.any(sims >= threshold))
```

### Code : calcul des métriques

```python
for q in questions:
    retrieved = index.retrieve(q['question'], top_k=5)

    # Context Recall : fraction des refs trouvées dans les 5 chunks
    refs = q.get('ref', [])
    matched = [r for r in refs if index.ref_matches_context(r, retrieved)]
    context_recall = len(matched) / len(refs)   # 0.0 → 1.0

    # Hit Rate : ≥1 ref trouvée ?
    hit = len(matched) > 0

    # Génération
    context_text = '\n\n---\n\n'.join(retrieved)
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {q['question']}\n\nAnswer:"
    pred = gen_client.generate([{'system_prompt': SYSTEM_PROMPT, 'user_prompt': user_prompt}])[0]

    # Answer F1 (recouvrement de mots)
    pred_tok = set(pred.lower().split())
    gold_tok = set(q['answer'].lower().split())
    common = pred_tok & gold_tok
    f1 = 2 * len(common) / (len(pred_tok) + len(gold_tok))
```

---

## 9. Dataset généré (run finale — job 13837→13839→13840)

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
> lors de la génération (le modèle a entouré le JSON de balises ` ```json...``` ` → analyse échouée).

### Format d'une question (JSON output)

```json
{
  "question_type": "Factual Question",
  "question": "What is the main goal of machine learning according to MI201?",
  "answer": "To give machines the ability to learn from data.",
  "ref": [
    "L'objectif principal de l'apprentissage automatique est de donner aux machines la capacité d'apprendre à partir des données."
  ],
  "chapter": "Ch1_Intro_ML"
}
```

**3 champs clés :**
- `question` — générée en **anglais** par Ministral depuis le PDF français
- `answer` (**réponse de référence**) — paraphrase anglaise synthétique du contenu
- `ref` — phrase(s) **verbatim en français** copiées caractère par caractère du document source

---

## 10. Architecture technique

### Infrastructure

| Composant | Valeur |
|-----------|--------|
| Cluster | ENSTA HPC (SLURM) |
| Partition | `ENSTA-h100` |
| GPU | NVIDIA H100 NVL — 95 GB VRAM |
| Modèle génération | Ministral-8B-Instruct-2410 · float16 · ~15 GB VRAM |
| Modèle embedding | paraphrase-multilingual-MiniLM-L12-v2 · 384-dim · ~470 MB |
| Chunking | Fenêtre glissante 500 chars · overlap 100 chars |
| Chunks indexés | 400 (80 par chapitre) |

### Pourquoi MiniLM multilingue et pas BM25

Le document est en **français**, les questions en **anglais** — BM25 (keyword matching) ne trouve
aucun token commun entre `"k-nearest neighbors"` et `"k plus proches voisins"`.

```
BM25 recall           : 0.037   ← quasi nul
MiniLM multilingue    : 0.818   ← espace vectoriel partagé FR/EN
```

`paraphrase-multilingual-MiniLM-L12-v2` projette toutes les langues dans le même espace vectoriel.
Une question anglaise et sa traduction française ont une similarité cosinus > 0.85.

---

## 11. Résultats finaux (Job 13840 — 2026-03-04)

### Métriques

| Métrique | Run propre (job 13840) | Run précédente (job 13817) | Δ |
|----------|------------------------|---------------------------|---|
| **Rappel du contexte@5** | **0.818** | 0.795 | +0,023 |
| **Taux de succès@5** | **0.936** (73/78) | 0.950 (76/80) | −0,014 |
| F1 de réponse | 0.154 | 0.198 | ⚠️ non significatif |
| Correspondance exacte | 0.000 | 0.011 | ⚠️ trop strict |
| Questions totales | 79 | 93 | (run propre, moins de duplicats) |
| Questions avec refs | **78/79 (98,7%)** | 80/93 (86%) | +12,7% |

### Par type de question

| Type | Rappel du contexte@5 | Taux de succès | Δ vs job 13817 |
|------|---------------------|----------------|----------------|
| **Factuelle** | **0.838** | 36/40 | stable |
| **Multi-sauts** | **0.787** | 28/29 | +0,030 |
| **Synthèse** | **0.826** | 9/9 | **+0,220** ← amélioration majeure |

> **Amélioration Synthèse :** avant le nettoyage, les premiers chunks contenaient des lignes
> de TOC (`. . . . . . 34`). Le retriever récupérait ces lignes au lieu des vraies sections.
> Après `clean_text()`, rappel synthèse : 0,606 → **0,826**.

### Pourquoi le F1 de réponse = 0,154 ne reflète pas la qualité réelle

```
Référence :  "To give machines the ability to learn from data."   ← 9 mots
Générée :    "The main goal of machine learning is to give machines the ability
              to learn from data, which is one of the most essential..."   ← 25 mots

F1 ≈ 0.50  (pénalisé par les mots supplémentaires de contexte)
```

Quand le retriever rate, le modèle répond honnêtement `"Unable to answer"` → F1 = 0,00
même si c'est le comportement attendu.

**Les métriques significatives sont le Rappel du contexte (0,818) et le Taux de succès (0,936).**

---

## 12. Métriques RAGEval complètes

### Métriques de récupération

| Métrique | Définition | Notre valeur |
|----------|-----------|-------------|
| **Rappel du contexte@k** | Fraction des phrases-ref présentes dans les k passages récupérés | **0,818** |
| **Taux de succès@k** | Fraction des questions avec ≥1 ref dans les k passages récupérés | **0,936** |

### Métriques de génération

| Métrique | Définition | Notre valeur | Fiable ? |
|----------|-----------|-------------|----------|
| **F1 de réponse** | Recouvrement de mots entre réponse générée et référence | 0,154 | ⚠️ biaisé |
| **Correspondance exacte** | Réponse générée = réponse de référence | 0,000 | ⚠️ trop strict |

### Métriques absentes (hors portée)

| Métrique | Définition | Pourquoi absente |
|----------|-----------|----------------|
| **Fidélité** (NLI) | Les affirmations générées sont-elles fondées dans le contexte ? | Nécessite un modèle NLI séparé (DeBERTa) |
| **Pertinence de la réponse** | La réponse est-elle pertinente à la question ? | Nécessite un LLM-juge externe |
| **ROUGE / BLEU** | Recouvrement de n-grammes vs référence | Non adapté aux réponses longues |

---

## 13. Obstacles surmontés (résumé)

| # | Symptôme | Cause | Correction |
|---|----------|-------|-----------|
| 1 | Job bloqué indéfiniment sans sortie | `device_map="auto"` sans `accelerate` | Chargement CPU puis `.cuda()` |
| 2 | 0 octet en sortie SLURM | Import OpenAI bloque sur nœud sans internet | Suppression OpenAI → `LocalModelClient` |
| 3 | Logs invisibles dans `.out` | `logging` écrit sur stderr par défaut | `stream=sys.stdout, force=True` |
| 4 | Questions sur les métadonnées, pas le ML | Prompt ne contenait pas le texte du cours | Injection de 8 000 chars du chapitre |
| 5 | Rappel BM25 = 0,037 | Doc FR / questions EN → 0 mot commun | Remplacement par MiniLM multilingue |
| 6 | Références générées en anglais | Exemples du prompt étaient en anglais | Règle linguistique explicite dans le prompt |
| 7 | 20/93 références seulement | 10-15 items par appel → JSON tronqué | 1 item par appel LLM |
| 8 | Texte illisible, accents cassés | Artefacts pdftotext | `clean_text()` reconstruit les accents |

---

## 14. Structure du code

```
RAGEval/RAGEval-main/rageval/qar_generation/
│
├── prepare_academic_data.py               ← Étape 0 : PDF → nettoyage → 5 chapitres
├── add_references_academic.py             ← Étape 2 : extraction refs verbatim FR
├── combine_dataset.py                     ← Étape 3 : fusion → eval_dataset.json
├── evaluate_rag.py                        ← Étape 4 : retrieval + génération + métriques
│
├── code/academic/en/
│   └── qra_pipeline_single_doc.py         ← Étape 1 : génération QA via Ministral-8B
│
├── prompts/
│   └── academic_en.jsonl                  ← 4 types de prompts (system + user templates)
│
├── data/academic/en/
│   ├── config/Ch{1..5}_*/0/0.json         ← configs chapitres (entrée pipeline)
│   └── doc/Ch{1..5}_*/0/0.txt             ← textes chapitres nettoyés
│
├── output/academic/en/
│   ├── config/Ch{1..5}_*/0/0.json         ← GÉNÉRÉ : QA pairs + refs par chapitre
│   ├── eval_dataset.json                  ← GÉNÉRÉ : benchmark combiné (79 questions)
│   └── eval_results.json                  ← GÉNÉRÉ : métriques + détails par question
│
└── scripts/
    ├── run_academic_single_doc.sbatch      ← SLURM : génération QA (~10 min)
    ├── run_add_references.sbatch           ← SLURM : extraction refs (~5 min)
    └── run_evaluate_rag.sbatch             ← SLURM : évaluation RAG (~5 min)
```

---

## 15. Commandes pour reproduire

```bash
cd RAGEval/RAGEval-main/rageval/qar_generation

# Étape 0 — Nettoyage PDF + découpe en 5 chapitres (CPU, instantané)
python prepare_academic_data.py

# Étape 1 — Génération QA : 3 types × 5 chapitres — ~10 min sur H100
sbatch scripts/run_academic_single_doc.sbatch
tail -f logs/academic_single_<ID_JOB>.out

# Étape 2 — Références verbatim françaises — ~5 min sur H100
sbatch scripts/run_add_references.sbatch
tail -f logs/academic_refs_<ID_JOB>.out

# Étape 3 — Fusion en un seul fichier (CPU, instantané)
python combine_dataset.py

# Étape 4 — Évaluation complète retrieval + génération — ~5 min sur H100
sbatch scripts/run_evaluate_rag.sbatch
tail -f logs/rag_eval_<ID_JOB>.out
```

---

## 16. Chiffres clés (exécution finale — job 13840)

| Indicateur | Valeur |
|------------|--------|
| Document source | Cours MI201, PDF français, 187K chars bruts → 160K propres |
| Chapitres | 5 parties ~32K chars chacune |
| Questions générées | **79** (3 types × 5 chapitres) |
| Questions avec références | **78 / 79 (98,7%)** |
| GPU | NVIDIA H100 NVL · 95 Go de VRAM |
| Modèle de génération | Ministral-8B-Instruct-2410 · float16 · ~15 Go VRAM |
| Modèle d'embedding | paraphrase-multilingual-MiniLM-L12-v2 · 384 dimensions |
| Passages indexés | 400 (80/chapitre · fenêtre 500 chars · overlap 100) |
| **Rappel du contexte@5** | **0,818** ← métrique principale |
| **Taux de succès@5** | **0,936** (73/78 questions) |
| Rappel synthèse avant/après nettoyage | 0,606 → **0,826** |
| F1 de réponse | 0,154 ⚠️ biaisé (voir §11) |
| Temps total pipeline (étapes 0–4) | ~20 min sur H100 |
