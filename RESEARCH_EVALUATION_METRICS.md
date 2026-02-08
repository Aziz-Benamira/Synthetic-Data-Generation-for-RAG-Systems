# Recherche sur les Métriques d'Évaluation QA - Ragas, G-Eval, Nvidia

## 📚 Résumé Exécutif

Ce document synthétise les **prompts exacts**, **méthodes de calcul** et **seuils** utilisés par les frameworks professionnels d'évaluation de systèmes RAG/QA.

---

## 🎯 1. RAGAS Framework

### 1.1 Faithfulness (Fidélité)

**Objectif** : Mesurer la cohérence factuelle entre la réponse et le contexte récupéré.

**Méthode en 2 étapes** :

1. **Décomposer** la réponse en statements individuels
2. **Vérifier** chaque statement contre le contexte (NLI)

**Prompt NLI (Natural Language Inference)** :
```
Your task is to judge the faithfulness of a series of statements based on a given context. 
For each statement you must return verdict as 1 if the statement can be directly inferred 
based on the context or 0 if the statement can not be directly inferred.
```

**Few-Shot Example** :
```
Context: John is a student at XYZ University. He is pursuing a degree in Computer Science. 
He is enrolled in several courses this semester, including Data Structures, Algorithms, 
and Database Management. John is a diligent student and spends a significant amount of time 
studying and completing assignments.

Statements:
1. John is majoring in Biology → verdict: 0 (Cannot be inferred from context)
2. John is taking a course in Computer Science → verdict: 1 (Directly inferable)
3. John is a dedicated student → verdict: 1 (Context mentions "diligent")
4. John has a part-time job → verdict: 0 (No information about employment)
```

**Formule** :
$$\text{Faithfulness} = \frac{\text{Nombre de statements supportés}}{\text{Nombre total de statements}}$$

**Scoring** : Binaire (0 ou 1) par statement, puis moyenne → Score [0, 1]

---

### 1.2 Context Precision

**Objectif** : Le contexte récupéré est-il utile pour arriver à la réponse de référence ?

**Prompt** :
```
Given question, answer and context verify if the context was useful in arriving at 
the given answer. Give verdict as "1" if useful and "0" if not with json output.
```

**Few-Shot Examples** :

**Example 1 (Verdict = 1)** :
```json
{
  "question": "What can you tell me about Albert Einstein?",
  "context": "Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist...",
  "answer": "Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist...",
  "reason": "The provided context was indeed useful in arriving at the given answer.",
  "verdict": 1
}
```

**Example 2 (Verdict = 0)** :
```json
{
  "question": "What is the tallest mountain in the world?",
  "context": "The Andes is the longest continental mountain range in the world, located in South America...",
  "answer": "Mount Everest.",
  "reason": "The provided context discusses the Andes, which does not include Mount Everest.",
  "verdict": 0
}
```

**Formule** : Average Precision sur les chunks ordonnés

---

### 1.3 Context Recall

**Objectif** : Vérifier si chaque phrase de la réponse de référence peut être attribuée au contexte.

**Prompt** :
```
Given a context and an answer, analyze each sentence in the answer and classify if 
the sentence can be attributed to the given context or not. 
Use only 'Yes' (1) or 'No' (0) as a binary classification. Output json with reason.
```

**Few-Shot Example** :
```
Question: What can you tell me about Albert Einstein?
Context: [Texte sur Einstein avec dates, prix Nobel, etc.]
Answer: "Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist... 
He published 4 papers in 1905. Einstein moved to Switzerland in 1895."

Classifications:
- Statement 1 (birthdate): attributed=1, reason="The date of birth is mentioned clearly in context"
- Statement 2 (Nobel Prize): attributed=1, reason="The exact sentence is present in context"  
- Statement 3 (4 papers): attributed=0, reason="There is no mention about papers in context"
- Statement 4 (Switzerland): attributed=0, reason="No supporting evidence in context"
```

**Formule** : 
$$\text{Context Recall} = \frac{\text{Statements attributables}}{\text{Total statements}}$$

---

### 1.4 Answer Relevancy (Pertinence de la Réponse)

**Objectif** : La réponse est-elle pertinente par rapport à la question ?

**Approche originale** :
1. Générer N questions à partir de la réponse
2. Calculer la similarité cosinus entre question originale et questions générées
3. Détecter les réponses "non-committal" (évasives)

**Prompt** :
```
Generate a question for the given answer and identify if answer is noncommittal.
```

**Score** :
$$\text{Answer Relevancy} = \text{mean}(\text{cosine\_similarity}) \times (1 - \text{is\_noncommittal})$$

---

## 🔬 2. G-EVAL Framework

**Référence** : Liu et al., "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment" (NeurIPS 2023)

**Principe clé** : Chain-of-Thought (CoT) + Form-filling paradigm

### 2.1 Coherence (Échelle 1-5)

```
You will be given one summary written for a news article.

Your task is to rate the summary on one metric.

Evaluation Criteria:

Coherence (1-5) - the collective quality of all sentences. We align this dimension 
with the DUC quality question of structure and coherence whereby "the summary should 
be well-structured and well-organized. The summary should not just be a heap of 
related information, but should build from sentence to a coherent body of 
information about a topic."

Evaluation Steps:

1. Read the news article carefully and identify the main topic and key points.
2. Read the summary and compare it to the news article. Check if the summary covers 
   the main topic and key points, and if it presents them in a clear and logical order.
3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 
   is the highest based on the Evaluation Criteria.

Source Text:
{{Document}}

Summary:
{{Summary}}

Evaluation Form (scores ONLY):
- Coherence:
```

---

### 2.2 Consistency (Échelle 1-5)

```
You will be given a news article. You will then be given one summary written for 
this article.

Your task is to rate the summary on one metric.

Evaluation Criteria:

Consistency (1-5) - the factual alignment between the summary and the summarized 
source. A factually consistent summary contains only statements that are entailed 
by the source document. Annotators were also asked to penalize summaries that 
contained hallucinated facts.

Evaluation Steps:

1. Read the news article carefully and identify the main facts and details it presents.
2. Read the summary and compare it to the article. Check if the summary contains 
   any factual errors that are not supported by the article.
3. Assign a score for consistency based on the Evaluation Criteria.

Source Text:
{{Document}}

Summary:
{{Summary}}

Evaluation Form (scores ONLY):
- Consistency:
```

---

### 2.3 Fluency (Échelle 1-3)

```
You will be given one summary written for a news article.

Your task is to rate the summary on one metric.

Evaluation Criteria:

Fluency (1-3): the quality of the summary in terms of grammar, spelling, 
punctuation, word choice, and sentence structure.

- 1: Poor. The summary has many errors that make it hard to understand or sound 
     unnatural.
- 2: Fair. The summary has some errors that affect the clarity or smoothness of 
     the text, but the main points are still comprehensible.
- 3: Good. The summary has few or no errors and is easy to read and follow.

Summary:
{{Summary}}

Evaluation Form (scores ONLY):
- Fluency (1-3):
```

---

### 2.4 Relevance (Échelle 1-5)

```
You will be given one summary written for a news article.

Your task is to rate the summary on one metric.

Evaluation Criteria:

Relevance (1-5) - selection of important content from the source. The summary 
should include only important information from the source document. Annotators 
were instructed to penalize summaries which contained redundancies and excess 
information.

Evaluation Steps:

1. Read the summary and the source document carefully.
2. Compare the summary to the source document and identify the main points of 
   the article.
3. Assess how well the summary covers the main points of the article, and how 
   much irrelevant or redundant information it contains.
4. Assign a relevance score from 1 to 5.

Source Text:
{{Document}}

Summary:
{{Summary}}

Evaluation Form (scores ONLY):
- Relevance:
```

---

## 🏢 3. NVIDIA Metrics (via Ragas)

### 3.1 Answer Accuracy

**Objectif** : Comparer la réponse avec une référence ground truth.

**Méthode** : Double évaluation LLM-as-Judge avec rôles inversés

**Échelle de rating** :
- **0** → La réponse est inexacte ou ne répond pas à la même question
- **2** → La réponse correspond partiellement à la référence
- **4** → La réponse correspond exactement à la référence

**Calcul** :
1. Template 1 : Compare réponse vs référence → rating (0, 2, 4)
2. Template 2 : Compare référence vs réponse (rôles inversés) → rating (0, 2, 4)
3. Normalisation : rating/4 → [0, 1]
4. Score final = moyenne des deux ratings normalisés

---

### 3.2 Context Relevance

**Objectif** : Le contexte récupéré est-il pertinent pour la requête ?

**Échelle** :
- **0** → Le contexte n'est pas du tout pertinent
- **1** → Le contexte est partiellement pertinent
- **2** → Le contexte est complètement pertinent

**Méthode** : Double évaluation LLM avec templates différents
**Normalisation** : rating/2 → [0, 1]
**Score final** : moyenne des deux ratings

---

### 3.3 Response Groundedness

**Objectif** : La réponse est-elle ancrée dans le contexte récupéré ?

**Échelle** :
- **0** → La réponse n'est pas du tout ancrée
- **1** → La réponse est partiellement ancrée
- **2** → La réponse est complètement ancrée

**Méthode** : Double évaluation LLM
**Normalisation** : rating/2 → [0, 1]

---

## 📊 4. Comparaison des Approches

| Métrique | Framework | LLM Calls | Token Usage | Explainabilité |
|----------|-----------|-----------|-------------|----------------|
| Faithfulness | Ragas | 2 (decompose + NLI) | Élevé | Haute (par statement) |
| Response Groundedness | Nvidia | 2 (double judge) | Faible | Faible (score brut) |
| Context Precision | Ragas | 1 | Moyen | Haute (avec raison) |
| Answer Accuracy | Nvidia | 2 (double judge) | Faible | Faible (score brut) |
| G-Eval (any) | G-Eval | 1 | Moyen | Moyenne (CoT implicit) |

---

## 🎯 5. Recommandations pour Notre Projet

### Métriques Prioritaires pour QA Académique

| Métrique | Utilité | Recommandation |
|----------|---------|----------------|
| **Anchoring/Groundedness** | CRITIQUE | Adapter Ragas Faithfulness (2-step NLI) |
| **Answer Accuracy** | HAUTE | Style Nvidia (double judge 0-2-4) |
| **Clarity** | MOYENNE | Style G-Eval (1-3, rubrique claire) |
| **Completeness** | MOYENNE | Adapter Context Recall |

### Approche Suggérée : 1 Prompt par Métrique

**Avantages** :
- Attention non-diluée du LLM
- Prompts spécialisés et calibrés
- Meilleur pour DeepSeek reasoning mode

### Seuils Recommandés

| Score | Interprétation |
|-------|----------------|
| **< 0.3** | Mauvais - REJETER |
| **0.3 - 0.5** | Médiocre - AMÉLIORER |
| **0.5 - 0.7** | Acceptable |
| **0.7 - 0.85** | Bon |
| **> 0.85** | Excellent |

⚠️ **Important** : Ces seuils doivent être **calibrés empiriquement** sur un échantillon de votre dataset !

---

## 💡 6. Design Patterns à Adopter

### Pattern 1 : NLI à 2 Étapes (Faithfulness)
```
Step 1: Décomposer la réponse en statements atomiques
Step 2: Pour chaque statement, verdict binaire (0/1) avec raison
Calcul: moyenne des verdicts
```

### Pattern 2 : Double Judge Symétrique (Answer Accuracy)
```
Judge 1: Compare A vs B → rating
Judge 2: Compare B vs A → rating
Score: moyenne normalisée
```

### Pattern 3 : Rubric Explicite (G-Eval Style)
```
Définir une rubrique avec critères clairs
Demander un score ET une justification step-by-step
```

### Pattern 4 : Few-Shot Générique (Non Domain-Specific)
```
Utiliser des exemples universels (Einstein, Super Bowl, etc.)
Pas d'exemples spécifiques au domaine (probabilités, M2, etc.)
```

---

## 📖 Sources

1. **Ragas GitHub** : https://github.com/explodinggradients/ragas
2. **G-Eval Paper** : Liu et al., arXiv:2303.16634 (2023)
3. **G-Eval Code** : https://github.com/nlpyang/geval
4. **Ragas Docs** : https://docs.ragas.io/en/latest/concepts/metrics/
5. **Nvidia Metrics** : https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/nvidia_metrics/

---

*Document généré le 2025-01-XX pour le projet Agentic AI - ENSTA*
