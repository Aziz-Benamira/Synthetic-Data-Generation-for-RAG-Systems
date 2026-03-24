# Pipeline QA Académique + Évaluation RAG

Pipeline bout-en-bout pour générer automatiquement un Gold Dataset de questions-réponses à partir de n'importe quel polycopié PDF, puis évaluer un système RAG dessus — sans annotation humaine.

---

## Ce que fait le projet

```
PDF académique
    │
    ▼
SemanticChunker          → découpe le PDF en unités sémantiques (définitions,
    │                       théorèmes, exemples, texte courant) en respectant
    │                       la structure TOC du document
    ▼
PipelineV4 + CriticV4    → génère des paires QA (DeepSeek R1-32B)
    │                       et les filtre via un Critic en 3 phases :
    │                         Phase 1 : qualité / standalone de la question
    │                         Phase 2 : complétude + ancrage au chunk
    │                         Phase 3 : niveau de difficulté cognitive (Bloom 1–5)
    ▼
Gold Dataset (.jsonl)    → 123 QA pour Physique Dunod, 84 QA pour MI201
    │                       enrichi avec difficulty_level, difficulty_label
    ▼
Évaluation RAG           → Qwen2.5-32B répond aux questions via RAG
    │                       (bge-m3 + ChromaDB, top-5)
    ▼
Métriques + Rapport      → Hit Rate, MRR, ROUGE-L, BERTScore, LLM Judge
                            stratifiés par niveau de difficulté (optionnel)
```

---

## Résultats obtenus

### Physique Dunod (1538 pages)

| Métrique | Score |
|---|---|
| Hit@5 (retrieval) | **99.2 %** |
| Hit@1 (retrieval) | **88.6 %** |
| MRR | **0.934** |
| ROUGE-L F1 | 0.431 |
| BERTScore F1 | 0.909 |
| LLM Judge | 3.96 / 5 |

### MI201 — Machine Learning (~200 pages)

| Métrique | Score |
|---|---|
| Hit@5 | 96.4 % |
| Hit@1 | 69.9 % |
| MRR | 0.822 |
| ROUGE-L F1 | 0.420 |
| BERTScore F1 | 0.916 |
| LLM Judge | 4.20 / 5 |

Les résultats complets sont dans [`results_final/`](results_final/).

---

## Structure du projet

```
Agentic_AI/
│
├── Projet_Pipeline.py          ← POINT D'ENTRÉE PRINCIPAL
│                                  Orchestre : chunking → Gold → RAG eval
│
├── run_pipeline_v4_full.py     ← Lancer la génération Gold Dataset (script)
│
├── scripts/
│   └── annotate_difficulty.py  ← Post-annoter un Gold Dataset existant
│                                  avec les niveaux de difficulté (Bloom 1–5)
│
├── src/
│   ├── chunking/
│   │   └── semantic_chunker.py   ← Découpage sémantique du PDF (TOC-aware)
│   ├── orchestrator/
│   │   └── pipeline_v4.py        ← Génération Gold Dataset (QA + CriticV4)
│   └── critic_v4/
│       ├── question_evaluator.py ← Orchestrateur Phase 1
│       ├── metrics/
│       │   ├── contextual_answerability.py
│       │   ├── pedagogical_value.py
│       │   ├── answer_completeness.py
│       │   ├── answer_anchoring.py
│       │   └── difficulty_grader.py  ← Phase 3 : niveau Bloom (1–5) [nouveau]
│       └── prompts/
│           ├── contextual_answerability_prompt.py
│           ├── pedagogical_value_prompt.py
│           ├── answer_completeness_prompt.py
│           ├── answer_anchoring_prompt.py
│           └── difficulty_grader_prompt.py        [nouveau]
│
├── evaluation/
│   ├── run_evaluation.py         ← Évaluation RAG complète
│   ├── rag_retriever.py          ← Retriever (bge-m3 + ChromaDB)
│   ├── rag_generator.py          ← Générateur RAG (Qwen2.5-32B)
│   ├── metrics.py                ← Hit Rate, MRR, ROUGE-L, BERTScore, Judge
│   └── repair_judge_errors.py    ← Réparer les erreurs de juge sans tout relancer
│
├── results_final/
│   ├── RAPPORT_PROJET.md         ← Rapport d'analyse complet
│   ├── mi201/                    ← Gold dataset + résultats évaluation MI201
│   └── physique_dunod/           ← Gold dataset + résultats évaluation Physique
│
├── requirements.txt              ← Dépendances Python (versions testées)
└── data/
    └── pdfs/                     ← Placer les PDF ici (gitignored)
```

---

## Prérequis

### Matériel

- GPU avec **≥ 40 GB VRAM** (testé : NVIDIA L40S 46 GB)
- CUDA 12+

### Modèles à télécharger manuellement

Placer dans `~/models/` :

```
~/models/deepseek-r1-distill-qwen-32b/DeepSeek-R1-Distill-Qwen-32B-IQ3_M.gguf
~/models/qwen2.5-32b-instruct/Qwen2.5-32B-Instruct-Q4_K_M.gguf
~/models/bge-m3/                   ← BAAI/bge-m3 (depuis HuggingFace)
```

### Environnement Python

```bash
# Créer un venv
python3 -m venv ~/envs/agentic_ai
source ~/envs/agentic_ai/bin/activate

# Installer les dépendances standard
pip install -r requirements.txt

# llama-cpp-python DOIT être installé avec CUDA
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python==0.3.16 --no-cache-dir
```

---

## Lancer le pipeline

### En local (interactif)

```bash
source ~/envs/agentic_ai/bin/activate
cd /chemin/vers/Agentic_AI

# Pipeline complet sur un PDF
python3 Projet_Pipeline.py \
    --pdf "data/pdfs/MonCours.pdf" \
    --output-base "output/moncours" \
    --num-chunks 200 \
    --seed 42
```

### Sur cluster SLURM (ENSTA)

```bash
# Editer run_physique_pipeline.sbatch pour changer le PDF et output-base
sbatch run_physique_pipeline.sbatch

# Suivre les logs
tail -f logs/physique_pipeline_<JOB_ID>.log
```

### Options du pipeline

| Option | Description | Défaut |
|---|---|---|
| `--pdf` | Chemin vers le PDF | obligatoire |
| `--output-base` | Dossier de sortie | `output/pipeline` |
| `--num-chunks` | Nombre de chunks à sélectionner | `200` |
| `--seed` | Reproductibilité de la sélection | `42` |
| `--skip-gold` | Sauter la génération Gold | — |
| `--skip-eval` | Sauter l'évaluation RAG | — |
| `--chunks-only` | Chunking uniquement | — |
| `--chunks-file` | Utiliser un fichier chunks existant | — |
| `--gold-file` | Utiliser un Gold Dataset existant | — |

### Fichiers produits

```
output/moncours/
├── chunks.json                         ← chunks extraits du PDF
├── gold_dataset.jsonl                  ← paires QA générées
├── gold_dataset.json                   ← idem en JSON lisible
├── gold_dataset_with_difficulty.jsonl  ← enrichi avec niveaux Bloom (optionnel)
└── evaluation/
    ├── detailed_<timestamp>.json       ← résultats détaillés par QA
    ├── summary_<timestamp>.json        ← métriques agrégées
    ├── rapport_lisible_<timestamp>.txt ← rapport lisible
    └── incremental_results.jsonl       ← sauvegarde progressive (reprise)
```

---

## Phase 3 — Évaluation de la difficulté cognitive (Bloom)

Le `DifficultyGrader` enrichit chaque entrée du Gold Dataset avec un niveau de difficulté
cognitive selon la **taxonomie de Bloom révisée** (5 niveaux).

| Niveau | Label | Opération cognitive |
|---|---|---|
| 1 | Factuel | Rappel direct d'une définition |
| 2 | Compréhension | Reformulation / explication |
| 3 | Application | Utilisation d'une méthode ou formule |
| 4 | Analyse | Comparaison, raisonnement causal |
| 5 | Synthèse | Connexion multi-concepts, évaluation critique |

### Activer inline (lors de la génération)

```python
config = PipelineV4Config(
    chunks_path="...",
    output_path="...",
    enable_difficulty_grading=True   # active la Phase 3
)
```

### Post-annoter un dataset existant

```bash
source ~/envs/agentic_ai/bin/activate
cd /chemin/vers/Agentic_AI

python3 scripts/annotate_difficulty.py \
    --gold  output/gold_dataset.jsonl \
    --chunks data/chunks/chunks_mi201.json \
    --output output/gold_dataset_with_difficulty.jsonl
```

### Résultats sur MI201 (84 entrées)

```
Level 2 — Compréhension :  7  ( 8.3%)
Level 3 — Application   : 30  (35.7%)
Level 4 — Analyse       : 47  (56.0%)
```

---

## Format du Gold Dataset

Le Gold Dataset produit est un fichier JSONL (une entrée par ligne) :

```jsonl
{
  "question":               "Expliquez pourquoi ...",
  "answer":                 "L'énergie potentielle est définie par ...",
  "chunk_id":               "2.1.4.c1",
  "chapter":                "DEUXIÈME PÉRIODE",
  "section":                "Mécanique 2",
  "page_range":             [580, 609],
  "global_score":           0.85,
  "difficulty_level":       4,
  "difficulty_label":       "Analyse",
  "difficulty_justification": "La question nécessite de comparer deux approches..."
}
```

**Lecture du `chunk_id`** : `2.1.4.c1` = Chapitre 2, Section 1, Sous-section 4, Chunk 1.

**Champs obligatoires pour l'évaluation** : `question`, `answer`, `chunk_id`.
Les champs `difficulty_*` sont optionnels (présents si Phase 3 activée).

---

## Architecture technique

| Composant | Modèle | Rôle |
|---|---|---|
| Gold QA Generator | DeepSeek-R1-Distill-Qwen-32B (IQ3_M) | Génère questions + réponses |
| CriticV4 Phase 1 | DeepSeek-R1-Distill-Qwen-32B (IQ3_M) | Valide qualité de la question |
| CriticV4 Phase 2 | DeepSeek-R1-Distill-Qwen-32B (IQ3_M) | Valide complétude + ancrage réponse |
| CriticV4 Phase 3 | DeepSeek-R1-Distill-Qwen-32B (IQ3_M) | Niveau difficulté Bloom 1–5 |
| Embedder | BAAI/bge-m3 (local, 1024-dim) | Encode chunks et questions |
| RAG Generator | Qwen2.5-32B-Instruct (Q4_K_M) | Génère les réponses RAG |
| LLM Judge | DeepSeek-R1-Distill-Qwen-32B (IQ3_M) | Note les réponses RAG (0–5) |
| Vector Store | ChromaDB (in-memory) | Stocke les embeddings |
| BERTScore | xlm-roberta-large | Similarité sémantique |

Budget VRAM : ~40 GB (les modèles ne sont jamais chargés simultanément).

---

## Branches actives

| Branche | Contributeur | Contenu |
|---|---|---|
| `Aziz_branch` | Ben Amira Aziz | Pipeline principal, CriticV4 Phase 1 & 2, évaluation RAG |
| `Ghozzi_branch` | Ghozzi | CriticV4 Phase 3 (DifficultyGrader), script annotate_difficulty |
| `Yassine_branch` | Zanned Yassine | Pipeline SoG, graphe de connaissances |
| `Seif_branch` | Seif | Validation symbolique déterministe (hard rules) |
| `Ameni_branch` | Ameni | — |
| `Maloe_branch` | Maloe | — |
