# Synthetic Data Generation for RAG Systems

> **ENSTA Paris — Projet IA Agentique 2025**

Génération automatique de paires Question-Réponse (QA) de haute qualité à partir de documents PDF scientifiques, avec évaluation critique multi-agent et intégration RAG.

## Contributeurs

| Membre | Module | Dossier |
|--------|--------|---------|
| **Aziz Ben Amira** | Pipeline principal de génération QA (agents, chunking, critic, orchestrateur, évaluation RAG) | `pipeline/` |
| **Yassine Jmal** | Synthesize-on-Graph (SoG) — génération par graphes de contexte, multi-hop, cross-document | `sog/` |
| **Ameni Zouaoui** | RAGEval — adaptation du framework d'évaluation RAG | `rageval/` |
| **Maloé Musik** | Métriques d'évaluation (traditionnelles + LLM-as-judge) | `metrics/` |
| **Seif Briki** | Chunking multimodal (images/captions) + estimation de difficulté des questions | `multimodal/` |

## Structure du projet

```
├── pipeline/                  # [Aziz] Pipeline de génération QA
│   ├── src/                   # Code source principal
│   │   ├── agents/            # Générateur de questions, générateur de réponses, agent critique
│   │   ├── chunking/          # Semantic chunking des PDFs
│   │   ├── critic_v2/         # Métriques critic v2 (anchoring, clarity, completeness)
│   │   ├── critic_v4/         # Critic v4 avec retry loop et hard rules
│   │   ├── llm/               # Abstraction LLM (Ollama, OpenRouter, HuggingFace)
│   │   ├── orchestrator/      # Pipeline orchestrator (v1 à v4)
│   │   ├── parsers/           # PDF parsing (ENSTA parser)
│   │   └── utils/             # Utilitaires (clients API, mémoire)
│   ├── evaluation/            # Pipeline d'évaluation RAG (Hit@k, MRR, ROUGE-L, BERTScore, LLM Judge)
│   ├── experiments/           # Baselines et expériences (critic_v2_baseline)
│   └── scripts/               # Scripts SLURM sbatch
│
├── sog/                       # [Yassine] Synthesize-on-Graph
│   ├── src/                   # Modules SoG (context_graph, cross_document_sampling, entity_extraction)
│   ├── experiment/            # Intégration SoG + Pipeline Aziz (pipeline_v4_sog, sog_retriever)
│   ├── outputs/               # Graphes de contexte et QA générés
│   └── inputs/                # PDFs source
│
├── rageval/                   # [Ameni] RAGEval adapté
│   └── RAGEval-main/rageval/  # Framework complet (qar_generation, prompts, évaluation)
│
├── metrics/                   # [Maloé] Suite de métriques d'évaluation
│   ├── evaluator.py           # Orchestrateur de métriques
│   ├── trad_metrics.py        # Métriques traditionnelles (ROUGE, BLEU, BERTScore)
│   ├── llm_metrics.py         # LLM-as-Judge
│   ├── test_sets/             # Gold datasets (physique, chimie, ML, géométrie algébrique)
│   └── tests/                 # Notebooks de test + RAGAgent
│
├── multimodal/                # [Seif] Multimodal + Difficulté
│   ├── src/                   # Extracteur d'images, VL captioner
│   ├── difficulty/            # Estimateur de difficulté, classifieur de type, diversity manager
│   ├── scripts/               # Scripts d'annotation et de démo
│   └── output/                # Datasets annotés avec difficulté
│
├── data/pdfs/                 # Documents PDF partagés
├── docs/papers/               # Articles de référence
├── requirements.txt
└── setup.py
```

## Exécution

### Prérequis
- Python 3.10+
- GPU NVIDIA (L40S recommandé pour les modèles 32B)
- Ollama (pour l'inférence locale des LLMs)

### Installation
```bash
pip install -r requirements.txt
```

### Pipeline principal (Aziz)
```bash
# Lancer la génération QA complète sur un PDF
python pipeline/run_pipeline_v4_full.py --pdf data/pdfs/MI201_2022_poly.pdf

# Évaluation RAG
python pipeline/evaluation/run_evaluation.py --config config.json
```

### SoG — Synthesize-on-Graph (Yassine)
```bash
# Construire le graphe de contexte et générer des QA multi-hop
python sog/experiment/run_full_pipeline_sog.py \
    --chunks sog/experiment/data/chunks_mi201.json \
    --graph sog/experiment/data/MI201_2022_poly_context_graph.json
```

### Métriques d'évaluation (Maloé)
```bash
python metrics/evaluator.py --dataset metrics/test_sets/gold_dataset_v4_full.jsonl
```

## Modèles utilisés

| Modèle | Usage |
|--------|-------|
| DeepSeek-R1-Distill-Qwen-32B (IQ3_M) | Génération + Évaluation critique |
| Qwen2.5-32B-Instruct (Q4_K_M) | Génération de réponses RAG |
| BAAI/bge-m3 | Embeddings pour retrieval RAG |
| BAAI/bge-small-en-v1.5 | Embeddings pour SoG retrieval |

## Résultats clés

**Pipeline Critic v4** : Score moyen critic 0.800/1.0 sur MI201 (10 paires gold)

**Évaluation RAG (plain RAG vs QA gold)** :

| Métrique | MI201 (10) | Tipler (74) |
|----------|-----------|-------------|
| Hit@1 | 0.600 | 0.770 |
| Hit@5 | 0.900 | 0.959 |
| MRR | 0.725 | 0.848 |
| ROUGE-L | 0.365 | 0.201 |
| BERTScore | 0.908 | 0.846 |
| LLM Judge | 3.80/5 | 3.755/5 |
