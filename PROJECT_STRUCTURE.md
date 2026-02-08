# 📁 Structure du Projet - Agentic AI

> Réorganisé le 4 février 2026 pour un workspace propre et professionnel

## 🗂️ Architecture du Répertoire

```
Agentic_AI/
├── 📄 README.md                          # Documentation principale
├── 📄 PROJECT_ARCHITECTURE.md            # Architecture technique du système
├── 📄 RESEARCH_EVALUATION_METRICS.md     # Recherche sur Ragas/G-Eval/Nvidia
├── 📄 CONTRIBUTING.md                    # Guidelines de contribution
├── 📄 requirements.txt                   # Dépendances Python
├── 📄 setup.py                          # Configuration d'installation
├── 📄 setup_gpu.ps1                     # Script setup GPU (Windows)
│
├── 📂 src/                              # Code source principal
│   ├── agents/                          # Agents (critic, question_gen, answer_gen)
│   ├── pipeline/                        # Pipeline de génération QA
│   ├── utils/                           # Utilitaires
│   └── config/                          # Configurations
│
├── 📂 tests/                            # Tests unitaires et d'intégration
│   ├── test_*.py                        # 24 fichiers de test
│   └── verify_*.py                      # Scripts de vérification
│
├── 📂 scripts/                          # Scripts d'analyse et d'exécution
│   ├── analyze_*.py                     # Scripts d'analyse
│   ├── run_*.py                         # Scripts d'exécution
│   ├── config_*.py                      # Configurations
│   └── quick_*.py                       # Tests rapides
│
├── 📂 data/                             # Données
│   ├── pdfs/                            # PDFs sources
│   └── README.md                        # Description des données
│
├── 📂 docs/                             # Documentation détaillée
│   ├── COMMANDES_TEST.md                # Commandes de test
│   ├── COMPARISON.md                    # Comparaisons
│   ├── DISCUSSION_MALOE_INTEGRATION.md  # Discussions techniques
│   ├── LITERATURE_REVIEW_SUMMARY.md     # Revue de littérature
│   ├── PLAN_IMPLEMENTATION_HYBRID.md    # Plans d'implémentation
│   ├── RAPPORT_FINAL_CRITIC.md          # Rapport final critic
│   ├── READY_FOR_TUTOR.md              # Rapports pour tuteurs
│   ├── SEIF_*.md                        # Documentation Seif
│   ├── SYSTEM_PRESENTATION.md           # Présentation système
│   └── CLUSTER_GUIDE_FOR_BEGINNERS.md   # Guide cluster
│
├── 📂 experiments/                      # Expérimentations
│   └── ...                              # Notebooks et expériences
│
├── 📂 output/                           # Sorties actuelles
│   └── test_pipeline/                   # Résultats du pipeline
│
├── 📂 logs/                             # Logs d'exécution
│
├── 📂 cluster_utils/                    # Utilitaires cluster ENSTA
│
├── 📂 maloe_metrics/                    # Métriques MALOE
│
└── 📂 archives/                         # Archives (anciens fichiers)
    ├── old_results/                     # Anciens résultats JSON
    ├── old_outputs/                     # Anciens dossiers output_*
    ├── enstrag-main/                    # Ancien code ENSTRAG
    ├── seif_changes_review/             # Revue changements Seif
    └── README.md                        # Documentation archives
```

## 🎯 Dossiers Clés

| Dossier | Description | Utilisation |
|---------|-------------|-------------|
| **src/** | Code source | Développement principal |
| **tests/** | Tests | Validation du code |
| **scripts/** | Scripts | Analyse et exécution |
| **data/** | Données | PDFs et datasets |
| **docs/** | Documentation | Rapports et guides |
| **experiments/** | Expérimentations | Notebooks et essais |
| **archives/** | Archives | Anciens fichiers (référence) |

## 🚀 Points d'Entrée Principaux

### Documentation
- **README.md** : Vue d'ensemble du projet
- **PROJECT_ARCHITECTURE.md** : Architecture technique
- **RESEARCH_EVALUATION_METRICS.md** : Recherche sur les métriques

### Code
- **src/pipeline/** : Pipeline de génération QA
- **src/agents/** : Agents (critic, générateurs)

### Tests
- **tests/test_pipeline.py** : Test du pipeline complet
- **tests/verify_local_setup.py** : Vérification setup

### Scripts
- **scripts/run_demo.py** : Démo du système
- **scripts/analyze_*.py** : Analyses de résultats

## 📝 Conventions

### Fichiers à Garder dans le Root
- README.md (principal)
- PROJECT_ARCHITECTURE.md
- RESEARCH_EVALUATION_METRICS.md
- CONTRIBUTING.md
- requirements.txt, setup.py, setup_gpu.ps1

### Fichiers à Ranger
- **Tests** → `tests/`
- **Scripts** → `scripts/`
- **Documentation** → `docs/`
- **Résultats anciens** → `archives/`

## 🧹 Maintenance

Cette structure a été établie pour maintenir un workspace propre. Les règles :
1. **Root = Fichiers essentiels uniquement**
2. **Tests dans tests/**
3. **Scripts dans scripts/**
4. **Documentation dans docs/**
5. **Anciens fichiers dans archives/**

---

*Dernière mise à jour : 4 février 2026*
