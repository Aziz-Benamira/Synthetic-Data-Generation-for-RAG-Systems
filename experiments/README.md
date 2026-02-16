# 🧪 Expériences Critic V2 - Guide d'Utilisation

## 📁 Structure des Expériences

```
experiments/
├── critic_v2_baseline/          # Expérience 1: Baseline avec config par défaut
│   ├── 01_extract_chunks.py     # Extraction chunks du M2
│   ├── 02_generate_qa_samples.py # Génération QA
│   ├── 03_run_critic_v2.py      # Évaluation Critic V2
│   ├── 04_analyze_results.py    # Analyse et rapport
│   ├── run_experiment.sh        # Pipeline complète
│   ├── config.json              # Configuration expérience
│   └── data/                    # Données générées
│       ├── chunks.json
│       └── qa_samples.json
│   └── results/                 # Résultats
│       ├── detailed_results.json
│       ├── summary_stats.json
│       ├── metrics_breakdown.json
│       └── analysis_report.md   # 📊 RAPPORT FINAL
│   └── logs/
│       └── experiment.log       # Logs détaillés
```

## 🚀 Quick Start

### Prérequis

1. **LLMs en cours d'exécution**:
   ```bash
   # Terminal 1: Ollama (pour génération QA)
   ollama serve
   
   # Terminal 2: llama.cpp (pour Critic V2 avec DeepSeek R1)
   cd ~/llama.cpp
   ./llama-server -m ~/models/deepseek-r1-distill-qwen-32b/*.gguf --port 8080 -ngl 99
   ```

2. **Document source**:
   - Placer `tipler_mosca_chapitre_m2.pdf` dans `data/pdfs/`

### Exécution

#### Option 1: Pipeline complète (recommandée)
```bash
cd experiments/critic_v2_baseline
./run_experiment.sh all
```

#### Option 2: Étape par étape
```bash
cd experiments/critic_v2_baseline

# Étape 1: Extraire 8 chunks variés
./run_experiment.sh 1

# Étape 2: Générer 3 QA par chunk (24 total)
./run_experiment.sh 2

# Étape 3: Évaluer avec Critic V2 (4 métriques)
./run_experiment.sh 3

# Étape 4: Analyser et générer le rapport
./run_experiment.sh 4
```

#### Option 3: Scripts Python individuels
```bash
python 01_extract_chunks.py
python 02_generate_qa_samples.py
python 03_run_critic_v2.py
python 04_analyze_results.py
```

## 📊 Résultats

### Fichiers générés

1. **`results/detailed_results.json`**: Résultats détaillés pour chaque QA
   - Scores par métrique
   - Raisonnement du LLM
   - Décision finale (pass/reject/improve)

2. **`results/summary_stats.json`**: Statistiques agrégées
   - Taux de passage
   - Moyennes par métrique
   - Distribution des scores

3. **`results/metrics_breakdown.json`**: Détails par métrique
   - Tous les scores d'une métrique
   - Raisonnements

4. **`results/analysis_report.md`**: 📄 **RAPPORT FINAL**
   - Résumé exécutif
   - Performance par métrique
   - Cas limites intéressants
   - Recommandations pour calibration

### Interpréter les résultats

```python
# Seuils de décision (par défaut)
score < 0.3   → REJECT    (Mauvais)
0.3-0.5       → IMPROVE   (Médiocre) 
score >= 0.5  → PASS      (Acceptable)

# Bandes de score
0.0-0.3       → BAD
0.3-0.5       → MEDIOCRE
0.5-0.7       → ACCEPTABLE
0.7-0.85      → GOOD
0.85-1.0      → EXCELLENT
```

## 🔧 Configuration

### Modifier les seuils

Éditer `config.json`:
```json
{
  "critic_v2_config": {
    "reject_threshold": 0.3,    // Seuil de rejet
    "pass_threshold": 0.5,      // Seuil de passage
    "metrics": {
      "anchoring": {
        "weight": 2.0,          // Poids dans score global
        "pass_threshold": 0.5   // Seuil individuel
      }
    }
  }
}
```

### Créer une nouvelle expérience

```bash
# Copier le template
cp -r experiments/critic_v2_baseline experiments/critic_v2_strict

# Modifier la config
nano experiments/critic_v2_strict/config.json

# Exécuter
cd experiments/critic_v2_strict
./run_experiment.sh all
```

## 📈 Workflow d'Analyse

1. **Exécuter l'expérience**: `./run_experiment.sh all`
2. **Lire le rapport**: `cat results/analysis_report.md`
3. **Examiner les cas limites** dans `detailed_results.json`
4. **Identifier les patterns**:
   - Métriques trop strictes/permissives?
   - Types de QA problématiques?
   - Corrélations entre métriques?
5. **Ajuster la configuration**
6. **Réexécuter et comparer**

## 🎯 Objectifs de l'Expérience Baseline

- [ ] Vérifier que les 4 métriques s'exécutent sans erreur
- [ ] Obtenir une distribution de scores variée (pas tous 0 ou 1)
- [ ] Identifier les métriques trop strictes/permissives
- [ ] Valider que le feedback est actionnable
- [ ] Calibrer les seuils selon les résultats observés

## 🔄 Prochaines Expériences

Après la baseline:

1. **`critic_v2_strict/`**: Seuils plus élevés (0.4, 0.6)
2. **`critic_v2_lenient/`**: Seuils plus bas (0.2, 0.4)
3. **`critic_v2_comparison/`**: Comparaison avec ancien critic
4. **`critic_v2_fewshot/`**: Optimisation des few-shot examples
5. **`critic_v2_weights/`**: Calibration des poids

## 📝 Logs

Tous les logs détaillés dans `logs/experiment.log`:
```bash
tail -f logs/experiment.log  # Suivre en temps réel
```

## ⚠️ Troubleshooting

### Erreur "llama-server not responding"
```bash
# Vérifier que llama-server est lancé sur le bon port
curl http://localhost:8080/v1/models
```

### Erreur "Ollama not found"
```bash
# Vérifier Ollama
ollama list
ollama pull qwen2.5:32b-instruct-q4_K_M
```

### Parsing PDF échoue
```bash
# Vérifier que le PDF existe
ls -lh data/pdfs/tipler_mosca_chapitre_m2.pdf
```

## 📚 Documentation Complémentaire

- [config.json](experiments/critic_v2_baseline/config.json): Configuration détaillée
- [README.md](experiments/critic_v2_baseline/README.md): Description de l'expérience
- [src/critic_v2/](src/critic_v2/): Code source du Critic V2
- [RESEARCH_EVALUATION_METRICS.md](docs/RESEARCH_EVALUATION_METRICS.md): Recherche sur les patterns

---

**Questions? Problèmes?** Consulter les logs ou ouvrir une issue.
