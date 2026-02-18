# Critic V4 - Métriques Phase 1

## Vue d'ensemble

Les métriques Phase 1 du Critic V4 filtrent les questions **AVANT** la génération des réponses, économisant du temps de calcul en rejetant les questions problématiques tôt dans le pipeline.

## Architecture 2-Phases

```
┌─────────────────────────────────────────────────────────────┐
│                        CRITIC V4                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  PHASE 1: Question Filtering (ce module)                    │
│  ┌────────────────────────────────────────────────────┐    │
│  │ 1. Contextual Answerability                        │    │
│  │    ├─ Chunk contient-il les informations?          │    │
│  │    ├─ Score: 0 → 3                                 │    │
│  │    └─ Seuil: ≥2.0 pour PASS                        │    │
│  │                                                     │    │
│  │ 2. Pedagogical Value                               │    │
│  │    ├─ tests_understanding (compréhension?)         │    │
│  │    ├─ non_trivial (non-triviale?)                  │    │
│  │    ├─ educational_utility (valeur éducative?)      │    │
│  │    └─ Seuil: ≥0.67 pour PASS (2/3 critères)        │    │
│  └────────────────────────────────────────────────────┘    │
│                          ↓                                   │
│                    Si PASS → Phase 2                         │
│                    Si REJECT → Nouvelle question             │
│                                                              │
│  PHASE 2: Answer Validation (à implémenter)                 │
│  ┌────────────────────────────────────────────────────┐    │
│  │ 3. Answer Completeness                             │    │
│  │ 4. Answer Anchoring                                │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Métriques Phase 1

### 1. Contextual Answerability

**Objectif**: Vérifier si le chunk contient suffisamment d'informations pour répondre à la question.

**Scoring**:
- **0**: Aucune information pertinente
- **1**: Informations partielles et insuffisantes
- **2**: Informations suffisantes mais incomplètes
- **3**: Informations complètes et précises

**Seuil**: ≥2.0 pour PASS

**Exemple PASS**:
```
Question: "Pourquoi la validation croisée est-elle préférable à un simple train/test split?"
Chunk: [Contient une explication détaillée de la validation croisée et ses avantages]
→ Score: 3/3 → PASS
```

**Exemple REJECT**:
```
Question: "Quels sont les avantages des réseaux de neurones profonds?"
Chunk: [Parle de l'apprentissage supervisé mais pas de réseaux de neurones]
→ Score: 0/3 → REJECT
```

**Utilisation**:
```python
from src.critic_v4.metrics import ContextualAnswerability
from llama_cpp import Llama

llm = Llama(model_path="...", n_gpu_layers=-1)
evaluator = ContextualAnswerability(llm=llm, temperature=0.1)

result = evaluator.evaluate(
    chunk_content="...",
    question="..."
)

print(result["decision"])  # "pass" ou "reject"
print(result["score"])     # 0.0 à 3.0
print(result["passages_pertinents"])  # Passages extraits du chunk
print(result["feedback"])  # Feedback pour le Question Generator
```

---

### 2. Pedagogical Value

**Objectif**: Évaluer la qualité pédagogique de la question pour détecter les questions circulaires, triviales, ou sans valeur éducative.

**Critères binaires (True/False)**:
1. **tests_understanding**: Teste la compréhension conceptuelle? (pas juste du copier-coller)
2. **non_trivial**: Question non-triviale? (pas une évidence)
3. **educational_utility**: Valeur éducative? (aide à maîtriser un concept clé)

**Scoring**: count(True) / 3

**Seuil**: ≥0.67 pour PASS (au moins 2/3 critères)

**Exemple PASS**:
```
Question: "Pourquoi la phase de validation est-elle essentielle dans l'apprentissage supervisé?"
→ tests_understanding: True (demande de comprendre le rôle)
→ non_trivial: True (nécessite une réflexion)
→ educational_utility: True (concept clé)
→ Score: 3/3 = 1.0 → PASS
```

**Exemple REJECT**:
```
Question: "Qu'est-ce que l'apprentissage supervisé?" (quand la définition exacte est dans le chunk)
→ tests_understanding: False (circulaire, juste répéter la définition)
→ non_trivial: True
→ educational_utility: True
→ Score: 2/3 = 0.67 → BORDERLINE (peut passer ou non selon le contexte)
```

**Utilisation**:
```python
from src.critic_v4.metrics import PedagogicalValue
from llama_cpp import Llama

llm = Llama(model_path="...", n_gpu_layers=-1)
evaluator = PedagogicalValue(llm=llm, temperature=0.1)

result = evaluator.evaluate(
    chunk_content="...",
    question="..."
)

print(result["decision"])  # "pass" ou "reject"
print(result["score"])     # 0.0 à 1.0
print(result["criteria"])  # {"tests_understanding": True, ...}
print(result["suggestions"])  # Suggestions d'amélioration
```

## Tests

### Lancer les tests unitaires

```bash
# Sur GPU (recommandé)
sbatch run_test_phase1_metrics.sbatch

# Ou directement
python3 test_phase1_metrics.py
```

Les tests vérifient:
- ✅ Fonctionnement des deux métriques
- ✅ Parsing JSON des réponses LLM
- ✅ Calcul des scores et décisions
- ✅ Génération de feedback

## Prochaines étapes

1. ✅ **Étape 1**: ScopedMemory (complété)
2. ✅ **Étape 2**: Métriques Phase 1 (complété)
3. **Étape 3**: QuestionEvaluator (orchestrateur Phase 1)
4. **Étape 4**: CriticV4 (intégration 2 phases complètes)
5. **Étape 5**: QuestionGeneratorV3 (avec ScopedMemory)
6. **Étape 6**: Workflow update (02_generate_qa_samples.py)
7. **Étape 7**: Tests end-to-end

## Références

- [ARCHITECTURE_DECISIONS.md](/home/ensta/ensta-ben-amira/projects/Agentic_AI/docs/ARCHITECTURE_DECISIONS.md): ADR-001 à ADR-004
- [ScopedMemory](/home/ensta/ensta-ben-amira/projects/Agentic_AI/src/utils/scoped_memory.py): Prévention de redondance
- Critic V2 baseline (pour comparaison)
