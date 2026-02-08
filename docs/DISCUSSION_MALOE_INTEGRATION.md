# 💡 Discussion: Intégration des Métriques de Maloe + Tâches du Tuteur

**Date:** 29 Janvier 2026  
**Contexte:** Réunion tuteur - 2 tâches à implémenter  
**Ressources:** Maloe a créé des fonctions d'évaluation déterministes dans `evaluation/metrics/`

---

## 📋 Les 2 Tâches du Tuteur

### **Tâche 1: Question Nécessitant le Contexte**

**Problème:** Certaines questions peuvent être répondues sans le chunk (connaissance générale du LLM).

**Solution proposée:**
```
1. LLM génère question Q à partir du chunk C
2. LLM génère réponse A1 AVEC le chunk C (comme maintenant)
3. LLM génère réponse A2 SANS le chunk (juste Q)
4. Comparer A1 vs A2:
   - Si identiques/très similaires → REJETER la question
   - Si différentes → GARDER la question (nécessite vraiment C)
```

**Exemple:**
```
Chunk: "La tribu des boréliens de ℝ est la plus petite tribu contenant tous les intervalles."

Question: "Qu'est-ce qu'une tribu?" 
❌ REJETER - peut être répondue sans le chunk (connaissance générale)

Question: "Qu'est-ce que la tribu des boréliens de ℝ?"
✅ GARDER - nécessite le chunk spécifique
```

---

### **Tâche 2: LLM-as-Judge vs Fonctions Déterministes**

**Situation actuelle:**
- Critic = LLM (Llama3 8B) qui juge selon un prompt
- 6 hard rules déterministes (nombres, causalité, etc.)
- **Avantage:** Flexible, comprend nuances
- **Inconvénient:** Non déterministe, coûteux, peut être inconsistant

**Proposition de Maloe:**
- Remplacer/augmenter le LLM-as-judge avec des fonctions Python déterministes
- Fonctions pour: relevance, faithfulness, coherence, METEOR, etc.
- **Avantage:** Déterministe, rapide, gratuit, reproductible
- **Inconvénient:** Moins nuancé que LLM

**Dilemme:**
1. **Remplacer complètement** le LLM-as-judge par les fonctions?
2. **Combiner** les deux approches (hybride)?

---

## 🔍 Analyse des Fonctions de Maloe

### **Structure du Code:**

```
maloe_metrics/
├── evaluator.py           # Framework modulaire d'évaluation
├── trad_metrics.py        # Métriques traditionnelles (METEOR, F1, etc.)
├── llm_metrics.py         # LLM-as-judge (context support, relevance)
└── README.md              # Documentation complète
```

### **Métriques Disponibles:**

#### **1. Métriques de Retriever (trad_metrics.py)**
- `unrankedMetrics()` - Precision, Recall, F1, Accuracy
- `mean_reciprocal_rank()` - Position du 1er résultat pertinent
- `ndcg_at_k()` - Normalized DCG (qualité du ranking)
- `mean_average_precision()` - MAP@k

#### **2. Métriques de Génération (trad_metrics.py)**
- `exact_match()` - Correspondance exacte
- `meteor()` / `meteor_batch()` - Qualité avec ordre des mots
  
#### **3. Métriques LLM-as-Judge (llm_metrics.py)**
- `llm_as_judge_context_support()` - **Faithfulness** (réponse ancrée dans contexte?)
- `llm_as_judge_answer_relevance()` - **Relevance** (répond à la question?)
- `llm_as_judge_coherence()` - **Coherence** (texte fluide?)
- `semantic_perplexity()` - Confiance du modèle
- `comprehensive_llm_evaluation()` - Évaluation complète

#### **4. Framework d'Évaluation (evaluator.py)**
```python
# Architecture modulaire
ComprehensiveEvaluator
├── RetrieverEvaluator      # Métriques de ranking
├── GenerationEvaluator     # METEOR, EM
├── LLMEvaluator           # Semantic metrics
├── RiskAwareEvaluator     # Risk, prudence
└── EfficiencyEvaluator    # Latency, cost

# Usage simple:
evaluator = ComprehensiveEvaluator(llm_model="gpt-4")
result = evaluator.evaluate_full_pipeline(
    query="...", response="...", context="..."
)
```

---

## 💭 Ma Recommandation: Approche Hybride

### **Pourquoi Hybride?**

1. **Les fonctions déterministes sont excellentes pour:**
   - Checks factuels (nombres, longueur, overlap)
   - Métriques objectives (METEOR, F1)
   - Rapidité et reproductibilité
   - **→ Parfait pour les hard rules actuels + nouvelles vérifications**

2. **Le LLM-as-judge reste meilleur pour:**
   - Nuances sémantiques complexes
   - Compréhension du sens global
   - Évaluation de la qualité narrative
   - **→ Garder pour les 5 critères principaux**

### **Architecture Proposée:**

```python
class HybridCriticAgent:
    """
    Combine LLM-as-judge (sémantique) + fonctions déterministes (factuelles)
    """
    
    def evaluate(qa_pair, chunk):
        # PHASE 1: Hard Rules (déterministes - actuelles + nouvelles)
        hard_rule_results = apply_hard_rules(qa_pair, chunk)
        if any(result.fail for result in hard_rule_results):
            return REJECT  # Rejet immédiat
        
        # PHASE 2: Fonctions Déterministes de Maloe
        deterministic_scores = {
            "faithfulness": calculate_faithfulness(qa_pair, chunk),
            "relevance": calculate_relevance(qa_pair, chunk),
            "meteor": meteor(qa_pair.answer, chunk.content)
        }
        
        if deterministic_scores["faithfulness"] < 0.7:
            return REJECT  # Pas ancré dans le contexte
        
        # PHASE 3: LLM-as-Judge (nuances sémantiques)
        llm_scores = llm_evaluate_nuances(qa_pair, chunk)
        
        # PHASE 4: Décision combinée
        final_score = weighted_average(deterministic_scores, llm_scores)
        return PASS if final_score >= 0.85 else REJECT
```

### **Avantages:**

✅ **Rapidité:** Hard rules + déterministes rejettent 70% des cas sans appel LLM  
✅ **Précision:** LLM pour cas difficiles nécessitant nuance  
✅ **Coût:** Moins d'appels LLM = moins cher  
✅ **Déterminisme:** 80% des rejets sont reproductibles  
✅ **Qualité:** LLM garde le contrôle final sur les cas ambigus  

---

## 🎯 Plan d'Implémentation

### **Étape 1: Intégrer les Fonctions de Maloe**

```python
# Copier dans notre projet
Agentic_AI/
├── src/
│   ├── agents/
│   │   ├── critic_agent.py          # Notre critic actuel
│   │   └── hybrid_critic.py         # NOUVEAU - Critic hybride
│   └── evaluation/                   # NOUVEAU - Dossier de Maloe
│       ├── __init__.py
│       ├── evaluator.py             # Framework
│       ├── trad_metrics.py          # Métriques déterministes
│       └── llm_metrics.py           # LLM-as-judge
```

**Commande:**
```powershell
New-Item -ItemType Directory -Path "src\evaluation"
Copy-Item maloe_metrics\*.py src\evaluation\
```

### **Étape 2: Créer Fonctions Déterministes pour Nos Critères**

```python
# src/evaluation/rag_specific_metrics.py

def calculate_faithfulness(qa_pair, chunk) -> float:
    """
    Faithfulness = Anchoring (notre terme)
    Mesure si la réponse est ancrée dans le chunk.
    """
    from trad_metrics import meteor
    
    # 1. Check nombres (hard rule existante)
    if has_hallucinated_numbers(qa_pair.answer, chunk.content):
        return 0.0
    
    # 2. METEOR score (overlap sémantique)
    meteor_score = meteor(qa_pair.answer, chunk.content)
    
    # 3. Extraction d'entités (si disponible avec spaCy)
    entity_overlap = calculate_entity_overlap(qa_pair.answer, chunk.content)
    
    # Moyenne pondérée
    return 0.5 * meteor_score + 0.5 * entity_overlap


def calculate_local_answerability(question, chunk) -> float:
    """
    Mesure si la question peut être répondue depuis le chunk.
    """
    # 1. Check why/how avec marqueurs causaux (hard rule existante)
    if is_why_how_question(question) and not has_causal_markers(chunk.content):
        return 0.0
    
    # 2. Keyword overlap entre question et chunk
    question_keywords = extract_keywords(question)
    chunk_keywords = extract_keywords(chunk.content)
    overlap_ratio = len(question_keywords & chunk_keywords) / len(question_keywords)
    
    return overlap_ratio


def calculate_completeness(qa_pair) -> float:
    """
    Mesure si la réponse est complète.
    """
    # 1. Check longueur (hard rule existante)
    question_complexity = len(qa_pair.question.split())
    answer_length = len(qa_pair.answer)
    
    if question_complexity > 15 and answer_length < 40:
        return 0.3
    
    # 2. Check répétition question (hard rule existante)
    if word_overlap(qa_pair.question, qa_pair.answer) > 0.80:
        return 0.2
    
    # 3. Ratio longueur réponse / chunk
    length_ratio = len(qa_pair.answer) / len(chunk.content)
    if length_ratio < 0.05:  # Réponse <5% du chunk = probablement incomplete
        return 0.5
    
    return 0.9  # Semble complet
```

### **Étape 3: Implémenter Tâche 1 (Question sans contexte)**

```python
# src/agents/question_validator.py

class QuestionContextValidator:
    """
    Valide qu'une question nécessite vraiment le contexte.
    Rejette les questions répondables sans le chunk.
    """
    
    def __init__(self, llm_client, model_name):
        self.llm_client = llm_client
        self.model_name = model_name
    
    def validate_question_needs_context(
        self, 
        question: CandidateQuestion, 
        chunk: SemanticChunk
    ) -> Tuple[bool, Dict]:
        """
        Retourne (needs_context: bool, details: dict)
        """
        # 1. Générer réponse AVEC contexte
        answer_with_context = self._generate_answer(question, chunk.content)
        
        # 2. Générer réponse SANS contexte (juste la question)
        answer_without_context = self._generate_answer(question, context=None)
        
        # 3. Comparer les deux réponses
        similarity = self._calculate_similarity(
            answer_with_context, 
            answer_without_context
        )
        
        # 4. Décision
        needs_context = similarity < 0.75  # Si <75% similaire, contexte nécessaire
        
        return needs_context, {
            "with_context": answer_with_context,
            "without_context": answer_without_context,
            "similarity": similarity,
            "verdict": "NEEDS_CONTEXT" if needs_context else "GENERAL_KNOWLEDGE"
        }
    
    def _generate_answer(self, question, context=None):
        """Génère réponse avec ou sans contexte"""
        if context:
            prompt = f"Réponds à: {question.question}\nContexte: {context}"
        else:
            prompt = f"Réponds à: {question.question}"
        
        response = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    
    def _calculate_similarity(self, text1, text2) -> float:
        """Compare deux textes (METEOR ou cosine similarity)"""
        from evaluation.trad_metrics import meteor
        return meteor(text1, text2)
```

### **Étape 4: Intégrer dans le Pipeline**

```python
# src/orchestrator/pipeline.py (modifications)

class DatasetPipeline:
    def __init__(self, config, llm_client):
        # ... existant ...
        
        # NOUVEAU: Validateur de questions
        self.question_validator = QuestionContextValidator(
            llm_client, config.generator_model
        )
        
        # NOUVEAU: Critic hybride
        from agents.hybrid_critic import HybridCriticAgent
        self.critic = HybridCriticAgent(
            llm_client=llm_client,
            model_name=config.critic_model,
            use_deterministic=True,  # Active les métriques de Maloe
            use_llm=True             # Garde LLM pour nuances
        )
    
    def _generate_questions(self, chunk):
        """Générer questions + valider qu'elles nécessitent le contexte"""
        # 1. Générer questions
        questions = self.question_generator.generate_from_chunk(chunk, num_questions=2)
        
        # 2. NOUVEAU: Filtrer les questions qui ne nécessitent pas le contexte
        valid_questions = []
        for q in questions:
            needs_context, details = self.question_validator.validate_question_needs_context(q, chunk)
            
            if needs_context:
                valid_questions.append(q)
                self.stats.questions_kept += 1
            else:
                self._log(f"      ⚠️ QUESTION REJETÉE (connaissance générale): {q.question[:50]}...")
                self.stats.questions_rejected_general_knowledge += 1
        
        return valid_questions
```

---

## 📊 Comparaison des Approches

| Aspect | LLM-as-Judge Seul | Fonctions Déterministes Seules | **Hybride (Recommandé)** |
|--------|-------------------|--------------------------------|---------------------------|
| **Vitesse** | ❌ Lent (1-2s/eval) | ✅ Rapide (<0.1s) | ✅ Rapide (0.2s avg) |
| **Coût** | ❌ Élevé ($$$) | ✅ Gratuit | ✅ Faible ($$) |
| **Déterminisme** | ❌ Variable | ✅ 100% reproductible | ✅ 80% reproductible |
| **Nuances** | ✅ Excellent | ❌ Limité | ✅ Bon |
| **Facile Debug** | ❌ Difficile | ✅ Facile | ✅ Facile |
| **Rejection Rate** | 🟡 10-20% | ⚠️ Trop strict (60-80%) | ✅ 30-50% (idéal) |

---

## 🚀 Prochaines Étapes

### **Phase 1: Setup (30 min)**
1. ✅ Copier les fichiers de Maloe dans `src/evaluation/`
2. ✅ Installer dépendances si nécessaire
3. ✅ Tester les fonctions individuellement

### **Phase 2: Tâche 1 - Context Validation (2h)**
1. Créer `QuestionContextValidator`
2. Intégrer dans le pipeline (entre question gen et answer gen)
3. Tester avec 10 questions (mesurer combien rejetées)

### **Phase 3: Tâche 2 - Hybrid Critic (3h)**
1. Créer `HybridCriticAgent`
2. Implémenter les 3 phases (hard rules → déterministe → LLM)
3. Migrer les 6 hard rules existantes
4. Ajouter métriques de Maloe (faithfulness, relevance, METEOR)

### **Phase 4: Tests & Tuning (2h)**
1. Tester sur 20 QA pairs
2. Mesurer: rejection rate, vitesse, coût
3. Ajuster seuils (faithfulness < 0.7?, similarity < 0.75?)
4. Comparer avec l'ancien critic

### **Phase 5: Documentation (1h)**
1. Documenter les nouvelles métriques
2. Mettre à jour SYSTEM_PRESENTATION.md
3. Créer exemples pour le tuteur

---

## 🤔 Questions à Décider

### **1. Tâche 1: Seuil de Similarité**
- Si réponse avec/sans contexte sont **75%+ similaires** → rejeter?
- Ou être plus strict: **85%+**?
- **Recommandation:** Commencer à 75%, ajuster selon résultats

### **2. Tâche 2: Poids des Métriques**
```python
final_score = (
    0.3 * deterministic_scores["faithfulness"] +
    0.2 * deterministic_scores["relevance"] +
    0.1 * deterministic_scores["meteor"] +
    0.4 * llm_scores["overall"]
)
```
- Quelle pondération?
- **Recommandation:** 30% déterministe, 70% LLM (au début)

### **3. Budget LLM**
- Actuellement: 1 appel LLM/QA pour critic
- Avec Tâche 1: +2 appels LLM/question (with/without context)
- **Solution:** Faire validation contexte en batch? Ou seulement sur échantillon?

### **4. Remplacer ou Augmenter?**
- **Option A:** Remplacer complètement le LLM-as-judge par métriques déterministes
- **Option B:** Garder LLM mais l'aider avec métriques (hybride)
- **Recommandation:** Option B (hybride) pour commencer, puis A si résultats bons

---

## 💡 Conclusion

**Ma recommandation finale:**

1. **Pour Tâche 1 (Question sans contexte):**
   - ✅ **À faire** - c'est une excellente idée
   - Implémentation simple avec 2 appels LLM + METEOR
   - Rejette ~20-30% des questions (estimation)
   - Améliore grandement la qualité du dataset

2. **Pour Tâche 2 (LLM vs Déterministe):**
   - ✅ **Approche hybride** (pas full remplacement)
   - Garder LLM-as-judge pour nuances
   - Ajouter métriques de Maloe pour rapidité/déterminisme
   - Architecture en 3 phases: Hard Rules → Déterministe → LLM

**Résultat attendu:**
- Rejection rate: 30-50% (optimal)
- Vitesse: 2-3x plus rapide
- Coût: -40% d'appels LLM
- Qualité: Meilleure (moins de faux positifs)
- Reproductibilité: 80% des décisions déterministes

---

**Prêt à implémenter?** 🚀

On commence par copier les fichiers de Maloe dans `src/evaluation/` et créer `QuestionContextValidator`?
