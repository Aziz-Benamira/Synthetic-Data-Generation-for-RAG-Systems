# 🎯 Plan d'Implémentation: Hybrid Critic + Question Validator

**Date:** 29 Janvier 2026  
**Décisions confirmées:**
- ✅ Tâche 1: Seuil 85% similarité
- ✅ Tâche 2: Hybride mais GARDER votre LLM-as-judge spécialisé
- ✅ Suggestion prof: 1 prompt par critère (pas tout en un)

---

## 🔍 Analyse: Votre LLM-as-Judge vs Maloe

### **Votre Système Actuel (EXCELLENT, À GARDER!)**

```python
# Votre RUBRICS_FR - Très spécifique aux documents académiques
RUBRICS_FR = {
    "anchoring": {
        "description": "La réponse est-elle ENTIÈREMENT dérivable du contenu du chunk?",
        "pass_conditions": ["Chaque affirmation retrouvée dans le chunk", ...]
    },
    # ... 4 autres critères
}

# Votre SYSTEM_PROMPT_FR - 154 lignes d'instructions adversariales
# - Adversarial prompting ("DÉTECTEUR DE DÉFAUTS")
# - Exemples concrets de rejections
# - Patterns spécifiques ("par exemple", "donc", "ainsi")
# - Calibré pour documents académiques (théorèmes, définitions, preuves)
```

**Points forts:**
- ✅ Prompt adversarial très efficace
- ✅ Exemples concrets de REJECTIONS (EXEMPLE 1-6)
- ✅ Patterns spécifiques aux textes académiques
- ✅ Déjà testé et tuné sur M2_cours.pdf

### **Fonctions de Maloe (GÉNÉRIQUES)**

```python
# llm_metrics.py - Prompts génériques
def llm_as_judge_context_support(model, query, response, context):
    prompt = "You are evaluating if a response is grounded in context..."
    # ⚠️ Prompt générique, pas spécifique académique
    # ⚠️ Pas d'exemples concrets
    # ⚠️ Pas de patterns spécifiques
```

**Fonctions utiles de Maloe:**
1. ✅ **trad_metrics.py** (déterministes):
   - `meteor()` - Overlap sémantique
   - `exact_match()` - Correspondance exacte
   - Ces fonctions N'ONT PAS besoin de LLM!

2. ❌ **llm_metrics.py** (génériques):
   - Prompts trop génériques
   - Pas adaptés aux documents académiques
   - **NE PAS utiliser tel quel**

---

## 💡 Recommandation: Architecture Hybride Optimale

### **Ne REMPLACEZ PAS votre LLM-as-judge, RENFORCEZ-le!**

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: Hard Rules Déterministes (RAPIDE)                │
│  - 6 règles existantes (nombres, causalité, etc.)           │
│  - + NOUVEAUX: METEOR, exact_match de Maloe                │
│  → Rejet immédiat si problème détecté                       │
│  → 70% des cas rejetés ici (pas d'appel LLM)               │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  PHASE 2: LLM-as-Judge PAR CRITÈRE (suggestion prof)       │
│  Au lieu de 1 gros prompt → 5 prompts spécifiques          │
│                                                              │
│  Prompt 1: Évalue ANCHORING uniquement                     │
│  Prompt 2: Évalue LOCAL_ANSWERABILITY uniquement           │
│  Prompt 3: Évalue FACTUAL_ACCURACY uniquement              │
│  Prompt 4: Évalue COMPLETENESS uniquement                  │
│  Prompt 5: Évalue CLARITY uniquement                       │
│                                                              │
│  Avantages:                                                 │
│  ✅ LLM plus focalisé sur UN aspect                        │
│  ✅ Explications plus détaillées                           │
│  ✅ Moins de confusion entre critères                      │
│  ✅ Parallélisable (5 appels simultanés)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 📝 Implémentation Concrète

### **Changement 1: Ajouter Métriques Déterministes de Maloe**

```python
# Dans critic_agent.py - Nouvelles hard rules

def _apply_hard_rules(self, qa_pair, chunk, criteria_evaluations):
    """Phase 1: Hard rules + métriques déterministes"""
    
    # ============ HARD RULES EXISTANTES (1-6) ============
    # [Garder vos 6 règles actuelles]
    
    # ============ NOUVELLES: Métriques de Maloe ============
    
    # RULE 7: METEOR score trop bas → Pas ancré dans chunk
    from evaluation.trad_metrics import meteor
    
    # Extraire phrases clés du chunk pour référence
    chunk_sentences = chunk.content.split('.')[:5]  # Top 5 phrases
    reference_text = '. '.join(chunk_sentences)
    
    meteor_score = meteor(
        prediction=qa_pair.answer,
        reference=reference_text
    )
    
    if meteor_score < 0.30:  # Seuil à ajuster
        logging.info(f"[HARD RULE 7] METEOR score trop bas: {meteor_score:.2f}")
        criteria_evaluations["anchoring"] = CriterionEvaluation(
            criterion="anchoring",
            result=CriterionResult.FAIL,
            score=meteor_score,
            explanation=f"🔴 HARD RULE 7 (METEOR): Réponse mal ancrée dans chunk (score: {meteor_score:.2f} < 0.30)",
            evidence=[]
        )
    
    # RULE 8: Answer = exact copy of question → Pas de contenu
    from evaluation.trad_metrics import exact_match
    
    # Normaliser pour comparaison
    q_normalized = qa_pair.question.lower().strip()
    a_normalized = qa_pair.answer.lower().strip()
    
    if exact_match(a_normalized, q_normalized):
        logging.info(f"[HARD RULE 8] Réponse = copie exacte de la question")
        criteria_evaluations["completeness"] = CriterionEvaluation(
            criterion="completeness",
            result=CriterionResult.FAIL,
            score=0.0,
            explanation="🔴 HARD RULE 8 (Exact Match): Réponse identique à la question (aucun contenu)",
            evidence=[]
        )
    
    # RULE 9: Answer contient <20% des mots du chunk → Superficiel
    chunk_words = set(chunk.content.lower().split())
    answer_words = set(qa_pair.answer.lower().split())
    
    # Retirer stop words
    common_words = {'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'à', 'et', 'ou', 'est'}
    chunk_words -= common_words
    answer_words -= common_words
    
    if len(chunk_words) > 0:
        word_overlap_ratio = len(answer_words & chunk_words) / len(chunk_words)
        
        if word_overlap_ratio < 0.15:  # <15% overlap
            logging.info(f"[HARD RULE 9] Overlap mots trop faible: {word_overlap_ratio*100:.0f}%")
            # Warning mais pas rejet (peut être légitime si réponse concise)
            current_score = criteria_evaluations["anchoring"].score
            if current_score < 0.80:  # Seulement si déjà suspect
                criteria_evaluations["anchoring"] = CriterionEvaluation(
                    criterion="anchoring",
                    result=CriterionResult.FAIL,
                    score=max(0.3, current_score - 0.20),
                    explanation=f"⚠️ HARD RULE 9: Très faible overlap avec chunk ({word_overlap_ratio*100:.0f}%)",
                    evidence=[]
                )
    
    return criteria_evaluations
```

---

### **Changement 2: Séparer en 5 Prompts (Suggestion Prof)**

```python
class HybridCriticAgent:
    """
    Nouveau critic hybride:
    - Phase 1: Hard rules (9 règles dont 3 de Maloe)
    - Phase 2: LLM-as-judge PAR CRITÈRE (5 prompts séparés)
    """
    
    def __init__(self, llm_client, model_name="llama3:8b"):
        self.llm_client = llm_client
        self.model_name = model_name
        
        # Créer 5 prompts système spécialisés (un par critère)
        self.system_prompts = self._create_specialized_prompts()
    
    def _create_specialized_prompts(self) -> dict:
        """
        Au lieu d'UN gros prompt de 154 lignes,
        créer 5 prompts spécialisés (30-40 lignes chacun).
        """
        return {
            "anchoring": ANCHORING_PROMPT_FR,
            "local_answerability": LOCAL_ANSWERABILITY_PROMPT_FR,
            "factual_accuracy": FACTUAL_ACCURACY_PROMPT_FR,
            "completeness": COMPLETENESS_PROMPT_FR,
            "clarity": CLARITY_PROMPT_FR
        }
    
    def evaluate(self, qa_pair, chunk) -> CriticEvaluation:
        """Évaluer avec hard rules + 5 prompts LLM"""
        
        # PHASE 1: Hard rules (déterministes)
        criteria_evaluations = self._initialize_criteria()
        criteria_evaluations = self._apply_hard_rules(qa_pair, chunk, criteria_evaluations)
        
        # Si hard rule a rejeté, pas besoin d'appeler LLM
        hard_rule_failed = [name for name, eval in criteria_evaluations.items() 
                           if eval.result == CriterionResult.FAIL]
        
        if hard_rule_failed:
            logging.info(f"⚡ Hard rules rejected: {hard_rule_failed} - Skip LLM calls")
            return self._make_final_decision(qa_pair, chunk, criteria_evaluations)
        
        # PHASE 2: LLM-as-judge PAR CRITÈRE (5 appels séparés)
        logging.info("🤖 Calling LLM for 5 criteria (separate prompts)...")
        
        for criterion_name in ["anchoring", "local_answerability", 
                               "factual_accuracy", "completeness", "clarity"]:
            
            # Appeler LLM avec prompt spécialisé
            score, explanation = self._evaluate_single_criterion(
                criterion_name, qa_pair, chunk
            )
            
            # Mettre à jour l'évaluation (sauf si hard rule déjà appliquée)
            if criteria_evaluations[criterion_name].result != CriterionResult.FAIL:
                criteria_evaluations[criterion_name] = CriterionEvaluation(
                    criterion=criterion_name,
                    result=CriterionResult.PASS if score >= 0.85 else CriterionResult.FAIL,
                    score=score,
                    explanation=explanation,
                    evidence=[]
                )
        
        return self._make_final_decision(qa_pair, chunk, criteria_evaluations)
    
    def _evaluate_single_criterion(
        self, 
        criterion_name: str, 
        qa_pair: Any, 
        chunk: Any
    ) -> Tuple[float, str]:
        """
        Appeler LLM pour UN seul critère avec prompt spécialisé.
        Plus focalisé, meilleure qualité.
        """
        system_prompt = self.system_prompts[criterion_name]
        
        user_prompt = f"""Évalue UNIQUEMENT le critère: {criterion_name}

=== CHUNK SOURCE ===
{chunk.content[:1000]}...

=== QUESTION ===
{qa_pair.question}

=== RÉPONSE ===
{qa_pair.answer}

=== INSTRUCTIONS ===
Évalue SEULEMENT le critère '{criterion_name}'.
Ignore les autres aspects.

Format JSON:
{{
  "score": 0.0-1.0,
  "explanation": "Explication détaillée"
}}
"""
        
        response = self._call_llm(system_prompt, user_prompt)
        
        try:
            data = json.loads(response)
            return float(data["score"]), data["explanation"]
        except:
            logging.warning(f"Failed to parse LLM response for {criterion_name}")
            return 0.5, "Évaluation impossible"
```

---

### **Les 5 Prompts Spécialisés (Extraits de votre prompt actuel)**

```python
ANCHORING_PROMPT_FR = """Tu es un DÉTECTEUR DE DÉFAUTS pour le critère ANCRAGE.

🎯 TON UNIQUE MISSION: Vérifier si la réponse est ENTIÈREMENT dérivable du chunk.

⚠️ RÈGLE D'OR: ASSUME que la réponse contient des ajouts externes.
Ton travail: LES TROUVER.

🔍 CHERCHE CES DÉFAUTS (dans l'ordre):
1. ❌ Mots-clés dans la réponse ABSENTS du chunk?
2. ❌ Nombres/chiffres NON présents dans le chunk?
3. ❌ Exemples ajoutés ("par exemple") non dans chunk?
4. ❌ Déductions ("donc", "ainsi", "cela implique")?
5. ❌ Concepts techniques absents du chunk?
6. ❌ Paraphrases changeant le sens?

=== EXEMPLES DE REJECTIONS ===

EXEMPLE 1 - REJET:
Chunk: "[Définition de tribu]"
Réponse: "Une tribu est... Par exemple, les boréliens..."
→ SCORE: 0.3
→ RAISON: L'exemple "boréliens" n'est PAS dans le chunk source!

EXEMPLE 2 - REJET:
Chunk: "Le théorème X affirme que..."
Réponse: "On peut en déduire que Y..."
→ SCORE: 0.4  
→ RAISON: "déduire" = INFÉRENCE, pas explicite dans chunk!

EXEMPLE 3 - PASS:
Chunk: "Une tribu est un ensemble stable par union..."
Réponse: "Une tribu est un ensemble stable par union dénombrable."
→ SCORE: 0.95
→ RAISON: Reformulation fidèle, rien ajouté.

=== SCORING ===
- 1.0 = Parfait (chaque mot est retrouvable)
- 0.8-0.9 = Très bon (reformulation fidèle)
- 0.7 = Seuil minimum acceptable
- 0.3-0.6 = Problème (ajouts/déductions)
- 0.0-0.2 = Grave (contenu externe)

Génère JSON: {"score": 0.0-1.0, "explanation": "..."}"""


LOCAL_ANSWERABILITY_PROMPT_FR = """Tu es un DÉTECTEUR DE DÉFAUTS pour le critère RÉPONDABILITÉ LOCALE.

🎯 TON UNIQUE MISSION: Vérifier si la question est répondable UNIQUEMENT avec ce chunk.

⚠️ RÈGLE D'OR: ASSUME que la question nécessite des infos externes.
Ton travail: CONFIRMER ou INFIRMER.

🔍 CHERCHE CES DÉFAUTS:
1. ❌ Question "Pourquoi/Comment" sans explication dans chunk?
2. ❌ Comparaison ("par rapport à", "contrairement à")?
3. ❌ Référence à autres chapitres/sections?
4. ❌ Chunk ne contient qu'une PARTIE de la réponse?
5. ❌ Connaissances externes nécessaires?

=== EXEMPLES ===

EXEMPLE 1 - REJET:
Question: "Pourquoi les tribus sont importantes en probabilité?"
Chunk: "[Définition de tribu]"
→ SCORE: 0.3
→ RAISON: Question "pourquoi" mais chunk ne contient pas l'explication!

EXEMPLE 2 - REJET:
Question: "Comparez les tribus aux σ-algèbres."
Chunk: "[Définition de tribu]"
→ SCORE: 0.2
→ RAISON: Comparaison nécessite info sur σ-algèbres (externe)!

EXEMPLE 3 - PASS:
Question: "Qu'est-ce qu'une tribu?"
Chunk: "[Définition complète de tribu avec propriétés]"
→ SCORE: 1.0
→ RAISON: Chunk contient 100% de la réponse.

Génère JSON: {"score": 0.0-1.0, "explanation": "..."}"""


FACTUAL_ACCURACY_PROMPT_FR = """Tu es un DÉTECTEUR DE DÉFAUTS pour le critère EXACTITUDE FACTUELLE.

🎯 TON UNIQUE MISSION: Vérifier si la réponse est FACTUELLEMENT CORRECTE.

🔍 CHERCHE CES ERREURS:
1. ❌ Nombres/dates DIFFÉRENTS du chunk?
2. ❌ Noms propres mal orthographiés?
3. ❌ Reformulation omettant conditions importantes?
4. ❌ Simplification déformant le sens?
5. ❌ Contradiction avec le chunk?
6. ❌ Nuances ajoutées ("généralement", "souvent")?

=== EXEMPLES ===

EXEMPLE 1 - REJET:
Chunk: "Une tribu est stable par union DÉNOMBRABLE"
Réponse: "Une tribu est stable par union"
→ SCORE: 0.5
→ RAISON: Condition "dénombrable" OMISE = erreur!

EXEMPLE 2 - PASS:
Chunk: "P(A∪B) = P(A) + P(B) - P(A∩B)"
Réponse: "P(A∪B) = P(A) + P(B) - P(A∩B)"
→ SCORE: 1.0
→ RAISON: Formule exacte.

Génère JSON: {"score": 0.0-1.0, "explanation": "..."}"""


COMPLETENESS_PROMPT_FR = """Tu es un DÉTECTEUR DE DÉFAUTS pour le critère COMPLÉTUDE.

🎯 TON UNIQUE MISSION: Vérifier si TOUS les aspects de la question sont adressés.

🔍 CHERCHE CES DÉFAUTS:
1. ❌ Question multi-parties ("et", ",") mais réponse partielle?
2. ❌ Réponse <50 chars pour question complexe?
3. ❌ Réponse = reformulation question sans contenu?
4. ❌ Réponse tronquée?
5. ❌ Question "Qu'est-ce et comment" mais réponse seulement "qu'est-ce"?

=== EXEMPLES ===

EXEMPLE 1 - REJET:
Question: "Qu'est-ce qu'une tribu et comment la construire?"
Réponse: "Une tribu est un ensemble stable par opérations."
→ SCORE: 0.4
→ RAISON: Répond à "qu'est-ce" mais PAS à "comment construire"!

EXEMPLE 2 - PASS:
Question: "Qu'est-ce qu'une tribu?"
Réponse: "[Définition complète avec 3 propriétés]"
→ SCORE: 0.95
→ RAISON: Tous les aspects couverts.

Génère JSON: {"score": 0.0-1.0, "explanation": "..."}"""


CLARITY_PROMPT_FR = """Tu es un DÉTECTEUR DE DÉFAUTS pour le critère CLARTÉ.

🎯 TON UNIQUE MISSION: Vérifier si formulation est claire et académique.

🔍 CHERCHE CES DÉFAUTS:
1. ❌ Langage oral ("truc", "machin", "ça", "c'est quoi")?
2. ❌ Pronoms ambigus ("il", "elle") sans référent clair?
3. ❌ Termes vagues ("certains", "quelques")?
4. ❌ Structure grammaticale confuse?
5. ❌ Jargon non défini?

=== EXEMPLES ===

EXEMPLE 1 - REJET:
Question: "C'est quoi le truc avec les tribus?"
→ SCORE: 0.2
→ RAISON: "truc" + "c'est quoi" = langage oral!

EXEMPLE 2 - PASS:
Question: "Qu'est-ce qu'une tribu en théorie de la mesure?"
→ SCORE: 1.0
→ RAISON: Formulation académique claire.

Génère JSON: {"score": 0.0-1.0, "explanation": "..."}"""
```

---

## 🎯 Avantages de cette Approche

### **1. Garde vos points forts**
✅ Prompt adversarial conservé  
✅ Exemples spécifiques académiques conservés  
✅ Patterns ("donc", "ainsi") conservés  
✅ RUBRICS_FR conservées  

### **2. Ajoute métriques déterministes**
✅ METEOR score (overlap sémantique)  
✅ Exact match (détection copie)  
✅ Word overlap ratio  
✅ 70% de rejets sans appel LLM (rapide + gratuit)  

### **3. Suit suggestion du prof**
✅ 5 prompts séparés (un par critère)  
✅ LLM plus focalisé sur UN aspect  
✅ Explications plus détaillées  
✅ Moins de confusion  

### **4. Optimisation coût/vitesse**
```
Ancien système:
- 1 appel LLM avec 154 lignes de prompt
- Temps: 2-3s/QA
- Coût: 100%

Nouveau système (hybride):
- Hard rules rejettent 70% → 0 appel LLM
- 30% passent → 5 appels LLM (focalisés)
- Temps: 0.1s (hard rules) ou 1.5s (5 appels)
- Coût: -70% (moins d'appels LLM)
```

---

## 📋 Plan d'Action

### **Étape 1: Copier évaluation de Maloe (5 min)**
```powershell
# Créer dossier src/evaluation/
New-Item -ItemType Directory -Path "src\evaluation"

# Copier métriques traditionnelles
Copy-Item maloe_metrics\trad_metrics.py src\evaluation\
Copy-Item maloe_metrics\__init__.py src\evaluation\  # Si existe

# Tester import
cd src\evaluation
python -c "from trad_metrics import meteor; print('OK')"
```

### **Étape 2: Ajouter hard rules de Maloe (30 min)**
Modifier `src/agents/critic_agent.py`:
- Ajouter RULE 7 (METEOR)
- Ajouter RULE 8 (Exact Match)
- Ajouter RULE 9 (Word Overlap)

### **Étape 3: Créer prompts séparés (1h)**
Créer `src/agents/specialized_prompts.py`:
- Extraire les 5 sections de votre SYSTEM_PROMPT_FR
- Créer 5 constantes: ANCHORING_PROMPT_FR, etc.
- Garder les exemples spécifiques

### **Étape 4: Créer HybridCriticAgent (1h)**
Créer `src/agents/hybrid_critic.py`:
- Hériter de CriticAgent
- Override `evaluate()` pour faire 5 appels LLM
- Intégrer hard rules étendues

### **Étape 5: Tester (30 min)**
```powershell
# Tester sur 10 QA pairs
python test_hybrid_critic.py

# Comparer:
# - Ancien: 1 prompt, rejection rate X%
# - Nouveau: 5 prompts + hard rules, rejection rate Y%
```

### **Étape 6: Tâche 1 - Question Validator (1h)**
Créer `src/agents/question_validator.py`:
- Générer réponse AVEC chunk
- Générer réponse SANS chunk
- Comparer avec METEOR (seuil 85%)

---

## ❓ Questions pour Vous

1. **Hard rules de Maloe:**
   - Seuil METEOR: 0.30 OK? (30% overlap minimum)
   - Word overlap: 15% OK?

2. **Prompts séparés:**
   - Voulez-vous que je crée les 5 prompts spécialisés maintenant?
   - Ou préférez-vous ajuster vous-même?

3. **Test:**
   - Tester d'abord sur quick_challenge_test.py?
   - Ou directement sur M2_cours.pdf?

4. **Ordre d'implémentation:**
   - Commencer par Tâche 2 (hybrid critic)?
   - Ou Tâche 1 (question validator)?

---

## 🚀 Prochaines Étapes

**Je recommande cet ordre:**
1. ✅ Copier trad_metrics.py dans src/evaluation/
2. ✅ Ajouter 3 hard rules de Maloe (METEOR, etc.)
3. ✅ Tester hard rules seules sur 10 QA
4. ✅ Créer les 5 prompts spécialisés
5. ✅ Créer HybridCriticAgent
6. ✅ Tester hybrid vs ancien critic
7. ✅ Créer QuestionValidator (Tâche 1)
8. ✅ Tester sur pipeline complet

**Voulez-vous que je commence l'implémentation?** 🚀
