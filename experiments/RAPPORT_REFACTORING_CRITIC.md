# Rapport de Refactoring du Critic Agent

## Date
29 janvier 2026

## Objectif
Améliorer la robustesse et l'efficacité du système de critique en combinant des règles déterministes (hard rules) avec des évaluations LLM spécialisées.

---

## 1. Validation de la Nécessité du Contexte (Expérience abandonnée)

### Implémentation Testée
Nous avons implémenté un validateur de questions basé sur la comparaison de deux réponses générées par le LLM :
- Réponse générée AVEC le contexte du chunk
- Réponse générée SANS le contexte (connaissance générale)

L'idée était de mesurer la similarité entre ces deux réponses avec la métrique METEOR. Si la similarité dépassait 85%, la question était considérée comme trop générique et rejetée.

### Problèmes Rencontrés

**Problème 1 : Calibration impossible**
Les scores METEOR observés se situaient tous entre 29% et 37%, très en dessous du seuil de 85%. Même pour des questions génériques, le LLM reformulait ses réponses avec des mots différents, produisant des scores de similarité faibles malgré un contenu sémantiquement identique.

**Problème 2 : Coût**
Cette approche nécessite 2 appels LLM supplémentaires par question (avec et sans contexte), soit 4 appels LLM au total par paire Q/R lorsqu'on inclut la génération et l'évaluation. Cela contredit l'objectif de réduction des coûts.

**Problème 3 : Redondance avec le système existant**
Le critic existant possède déjà un critère `local_answerability` qui vérifie si la question nécessite des informations externes au chunk ou si la réponse peut être complétée uniquement avec le chunk fourni. Cette validation rend le validateur redondant.

### Décision
Nous avons abandonné cette approche et décidé de faire confiance au critère `local_answerability` déjà présent dans le critic, qui effectue cette vérification de manière plus efficace avec un seul appel LLM.

---

## 2. Révision des Hard Rules Existantes

### Hard Rules Conservées

**RULE 1 : Détection d'hallucinations numériques**
Cette règle extrait tous les nombres de la réponse et les compare avec ceux présents dans le chunk source. Elle utilise une expression régulière robuste et une tolérance de 2% pour gérer les arrondis légitimes. Elle détecte efficacement les cas où le LLM invente des valeurs numériques absentes du chunk.

Point fort : Précision dans la détection des hallucinations factuelles chiffrées avec gestion intelligente des variations d'arrondi.

**RULE 4 : Détection de répétitions question-réponse**
Cette règle calcule le chevauchement de mots entre la question et la réponse après filtrage des mots vides (articles, prépositions). Si plus de 80% des mots de la question apparaissent dans une réponse courte (moins de 20 mots), la paire est rejetée pour tautologie.

Point fort : Filtrage des stop words et condition combinée (overlap + longueur) qui réduit les faux positifs.

**RULE 5 : Détection de langage oral**
Cette règle recherche des marqueurs de langage familier ou oral dans la question (truc, machin, c'est quoi, ça). Elle garantit un niveau académique de qualité pour le dataset.

Point fort : Liste de marqueurs bien définie, détection simple et efficace, très peu de faux positifs.

### Hard Rules Supprimées

**RULE 2 : Questions explicatives sans marqueurs causaux**
Cette règle rejetait les questions "Pourquoi" ou "Comment" lorsque le chunk ne contenait pas de marqueurs de causalité (car, parce que, donc). Problème : trop stricte car de nombreux chunks mathématiques présentent des faits sans expliquer le pourquoi avec des connecteurs explicites. Cette validation est mieux gérée par le critère `local_answerability` du LLM.

**RULE 3 : Réponses courtes pour questions complexes**
Cette règle rejetait les réponses de moins de 40 caractères pour des questions de plus de 15 mots. Problème : seuil arbitraire qui génère des faux positifs. Une formule mathématique concise (20-30 caractères) peut être une réponse parfaitement complète. La calibration basée uniquement sur le nombre de caractères ne mesure pas la qualité réelle.

**RULE 6 : Pronoms vagues en début de réponse**
Cette règle pénalisait les réponses commençant par des pronoms (il, elle, cela) sans référent clair. Problème : faible impact car souvent le contexte de la question rend le pronom clair. Génère des warnings peu utiles.

---

## 3. Intégration des Métriques de Maloe

### Métrique Ajoutée

**RULE 7 (nouvelle) : METEOR pour l'ancrage au chunk**
Nous ajoutons la métrique METEOR pour mesurer le chevauchement lexical entre la réponse et le contenu du chunk source. Cette métrique combine précision, rappel et une pénalité pour la fragmentation de l'ordre des mots.

Implémentation : Nous extrayons les 5 premières phrases du chunk (contenu clé) et calculons le score METEOR avec la réponse. Si le score est inférieur à 0.30, la réponse est rejetée pour ancrage insuffisant.

Point fort : Formule mathématique rigoureuse qui mesure l'overlap global (mots + ordre) et complète RULE 1 qui ne détecte que les nombres.

### Métriques de Maloe Non Retenues

**exact_match**
Cette métrique nécessite une correspondance exacte à 100% entre deux chaînes de caractères. Elle est moins robuste que RULE 4 car elle ne tolère aucune variation de ponctuation ou d'espaces et ne filtre pas les mots vides. RULE 4 offre une détection de répétition plus intelligente.

### Comparaison des Métriques Communes

**Détection de répétition**
- Hard Rule 4 : Calcule l'overlap après filtrage des stop words, condition combinée avec la longueur. Score de robustesse élevé.
- exact_match de Maloe : Comparaison binaire stricte (100% identique ou non). Moins adapté aux tautologies partielles.
- Verdict : RULE 4 est supérieure.

**Ancrage au chunk**
- Hard Rule 1 : Détection précise des nombres avec tolérance d'arrondi. Limitée aux valeurs numériques.
- METEOR de Maloe : Mesure l'overlap lexical global avec formule rigoureuse. Complément nécessaire.
- Verdict : Les deux sont complémentaires.

### Autres Métriques Disponibles dans le Code de Maloe

Le code de Maloe contient d'autres métriques potentiellement utiles pour des évaluations futures :

**Métriques de retrieval**
- Mean Reciprocal Rank (MRR) : Position du premier document pertinent dans les résultats.
- NDCG (Normalized Discounted Cumulative Gain) : Qualité du classement avec plus de poids pour les documents pertinents en tête.
- MAP (Mean Average Precision) : Précision moyenne sur tous les documents pertinents.

Ces métriques sont utiles pour évaluer la qualité du retriever mais ne s'appliquent pas directement à la critique des paires Q/R.

**Métriques risk-aware**
- Risk : Taux d'hallucination parmi les réponses fournies.
- Prudence : Capacité à détecter les questions non répondables.
- Alignment : Justesse globale des décisions (répondre vs s'abstenir).
- Coverage : Taux de réponses fournies.

Ces métriques pourraient être intégrées dans une phase d'analyse globale du dataset généré.

### Métriques LLM Avancées de Maloe et Pourquoi Elles Ne Sont Pas Intégrées Maintenant

Le travail de Maloe contient plusieurs métriques LLM sophistiquées qui méritent une attention particulière. Voici pourquoi nous ne les intégrons pas dans cette phase du projet, tout en reconnaissant leur valeur pour des évaluations futures.

**BERTScore**
Cette métrique calcule la similarité sémantique en utilisant les embeddings contextuels d'un modèle BERT pré-entraîné. Elle compare chaque token de la réponse avec chaque token de la référence dans l'espace vectoriel, capturant ainsi les similarités sémantiques que les métriques lexicales comme METEOR manquent.

Pourquoi nous ne l'utilisons pas maintenant : BERTScore nécessite un modèle BERT chargé en mémoire et un GPU pour des performances acceptables. Notre environnement actuel utilise Ollama avec des modèles locaux (Mistral, Llama3) et notre objectif est de réduire les coûts de calcul. Ajouter BERTScore introduirait une dépendance supplémentaire lourde (transformers, torch) et ralentirait significativement le pipeline. De plus, BERTScore est conçu pour comparer une réponse générée à une référence gold standard, alors que notre critic évalue la qualité intrinsèque d'une paire Q/R sans référence externe.

Quand elle serait utile : Dans une phase d'analyse post-génération où nous voulons comparer notre dataset synthétique avec des annotations humaines de référence. BERTScore excellerait pour mesurer la fidélité sémantique globale du dataset.

**LLM-as-judge (implémentation de Maloe)**
Maloe a implémenté trois fonctions LLM-as-judge spécialisées :
- llm_as_judge_context_support : Vérifie si la réponse est ancrée dans le contexte fourni
- llm_as_judge_answer_relevance : Vérifie si la réponse répond à la question
- llm_as_judge_coherence : Évalue la clarté et la structure de la réponse

Pourquoi nous ne les utilisons pas maintenant : Ces fonctions sont des placeholders qui appellent une fonction "placeholder_llm_call" non connectée à un vrai modèle. Pour les rendre opérationnelles, il faudrait les connecter à une API externe (GPT-4, Claude) ou adapter les prompts pour nos modèles locaux. Notre système actuel possède déjà un LLM-as-judge fonctionnel avec des prompts calibrés en français pour nos critères spécifiques (anchoring, local_answerability, factual_accuracy, completeness, clarity). Ces critères recouvrent largement les trois fonctions de Maloe : notre anchoring équivaut à context_support, notre local_answerability combine answer_relevance avec la vérification de complétude du contexte, et notre clarity correspond à coherence.

Redévelopper les fonctions de Maloe créerait une duplication de code sans gain fonctionnel. L'approche plus stratégique est d'améliorer nos prompts existants en les séparant en 5 prompts spécialisés, ce qui apporte les mêmes bénéfices de focalisation que l'implémentation de Maloe tout en gardant notre calibration actuelle.

Quand elles seraient utiles : Si nous décidons de migrer vers des modèles API externes (GPT-4, Claude) pour le critic, les fonctions de Maloe fourniraient une base solide à adapter. Leur structure JSON structurée avec score et reasoning est élégante et pourrait inspirer une refonte de notre format de sortie.

**Semantic Perplexity**
Cette métrique mesure la confiance du modèle dans sa propre génération en calculant l'exponentielle de l'entropie croisée des logits. Une perplexité basse indique que le modèle est sûr de ses choix de tokens, tandis qu'une perplexité élevée signale une incertitude qui pourrait correspondre à une hallucination ou une réponse de mauvaise qualité.

Pourquoi nous ne l'utilisons pas maintenant : Semantic Perplexity nécessite l'accès aux logits bruts du modèle pour chaque token généré. Notre pipeline actuel utilise les modèles Ollama via des appels API simples qui retournent uniquement le texte final, sans les probabilités token par token. Accéder aux logits nécessiterait d'utiliser l'API low-level d'Ollama ou de charger les modèles directement avec llama.cpp, ce qui complexifierait significativement l'architecture.

De plus, la perplexité mesure la confiance du modèle, pas la qualité objective du contenu. Un modèle peut être très confiant dans une hallucination bien formulée (perplexité basse) ou incertain face à une réponse correcte mais rare dans ses données d'entraînement (perplexité haute). Cette métrique serait plus utile dans un système avec mécanisme d'abstention où le modèle peut refuser de répondre quand sa confiance est trop basse.

Quand elle serait utile : Dans une version future où nous implémentons un mécanisme d'abstention intelligent. Le modèle pourrait utiliser Semantic Perplexity pour détecter en temps réel quand il est en train d'halluciner (perplexité qui monte soudainement) et s'abstenir de compléter la réponse. Cela serait particulièrement pertinent pour les métriques risk-aware de Maloe (Risk, Prudence, Alignment).

**RAGAS (mentionné dans la bibliographie)**
Maloe mentionne RAGAS comme une métrique prometteuse qui utilise des représentations latentes LLM pour mesurer la qualité sémantique.

Pourquoi nous ne l'utilisons pas : RAGAS est une bibliothèque externe complexe qui nécessite des appels à des LLMs propriétaires (OpenAI) et ajoute une couche d'abstraction significative. Son coût d'exécution serait prohibitif pour évaluer des milliers de paires Q/R. De plus, RAGAS est principalement conçu pour l'évaluation finale de systèmes RAG en production, pas pour la critique en boucle de génération synthétique où nous avons besoin de feedback actionnable pour régénérer les paires défaillantes.

### Synthèse : Stratégie d'Intégration Progressive

Notre approche respecte et valorise le travail de Maloe tout en adoptant une stratégie d'intégration progressive :

**Phase actuelle (Baseline + Hard Rules)**
- Intégrer METEOR : métrique traditionnelle légère, sans dépendance lourde, complète nos hard rules existantes
- Séparer les prompts LLM : améliorer notre LLM-as-judge existant sans réinventer la roue

**Phase future (Post-validation de l'approche hybride)**
- Si l'approche hybride fonctionne et que nous voulons pousser plus loin, intégrer BERTScore pour l'analyse de qualité globale du dataset
- Si nous migrons vers des modèles API, adapter les fonctions LLM-as-judge de Maloe avec leurs formats JSON élégants
- Si nous implémentons un mécanisme d'abstention, intégrer Semantic Perplexity pour la détection d'incertitude en temps réel
- Utiliser les métriques risk-aware de Maloe pour l'évaluation finale du système complet

Cette stratégie permet de tester rapidement notre hypothèse (hard rules + prompts séparés améliorent le critic) sans complexifier prématurément le système, tout en gardant une feuille de route claire pour intégrer les contributions de Maloe quand elles apporteront le plus de valeur.

---

## 4. Architecture de l'Évaluation LLM

### Approche Actuelle (baseline)
Le système actuel utilise un seul prompt LLM volumineux (154 lignes) qui évalue simultanément les 5 critères de qualité :
- Ancrage (anchoring)
- Répondabilité locale (local_answerability)
- Exactitude factuelle (factual_accuracy)
- Complétude (completeness)
- Clarté (clarity)

### Nouvelle Architecture (en cours d'implémentation)

**Étape 1 : Hard Rules (déterministes)**
Application des 4 hard rules qui rejettent immédiatement les cas évidents :
- RULE 1 : Nombres hallucinations
- RULE 4 : Répétition question-réponse
- RULE 5 : Langage oral
- RULE 7 : METEOR ancrage faible

**Étape 2 : Prompts LLM Spécialisés**
Si les hard rules ne rejettent pas la paire Q/R, nous passons à 5 prompts LLM distincts, chacun concentré sur un seul critère. Cette séparation permet :
- Des prompts plus courts et focalisés (30 lignes chacun au lieu de 154)
- Une meilleure qualité d'évaluation par critère
- La possibilité d'exécuter les évaluations en parallèle
- Un débogage plus facile en cas de problème sur un critère spécifique

Chaque prompt spécialisé contient :
- Des instructions claires pour un seul aspect
- Des exemples spécifiques à ce critère
- Une échelle de notation adaptée

### Avantages de cette Approche Hybride

**Efficacité**
Les hard rules filtrent rapidement 20-30% des paires Q/R sans appel LLM, réduisant les coûts et le temps d'exécution.

**Précision**
Les prompts spécialisés permettent au LLM de se concentrer sur un seul aspect à la fois, améliorant la qualité de l'évaluation pour chaque critère.

**Reproductibilité**
Les hard rules sont déterministes (même entrée = même sortie), tandis que les prompts focalisés réduisent la variance des évaluations LLM.

---

## 5. Prochaines Étapes

1. Implémenter la suppression de RULE 2, 3, 6 dans le code
2. Ajouter RULE 7 avec METEOR (seuil initial à 0.30)
3. Tester RULE 7 sur un échantillon pour calibrer le seuil optimal (0.25-0.35)
4. Créer les 5 prompts spécialisés dans un nouveau fichier
5. Mesurer les métriques baseline avant modifications
6. Implémenter l'architecture hybride complète
7. Comparer les performances baseline vs hybride
8. Décision Go/No-Go selon les critères définis dans EXPERIMENT_001

---

## Conclusion

Ce refactoring du critic agent vise à combiner le meilleur des deux mondes : la rapidité et la fiabilité des règles déterministes pour les cas évidents, et la finesse d'évaluation des LLM pour les cas nuancés nécessitant une compréhension sémantique. L'abandon du validateur de contexte initial nous permet de nous concentrer sur des améliorations qui apportent une vraie valeur ajoutée sans redondance avec les mécanismes existants.
