# Métriques pour l'évaluation de RAG

**Auteur :** Maloe Aymonier (ENSTA - Institut Polytechnique de Paris)
**Date :** Décembre 2025

-----

## 1\. Retriever (Récupérateur)

Cette section évalue la capacité du système à trouver les documents pertinents dans le corpus en fonction de la requête utilisateur.

### 1.1 Métriques "Non-rank based" (Non ordonnées)

Ces métriques évaluent la pertinence brute sans se soucier de l'ordre d'apparition des documents.

  * **Input :** Query, ensemble de chunks, et la Ground Truth (vérité terrain).
  * **Output :** Liste non ordonnée de chunks jugés pertinents.

**Définitions et Formules :**
En se basant sur la matrice de confusion (Vrais Positifs TP, Faux Positifs FP, Vrais Négatifs TN, Faux Négatifs FN):

  * **Accuracy (Exactitude) :** Proportion de prédictions correctes (positives et négatives) sur le total.
    $$
    \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
    $$ 

  * **Precision (Précision) :** Proportion de documents pertinents parmi ceux récupérés.
    $$
    \text{Precision} = \frac{TP}{TP + FP}
    $$ 

  * **Recall (Rappel) :** Proportion de documents pertinents récupérés par rapport à tous les documents pertinents existants.
    $$
    \text{Recall} = \frac{TP}{TP + FN}
    $$ 

  * **F1-Score :** Moyenne harmonique de la précision et du rappel (utile pour trouver un équilibre entre les deux).
    $$
    F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
    $$ 

### 1.2 Métriques "Rank based" (Ordonnées)

Ces métriques prennent en compte la position des documents pertinents dans la liste de résultats. Elles sont cruciales car un utilisateur regarde rarement au-delà des premiers résultats.

  * **Input :** Ensemble de queries, chunks d'information, Ground Truth ordonnée.
  * **Output :** Listes ordonnées de chunks.

**Les métriques principales :**

  * **MRR (Mean Reciprocal Rank) :** La position moyenne du *premier* chunk pertinent.
    $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
    *Où $\text{rank}_i$ est la position du premier résultat pertinent pour la requête $i$.* 

  * **NDCG (Normalized Discounted Cumulative Gain) :** Mesure la qualité du classement en donnant plus de poids aux documents pertinents situés en haut de la liste.
    $$\text{NDCG}_p = \frac{DCG_p}{IDCG_p}$$
    *Où $DCG$ (Discounted Cumulative Gain) pénalise la pertinence logarithmiquement par la position, et $IDCG$ est le score idéal maximal.* 

  * **MAP (Mean Average Precision) :** La moyenne des précisions moyennes pour chaque requête. Elle prend en compte l'ordre de tous les documents pertinents, pas seulement le premier. 

-----

## 2\. Model-wide (Évaluation globale)

Cette section concerne l'évaluation de la qualité du texte généré par rapport à une référence ou au contexte.

### 2.1 Métriques Traditionnelles

| Nom | Description / Formule | Avantage | Inconvénient |
| :--- | :--- | :--- | :--- |
| **Exact Match (EM)** | Compare si la réponse est *identique* à la référence (binaire 0 ou 1). | Gratuit, utile pour les réponses formatées (dates, noms). | Quasiment inutile pour du texte libre (nuances, synonymes).  |
| **METEOR** | $$(1-p) \frac{(\alpha^{2}+1)\text{Prec} \times \text{Rec}}{\text{Rec} + \alpha \text{Prec}}$$ <br> Mélange BLEU et ROUGE avec gestion des synonymes et pénalité d'ordre ($p$). | Peu coûteux, meilleure corrélation avec le jugement humain que BLEU. | Reste limité : ignore souvent le contexte profond et la polysémie complexe.  |

### 2.2 Métriques Deep Learning

| Nom | Fonctionnement | Avantages | Inconvénients |
| :--- | :--- | :--- | :--- |
| **BERTScore** | Calcule la Précision, le Rappel et le F1 sur les *embeddings* contextuels (vecteurs) produits par un modèle BERT. | Prend en compte le contexte sémantique. Très bons résultats généraux. | Coûteux en calcul (nécessite un GPU).  |
| **RAGAS** | Génère une représentation latente de l'output avec un LLM puis utilise la similarité cosinus. | Considérée comme "probablement la meilleure métrique" actuelle. | Encore plus coûteux et "Black box" (difficile à expliquer/justifier).  |

### 2.3 Métriques "LLM Based"

Utilisation des LLMs eux-mêmes comme outil d'évaluation.

  * **LLM-as-judge :** Consiste à prompter un modèle puissant (ex: GPT-4) pour évaluer la réponse.
      * *Exemple de prompt :* "Check if the response is supported by the retrieved context." 
  * **Semantic Perplexity :** Calcule l'incertitude du modèle sur sa propre génération.
      * *Définition :* L'exponentielle de l'entropie croisée (cross-entropy) des logits. Elle mesure la probabilité accordée à chaque token généré. Une perplexité basse indique que le modèle est "confiant" et cohérent. 

-----

## 3\. Generator

*Note : Cette section est présente dans le sommaire  mais ne contient pas de slides de contenu spécifiques dans la présentation fournie.*

-----

## 4\. Autres aspects à évaluer

### 4.1 Métriques "Risk-aware" (Gestion du risque)

Pour un système conçu pour pouvoir s'abstenir de répondre (afin d'éviter les hallucinations), on classe les couples questions/réponses en 4 catégories:

  * **AK (Answerable, Kept) :** Question répondable, réponse fournie (Succès).
  * **UK (Unanswerable, Kept) :** Question irrépondable, réponse fournie (Hallucination/Danger).
  * **UD (Unanswerable, Discarded) :** Question irrépondable, rejetée correctement (Sécurité).
  * **AD (Answerable, Discarded) :** Question répondable, rejetée (Occasion manquée).

**Les 4 métriques dérivées :**

1.  **Le Risque :** Proportion d'erreurs (hallucinations) parmi les réponses fournies.
    $$
    \text{Risque} = \frac{UK}{AK + UK}
    $$
2.  **La Prudence :** Capacité à détecter les questions pièges (Rappel sur les questions irrépondables).
    $$
    \text{Prudence} = \frac{UD}{UK + UD}
    $$ 
3.  **L'Alignement :** Précision globale de la décision de répondre ou non.
    $$
    \text{Alignement} = \frac{AK + UD}{\text{Total}}
    $$
4.  **La Couverture :** Taux de réponse du système.
    $$
    \text{Couverture} = \frac{AK + UK}{\text{Total}} 
    $$ 

### 4.2 Coûts et Efficacité

Il ne suffit pas d'être précis, il faut être performant et économiquement viable.

  * **Latence (Vitesse) :**
      * **TTFT (Time to First Token) :** Temps d'attente avant le début de l'affichage de la réponse (critique pour l'expérience utilisateur).
      * **Latence totale :** Durée complète de la génération.
  * **Coûts financiers :**
      * Coût par token (entrée vs sortie, souvent différent selon les fournisseurs API).
      * Coûts de stockage (notamment pour les bases de données vectorielles volumineuses).
  * **ROI du Retriever :**
      * Un retriever plus performant peut être plus cher à l'unité, mais s'il renvoie une information plus dense et précise, il réduit la taille du contexte envoyé au LLM (moins de tokens d'entrée), ce qui peut réduire le coût global.

### 4.3 Sécurité

  * **Adversarial Attacks (Attaques contradictoires) :** Évaluation de la robustesse face à des corpus "empoisonnés" conçus pour tromper le modèle.
  * **Protection des données :** Fuite d'informations sensibles présentes dans le contexte.
  * **Métrique principale :** Success rate (taux de réussite) des attaques.

## Bibliographie
  Source principale (unique pour l'instant) : 

  [Retrieval Augmented Generation Evaluation in the Era of Large Language Models: A Comprehensive Survey, ](https://arxiv.org/abs/2504.14891)