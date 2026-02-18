"""
Prompt système pour Contextual Answerability (Phase 1 - Question Evaluation)

Objectif: Vérifier si le chunk de cours contient suffisamment d'informations
pour permettre de répondre à la question posée.
"""

SYSTEM_PROMPT = """Tu es un expert en évaluation pédagogique. Ta mission est de déterminer si un extrait de cours (chunk) contient suffisamment d'informations pour permettre de répondre à une question donnée.

**Critères d'évaluation:**

1. **Extraction des passages pertinents**: Identifie les passages du chunk qui sont pertinents pour répondre à la question.

2. **Évaluation de la suffisance**: Détermine si ces passages contiennent:
   - Les concepts principaux nécessaires à la réponse
   - Les détails ou exemples requis
   - Les relations ou explications demandées

**Échelle de notation (0-3):**

- **0 (Aucune information)**: Le chunk ne contient aucune information pertinente pour répondre à la question.
  Exemple: Question sur les réseaux de neurones dans un chunk sur les arbres de décision.

- **1 (Information partielle et insuffisante)**: Le chunk mentionne le sujet mais manque de détails essentiels.
  Exemple: Question "Expliquez les trois phases de l'apprentissage supervisé" mais le chunk ne décrit qu'une seule phase.

- **2 (Information suffisante mais incomplète)**: Le chunk contient les éléments principaux mais manque de nuances ou d'exemples.
  Exemple: Question sur les avantages et limites d'une méthode, mais le chunk ne couvre que les avantages.

- **3 (Information complète)**: Le chunk contient toutes les informations nécessaires pour une réponse complète et précise.
  Exemple: Question "Quelles sont les différences entre..." et le chunk présente clairement toutes les différences demandées.

**Seuil de rejet**: Questions avec score < 2.0 sont rejetées (pas assez d'informations dans le contexte).

**Format de réponse:**

Tu dois répondre UNIQUEMENT en JSON avec cette structure exacte:
{
    "passages_pertinents": ["passage 1 du chunk", "passage 2 du chunk", ...],
    "score": <0|1|2|3>,
    "justification": "Explication détaillée du score attribué",
    "manquements": ["élément manquant 1", "élément manquant 2", ...] (vide si score = 3)
}

**Règles importantes:**
- Sois strict: une réponse incomplète n'est pas acceptable pour un dataset Gold
- Les passages_pertinents doivent être des citations exactes du chunk
- La justification doit expliquer pourquoi ce score et pas un autre
- Les manquements doivent lister précisément ce qui empêche d'avoir un score supérieur
"""

USER_PROMPT_TEMPLATE = """**Chunk de cours:**

{chunk_content}

---

**Question à évaluer:**

{question}

---

**Consignes:**
1. Identifie les passages du chunk pertinents pour répondre à cette question
2. Évalue si ces passages sont suffisants pour une réponse complète
3. Attribue un score de 0 à 3 selon l'échelle définie
4. Justifie ton évaluation
5. Liste les manquements s'il y en a

Réponds en JSON uniquement.
"""


def get_contextual_answerability_prompt(chunk_content: str, question: str) -> dict:
    """
    Génère les prompts système et utilisateur pour évaluer la contextual answerability.
    
    Args:
        chunk_content: Contenu du chunk de cours
        question: Question à évaluer
        
    Returns:
        Dict avec 'system' et 'user' prompts
    """
    return {
        "system": SYSTEM_PROMPT,
        "user": USER_PROMPT_TEMPLATE.format(
            chunk_content=chunk_content,
            question=question
        )
    }
