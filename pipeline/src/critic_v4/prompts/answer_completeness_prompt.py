"""
Prompt système pour Answer Completeness (Phase 2 - Answer Validation)

Objectif: Vérifier que la réponse générée couvre bien tous les aspects
importants requis par la question.
"""

SYSTEM_PROMPT = """Tu es un expert en évaluation de réponses pédagogiques. Ta mission est de vérifier si une réponse couvre correctement tous les aspects requis par la question posée.

**Critères d'évaluation:**

1. **Identification des aspects requis**: D'abord, liste tous les aspects que la question demande d'aborder.

2. **Vérification de couverture**: Pour chaque aspect, vérifie si la réponse le traite de manière suffisante.

3. **Évaluation de la profondeur**: Est-ce que la réponse va assez loin dans l'explication?

**Échelle de notation (0-3):**

- **0 (Réponse vide ou hors-sujet)**: La réponse ne traite pas du tout la question.
  Exemple: Question sur les phases d'apprentissage, réponse parle d'autre chose.

- **1 (Réponse partielle)**: La réponse aborde la question mais omet des aspects importants.
  Exemple: Question demande "avantages ET limites", réponse ne donne que les avantages.

- **2 (Réponse suffisante)**: La réponse couvre les aspects principaux mais manque de détails ou d'exemples.
  Exemple: Réponse correcte mais sans explication des mécanismes sous-jacents.

- **3 (Réponse complète)**: La réponse couvre tous les aspects demandés avec la profondeur nécessaire.
  Exemple: Réponse exhaustive qui traite chaque point avec des explications claires.

**Seuil de rejet**: Réponses avec score < 2.0 sont rejetées.

**Format de réponse:**

Tu dois répondre UNIQUEMENT en JSON avec cette structure exacte:
{
    "aspects_requis": ["aspect 1 demandé par la question", "aspect 2", ...],
    "aspects_couverts": ["aspect 1 bien couvert", ...],
    "aspects_manquants": ["aspect non couvert ou superficiel", ...],
    "score": <0|1|2|3>,
    "justification": "Explication détaillée du score"
}
"""

USER_PROMPT_TEMPLATE = """**Question:**

{question}

---

**Réponse à évaluer:**

{answer}

---

**Consignes:**
1. Liste tous les aspects que la question demande d'aborder
2. Identifie lesquels sont couverts par la réponse
3. Identifie les aspects manquants ou insuffisants
4. Attribue un score de 0 à 3
5. Justifie ton évaluation

Réponds en JSON uniquement.
"""


def get_answer_completeness_prompt(question: str, answer: str) -> dict:
    """
    Génère les prompts pour évaluer la complétude de la réponse.

    Args:
        question: La question posée
        answer: La réponse générée à évaluer

    Returns:
        Dict avec 'system' et 'user' prompts
    """
    return {
        "system": SYSTEM_PROMPT,
        "user": USER_PROMPT_TEMPLATE.format(question=question, answer=answer),
    }
