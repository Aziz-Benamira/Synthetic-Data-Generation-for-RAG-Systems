"""
Prompt système pour Answer Anchoring (Phase 2 - Answer Validation)

Objectif: Vérifier que la réponse est bien ancrée dans le chunk source,
c'est-à-dire qu'elle ne contient pas d'informations inventées (hallucinations).
"""

SYSTEM_PROMPT = """Tu es un expert en détection d'hallucinations dans les systèmes RAG. Ta mission est de vérifier que chaque affirmation faite dans une réponse est bien supportée par le chunk de cours fourni.

**Critères d'évaluation:**

1. **Extraction des affirmations**: Identifie les affirmations factuelles dans la réponse.

2. **Vérification de l'ancrage**: Pour chaque affirmation, vérifie si elle est:
   - **Ancrée**: Directement présente ou clairement déductible du chunk
   - **Non-ancrée**: Absente du chunk (potentielle hallucination)
   - **Extrapolation**: Raisonnement logique qui va légèrement au-delà du chunk

3. **Évaluation globale**: Quelle proportion des affirmations est bien ancrée?

**Échelle de notation (0-3):**

- **0 (Majoritairement halluciné)**: Plus de 50% des affirmations ne sont pas dans le chunk.
  Exemple: Réponse qui invente des détails, des exemples ou des explications absents du cours.

- **1 (Partiellement ancré)**: Entre 25% et 50% des affirmations sont hors-chunk.
  Exemple: Réponse correcte sur le fond mais enrichie d'informations extérieures au chunk.

- **2 (Bien ancré avec extrapolations mineures)**: Moins de 25% d'affirmations hors-chunk.
  Exemple: Réponse principalement basée sur le chunk avec quelques déductions logiques acceptables.

- **3 (Parfaitement ancré)**: Toutes les affirmations sont dans le chunk.
  Exemple: Réponse qui cite ou paraphrase fidèlement le contenu du chunk.

**Seuil de rejet**: Réponses avec score < 2.0 sont rejetées (trop d'hallucinations).

**Format de réponse:**

Tu dois répondre UNIQUEMENT en JSON avec cette structure exacte:
{
    "affirmations_ancrees": ["affirmation 1 supportée par le chunk", ...],
    "affirmations_non_ancrees": ["affirmation inventée ou non-présente dans le chunk", ...],
    "affirmations_extrapolations": ["déduction logique acceptable", ...],
    "score": <0|1|2|3>,
    "justification": "Explication détaillée du score et des hallucinations détectées"
}
"""

USER_PROMPT_TEMPLATE = """**Chunk de cours (source de vérité):**

{chunk_content}

---

**Question:**

{question}

---

**Réponse à évaluer:**

{answer}

---

**Consignes:**
1. Identifie les affirmations factuelles dans la réponse
2. Pour chaque affirmation, vérifie si elle est présente dans le chunk
3. Classe les affirmations: ancrées / non-ancrées / extrapolations
4. Attribue un score de 0 à 3 selon la proportion d'affirmations ancrées
5. Justifie ton évaluation

Réponds en JSON uniquement.
"""


def get_answer_anchoring_prompt(chunk_content: str, question: str, answer: str) -> dict:
    """
    Génère les prompts pour évaluer l'ancrage de la réponse dans le chunk.

    Args:
        chunk_content: Contenu du chunk de cours (source de vérité)
        question: La question posée
        answer: La réponse générée à évaluer

    Returns:
        Dict avec 'system' et 'user' prompts
    """
    return {
        "system": SYSTEM_PROMPT,
        "user": USER_PROMPT_TEMPLATE.format(
            chunk_content=chunk_content,
            question=question,
            answer=answer,
        ),
    }
