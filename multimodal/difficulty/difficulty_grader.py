"""
Difficulty Grader (Phase 3 - Question Difficulty)

Évalue le niveau de difficulté cognitif d'une question selon la taxonomie
de Bloom révisée, adaptée au contexte académique francophone.

Niveaux:
  1 - Factuel       : rappel direct (recall)
  2 - Compréhension : reformulation / explication
  3 - Application   : application d'une méthode ou formule
  4 - Analyse       : comparaison, causalité, décomposition
  5 - Synthèse      : connexion multi-concepts, évaluation critique
"""

import json
import logging
from typing import Any, Dict, Optional

from ..prompts.difficulty_grader_prompt import get_difficulty_grader_prompt

logger = logging.getLogger(__name__)

# Correspondance niveau → label canonique
LEVEL_LABELS = {
    1: "Factuel",
    2: "Compréhension",
    3: "Application",
    4: "Analyse",
    5: "Synthèse",
}

# Alias acceptés dans les réponses LLM (variations de casse/accents)
LABEL_ALIASES = {
    "factuel": 1, "factuelle": 1, "recall": 1, "mémorisation": 1,
    "compréhension": 2, "comprehension": 2, "understanding": 2,
    "application": 3, "applying": 3,
    "analyse": 4, "analysis": 4, "analyzing": 4, "analyzing": 4,
    "synthèse": 5, "synthese": 5, "synthesis": 5, "évaluation": 5, "evaluation": 5,
}


class DifficultyGrader:
    """
    Évalue le niveau de difficulté cognitif d'une question (1–5, taxonomie de Bloom).

    Ce composant peut être utilisé de deux façons :
      - Inline dans PipelineV4 (Phase 3, optionnelle via config.enable_difficulty_grading)
      - En post-traitement sur un Gold Dataset existant (scripts/annotate_difficulty.py)

    Le niveau est déterminé uniquement à partir de la question et du chunk source.
    La réponse générée n'est pas utilisée pour ne pas biaiser l'évaluation.
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        temperature: float = 0.1,
        max_tokens: int = 600,
    ):
        """
        Args:
            llm: Instance Llama (si None, doit être fourni lors de grade())
            temperature: Faible pour un jugement déterministe
            max_tokens: Suffisant pour le JSON + justification (600 tokens)
        """
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens

    def grade(
        self,
        question: str,
        chunk_content: str,
        llm: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Évalue le niveau de difficulté de la question.

        Args:
            question: La question à évaluer
            chunk_content: Le chunk source utilisé pour générer la question
            llm: Instance Llama (optionnel si fourni dans __init__)

        Returns:
            Dict avec:
            - level: int (1–5)
            - label: str (Factuel / Compréhension / Application / Analyse / Synthèse)
            - justification: str — explication du niveau attribué
            - linguistic_signals: List[str] — signaux détectés dans la question
            - bloom_operations: List[str] — opérations cognitives requises

        Raises:
            ValueError: Si aucun LLM fourni
            RuntimeError: Si appel LLM ou parsing échoue
        """
        active_llm = llm or self.llm
        if active_llm is None:
            raise ValueError("Un LLM doit être fourni soit dans __init__ soit dans grade()")

        prompts = get_difficulty_grader_prompt(question, chunk_content)

        logger.info(f"Évaluation Difficulty Grader pour: {question[:80]}...")

        try:
            response = active_llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": prompts["system"]},
                    {"role": "user",   "content": prompts["user"]},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            llm_output = response["choices"][0]["message"]["content"]
            logger.debug(f"Réponse LLM brute: {llm_output}")

            result = self._parse_llm_response(llm_output)

            logger.info(
                f"Difficulty: Level {result['level']} — {result['label']}"
            )
            return result

        except Exception as e:
            logger.error(f"Erreur Difficulty Grader: {e}")
            raise RuntimeError(f"Échec du grading de difficulté: {e}") from e

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse_llm_response(self, llm_output: str) -> Dict[str, Any]:
        """
        Parse la réponse JSON du LLM. Gère :
          - Balises <think>...</think> de DeepSeek
          - Blocs markdown ```json ... ```
          - Alias de labels (accents, variantes EN/FR)
          - level fourni comme string ("3") ou int (3)
        """
        try:
            cleaned = self._clean_raw_output(llm_output)
            evaluation = json.loads(cleaned)

            # Résoudre le level
            level = self._resolve_level(evaluation)

            # Label canonique
            label = LEVEL_LABELS[level]

            return {
                "level": level,
                "label": label,
                "justification": evaluation.get("justification", ""),
                "linguistic_signals": evaluation.get("linguistic_signals", []),
                "bloom_operations": evaluation.get("bloom_operations", []),
            }

        except json.JSONDecodeError as e:
            logger.error(f"Parsing JSON échoué: {e}\nSortie: {llm_output}")
            raise RuntimeError(f"Réponse LLM invalide (JSON attendu): {e}") from e
        except (ValueError, KeyError) as e:
            logger.error(f"Structure JSON invalide: {e}\nSortie: {llm_output}")
            raise RuntimeError(f"Structure JSON invalide: {e}") from e

    @staticmethod
    def _clean_raw_output(raw: str) -> str:
        """Retire <think>...</think>, balises markdown, espaces superflus."""
        # Retirer le bloc <think> de DeepSeek
        if "</think>" in raw:
            raw = raw.split("</think>", 1)[-1]

        cleaned = raw.strip()

        # Retirer les blocs markdown ```json ... ``` ou ``` ... ```
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        return cleaned.strip()

    @staticmethod
    def _resolve_level(evaluation: Dict[str, Any]) -> int:
        """
        Résout le niveau depuis 'level' (int/str) ou 'label' (str alias).
        Valide que le niveau est dans [1, 5].
        """
        # Priorité au champ "level"
        if "level" in evaluation:
            level = int(evaluation["level"])
            if not 1 <= level <= 5:
                raise ValueError(f"Niveau hors borne: {level} (attendu 1–5)")
            return level

        # Fallback sur "label" via les alias
        if "label" in evaluation:
            label_raw = str(evaluation["label"]).strip().lower()
            level = LABEL_ALIASES.get(label_raw)
            if level is None:
                raise ValueError(f"Label non reconnu: '{evaluation['label']}'")
            return level

        raise ValueError("Ni 'level' ni 'label' trouvé dans la réponse LLM")
