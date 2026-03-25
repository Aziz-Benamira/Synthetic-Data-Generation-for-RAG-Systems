"""
Answer Generator V2
====================

Generates strictly-anchored answers from chunks using LLM, designed for 
DeepSeek R1 and the Critic V3 feedback loop.

Key improvements over V1:
- Works with raw chunk dicts (not SemanticChunk objects)
- Handles DeepSeek R1 <think> tags
- Robust JSON parsing with multiple fallbacks
- Strictly anchored: explicit instructions to use ONLY the chunk
- Supports regeneration with Critic V3 evolutionary feedback

Input:  question (str) + chunk (dict)
Output: str (the answer text)
"""

import json
import re
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


# ─── Reuse helpers from question_generator_v2 ──────────────────────────────

from .question_generator_v2 import strip_think_tags, extract_json


# ─── Prompts ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un assistant pédagogique expert en mathématiques financières et probabilités (niveau Master 2).

RÈGLES ABSOLUES:
1. Utilise STRICTEMENT ET UNIQUEMENT le contenu fourni pour répondre
2. JAMAIS de connaissances externes — même si tu les connais
3. Si l'information n'est pas dans le contenu, dis-le clairement  
4. Cite des passages EXACTS du texte source entre guillemets
5. Réponse claire, structurée, en français académique
6. 2-5 phrases — concis mais complet

FORMAT DE SORTIE: UN JSON strict
{
  "answer": "La réponse complète",
  "supporting_quotes": ["citation exacte 1", "citation exacte 2"]
}

IMPORTANT: Réponds UNIQUEMENT avec le JSON, rien d'autre."""


USER_PROMPT_TEMPLATE = """Réponds à la question en utilisant UNIQUEMENT le contenu ci-dessous.

=== QUESTION ===
{question}

=== CONTENU SOURCE (seule information autorisée) ===
Chapitre: {chapter}
Section: {section}
Pages: {pages}

{content}

=== SORTIE ===
{{"answer": "...", "supporting_quotes": ["..."]}}"""


REGENERATE_PROMPT_TEMPLATE = """La réponse précédente a été évaluée par le Critic. Génère une MEILLEURE réponse.

=== QUESTION ===
{question}

=== RÉPONSE PRÉCÉDENTE ===
{previous_answer}

=== FEEDBACK DU CRITIC ===
{feedback}

=== CONTENU SOURCE (seule information autorisée) ===
Chapitre: {chapter}
Section: {section}

{content}

=== INSTRUCTIONS DE CORRECTION ===
- Corrige PRÉCISÉMENT les problèmes mentionnés dans le feedback
- Utilise UNIQUEMENT le contenu source ci-dessus
- N'INVENTE RIEN — chaque affirmation doit être vérifiable dans le texte
- Cite des passages EXACTS du texte entre guillemets
- Garde les éléments de la réponse précédente qui étaient corrects

{{"answer": "...", "supporting_quotes": ["..."]}}"""


# ─── Generator ──────────────────────────────────────────────────────────────

class AnswerGeneratorV2:
    """
    Generates a strictly-anchored answer from a chunk.
    
    Usage:
        from src.llm import LLMManager
        llm = LLMManager.from_direct_llamacpp(...)
        gen = AnswerGeneratorV2(llm)
        
        answer = gen.generate(question_text, chunk_dict)
        # → "Dans le modèle CIR, la dynamique du taux r est..."
    """
    
    def __init__(
        self,
        llm_manager: Any,
        temperature: float = 0.3,
        max_tokens: int = 600,
    ):
        self.llm = llm_manager
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    # ── Public API ──────────────────────────────────────────────────────
    
    def generate(self, question: str, chunk: Dict[str, Any]) -> str:
        """
        Generate an answer for a question using chunk content.
        
        Args:
            question: The question to answer.
            chunk: dict with keys content, chapter, section, page_range, etc.
        
        Returns:
            Answer string.
        
        Raises:
            ValueError if generation fails after all fallbacks.
        """
        user_prompt = USER_PROMPT_TEMPLATE.format(
            question=question,
            chapter=chunk.get('chapter', 'N/A'),
            section=chunk.get('section', 'N/A'),
            pages=self._format_pages(chunk.get('page_range')),
            content=chunk['content'][:3000],
        )
        
        raw = self._call_llm(user_prompt)
        answer = self._parse_answer(raw)
        
        if not answer:
            raise ValueError(
                f"Failed to parse answer from LLM output for chunk {chunk.get('chunk_id')}"
            )
        
        return answer
    
    def regenerate(
        self,
        question: str,
        chunk: Dict[str, Any],
        previous_answer: str,
        feedback: str,
    ) -> str:
        """
        Regenerate an answer after Critic V3 feedback.
        
        Args:
            question: The question.
            chunk: The source chunk dict.
            previous_answer: The answer that failed evaluation.
            feedback: Critic V3 formatted feedback message.
        
        Returns:
            New answer string.
        """
        user_prompt = REGENERATE_PROMPT_TEMPLATE.format(
            question=question,
            previous_answer=previous_answer,
            feedback=feedback,
            chapter=chunk.get('chapter', 'N/A'),
            section=chunk.get('section', 'N/A'),
            content=chunk['content'][:3000],
        )
        
        raw = self._call_llm(user_prompt)
        answer = self._parse_answer(raw)
        
        return answer or previous_answer  # fallback to original
    
    # ── Private ─────────────────────────────────────────────────────────
    
    def _call_llm(self, user_prompt: str) -> str:
        """Call LLM with system + user prompts."""
        from src.llm import LLMConfig
        config = LLMConfig(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        response = self.llm.generate(
            prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            config=config,
        )
        return response.content
    
    def _parse_answer(self, raw: str) -> Optional[str]:
        """
        Parse answer from LLM output with multiple fallbacks.
        
        Pipeline:
        1. Strip <think> tags
        2. Try JSON parse → extract "answer" key
        3. Fallback: use cleaned text as-is (if it looks like an answer)
        """
        cleaned = strip_think_tags(raw)
        
        # Try JSON
        data = extract_json(cleaned)
        if data and 'answer' in data:
            answer = data['answer'].strip()
            if len(answer) > 20:  # sanity check
                return answer
        
        # Fallback: if cleaned text is reasonable, use it directly
        # Remove any remaining JSON artifacts
        text = re.sub(r'[{}":]', '', cleaned).strip()
        # Remove "answer" prefix if present
        text = re.sub(r'^answer\s*', '', text, flags=re.IGNORECASE).strip()
        
        if len(text) > 30:
            logger.warning(f"Using raw text as answer (JSON parse failed)")
            return text
        
        logger.warning(f"Could not parse answer from: {cleaned[:200]}")
        return None
    
    @staticmethod
    def _format_pages(page_range) -> str:
        if isinstance(page_range, (list, tuple)) and len(page_range) >= 2:
            return f"{page_range[0]}-{page_range[1]}"
        return str(page_range or 'N/A')
