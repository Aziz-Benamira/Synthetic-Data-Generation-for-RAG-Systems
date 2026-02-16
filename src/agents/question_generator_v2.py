"""
Question Generator V2
=====================

Generates questions from chunks using LLM, designed for DeepSeek R1 
and the Critic V3 feedback loop.

Key improvements over V1:
- Works with raw chunk dicts (not SemanticChunk objects)
- Handles DeepSeek R1 <think> tags
- Robust JSON parsing with multiple fallbacks
- Single question per chunk (quality over quantity)
- Supports regeneration with Critic V3 feedback

Input:  dict with keys: chunk_id, content, chapter, section, etc.
Output: str (the question text)
"""

import json
import re
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


# ─── Helpers ────────────────────────────────────────────────────────────────

def strip_think_tags(text: str) -> str:
    """Remove DeepSeek R1 <think>...</think> blocks from output."""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned.strip()


def extract_json(text: str) -> Optional[dict]:
    """Try to extract a JSON object from noisy LLM output."""
    # 1. Try the full text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 2. Find outermost { ... }
    brace_depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if brace_depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    start = None
    
    return None


def extract_question_text(text: str) -> Optional[str]:
    """Last-resort: pull a question out of free-form text."""
    # Look for a sentence ending with ?
    matches = re.findall(r'[A-ZÀ-Ü][^.!?]*\?', text)
    if matches:
        # Pick the longest one (likely the real question)
        return max(matches, key=len).strip()
    return None


# ─── Prompts ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un expert en création de questions pédagogiques de niveau Master 2 en mathématiques financières et probabilités.

RÈGLES ABSOLUES:
1. Génère EXACTEMENT UNE question en français académique
2. La question DOIT être répondable UNIQUEMENT avec le contenu fourni
3. La question doit être auto-suffisante (compréhensible sans voir le contexte)
4. Pas de questions triviales (oui/non) — exige une explication, démonstration, ou analyse
5. Le sujet mathématique doit être précis et ciblé

TYPES DE QUESTIONS POSSIBLES:
- Conceptuel: "Expliquez pourquoi...", "Quelle est la signification de..."
- Application: "Comment applique-t-on... pour calculer..."
- Démonstration: "Démontrez que...", "Montrez que..."
- Comparaison: "Comparez... et... en termes de..."
- Causal: "Pourquoi... implique-t-il..."

FORMAT DE SORTIE: UN JSON strict
{"question": "La question complète"}

IMPORTANT: Réponds UNIQUEMENT avec le JSON, rien d'autre."""


USER_PROMPT_TEMPLATE = """Génère UNE question académique à partir du contenu suivant.

=== MÉTADONNÉES ===
Chapitre: {chapter}
Section: {section}
Type: {semantic_type}
Pages: {pages}

=== CONTENU ===
{content}

=== SORTIE ===
{{"question": "..."}}"""


REGENERATE_PROMPT_TEMPLATE = """La question précédente a été rejetée par le Critic. Génère une MEILLEURE question.

=== QUESTION REJETÉE ===
{previous_question}

=== FEEDBACK DU CRITIC ===
{feedback}

=== CONTENU SOURCE ===
Chapitre: {chapter}
Section: {section}

{content}

=== INSTRUCTIONS ===
- Corrige les problèmes identifiés dans le feedback
- La question DOIT être répondable avec CE contenu
- Formulation académique claire et précise

{{"question": "..."}}"""


# ─── Generator ──────────────────────────────────────────────────────────────

class QuestionGeneratorV2:
    """
    Generates a single, high-quality question from a chunk.
    
    Usage:
        from src.llm import LLMManager
        llm = LLMManager.from_direct_llamacpp(...)
        gen = QuestionGeneratorV2(llm)
        
        question = gen.generate(chunk_dict)
        # → "Expliquez comment le modèle CIR modifie..."
    """
    
    def __init__(
        self,
        llm_manager: Any,
        temperature: float = 0.7,
        max_tokens: int = 300,
    ):
        self.llm = llm_manager
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    # ── Public API ──────────────────────────────────────────────────────
    
    def generate(self, chunk: Dict[str, Any]) -> str:
        """
        Generate one question from a chunk dict.
        
        Args:
            chunk: dict with keys chunk_id, content, chapter, section, 
                   semantic_type, page_range, etc.
        
        Returns:
            Question string.
        
        Raises:
            ValueError if generation fails after all fallbacks.
        """
        user_prompt = USER_PROMPT_TEMPLATE.format(
            chapter=chunk.get('chapter', 'N/A'),
            section=chunk.get('section', 'N/A'),
            semantic_type=chunk.get('semantic_type', 'N/A'),
            pages=self._format_pages(chunk.get('page_range')),
            content=chunk['content'][:3000],
        )
        
        raw = self._call_llm(user_prompt)
        question = self._parse_question(raw)
        
        if not question:
            raise ValueError(
                f"Failed to parse question from LLM output for chunk {chunk.get('chunk_id')}"
            )
        
        return question
    
    def regenerate(
        self,
        chunk: Dict[str, Any],
        previous_question: str,
        feedback: str,
    ) -> str:
        """
        Regenerate a question after Critic feedback.
        
        Args:
            chunk: The source chunk dict.
            previous_question: The rejected question.
            feedback: Critic V3 feedback message.
        
        Returns:
            New question string.
        """
        user_prompt = REGENERATE_PROMPT_TEMPLATE.format(
            previous_question=previous_question,
            feedback=feedback,
            chapter=chunk.get('chapter', 'N/A'),
            section=chunk.get('section', 'N/A'),
            content=chunk['content'][:3000],
        )
        
        raw = self._call_llm(user_prompt)
        question = self._parse_question(raw)
        
        return question or previous_question  # fallback to original
    
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
    
    def _parse_question(self, raw: str) -> Optional[str]:
        """
        Parse question from LLM output with multiple fallbacks.
        
        Pipeline:
        1. Strip <think> tags
        2. Try JSON parse → extract "question" key
        3. Fallback: find a sentence ending with ?
        """
        cleaned = strip_think_tags(raw)
        
        # Try JSON
        data = extract_json(cleaned)
        if data and 'question' in data:
            q = data['question'].strip()
            if len(q) > 20:  # sanity check
                return q
        
        # Fallback: extract question sentence
        q = extract_question_text(cleaned)
        if q and len(q) > 20:
            return q
        
        logger.warning(f"Could not parse question from: {cleaned[:200]}")
        return None
    
    @staticmethod
    def _format_pages(page_range) -> str:
        if isinstance(page_range, (list, tuple)) and len(page_range) >= 2:
            return f"{page_range[0]}-{page_range[1]}"
        return str(page_range or 'N/A')
