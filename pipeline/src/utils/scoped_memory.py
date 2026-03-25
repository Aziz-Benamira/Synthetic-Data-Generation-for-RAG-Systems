"""
Scoped Memory — Contextual Memory for Question Diversity
=========================================================

Tracks concepts and questions already generated within a TOC section
to prevent redundancy in the Question Generator.

Architecture Decision (2026-02-17):
- Reset per section AND chapter: if either changes → memory clears
- Last 5 questions in prompt: limits token usage (n_ctx=4096)
- Concepts provided optionally by Phase 1 (QuestionEvaluator)
- No embeddings (V1): simple text list, embedding check deferred to V2
- Location: src/utils/ alongside other utility modules

Integration:
    The diversity prompt is injected into QuestionGeneratorV3's user prompt.
    ScopedMemory is instantiated ONCE in the main generation loop and
    updated after each validated QA pair.

Usage:
    memory = ScopedMemory()
    
    for chunk in chunks:
        section_changed = memory.update_scope(chunk)
        diversity_prompt = memory.get_diversity_prompt()
        
        question = question_gen.generate(chunk, diversity_prompt)
        # ... Phase 1 + Phase 2 evaluation ...
        
        memory.register_question(question, concepts=["concept1", "concept2"])
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ScopedMemory:
    """
    Contextual memory scoped by TOC section for question diversity.
    
    Prevents the Question Generator from producing redundant questions
    within the same course section by tracking covered concepts and
    previously generated questions.
    
    The memory resets automatically when the pipeline moves to a new
    section or chapter, since concepts from different sections are
    unrelated and should not influence generation.
    
    Attributes:
        current_section:   Active TOC section title
        current_chapter:   Active TOC chapter title
        covered_concepts:  Concepts already addressed in this section
        covered_questions: Questions already generated in this section
        chunk_count:       Number of chunks processed in this section
    """
    
    MAX_QUESTIONS_IN_PROMPT = 5  # Limit to control prompt size
    
    def __init__(self):
        self.current_section: str = ""
        self.current_chapter: str = ""
        self.covered_concepts: List[str] = []
        self.covered_questions: List[str] = []
        self.chunk_count: int = 0
    
    # ── Public API ──────────────────────────────────────────────────────
    
    def update_scope(self, chunk: Dict[str, Any]) -> bool:
        """
        Check if we moved to a new section and reset memory if so.
        
        Args:
            chunk: dict with keys 'chapter', 'section' (from SemanticChunker)
        
        Returns:
            True if memory was reset (new section detected),
            False if still in the same section.
        """
        new_section = chunk.get('section', '')
        new_chapter = chunk.get('chapter', '')
        
        if new_section != self.current_section or new_chapter != self.current_chapter:
            old = f"{self.current_chapter}/{self.current_section}"
            self.current_section = new_section
            self.current_chapter = new_chapter
            
            # Log before clearing
            if self.covered_questions:
                logger.info(
                    f"Section change: '{old}' → '{new_chapter}/{new_section}'. "
                    f"Clearing memory ({len(self.covered_questions)} questions, "
                    f"{len(self.covered_concepts)} concepts)."
                )
            
            self.covered_concepts = []
            self.covered_questions = []
            self.chunk_count = 0
            return True
        
        self.chunk_count += 1
        return False
    
    def register_question(
        self,
        question: str,
        concepts: Optional[List[str]] = None,
    ) -> None:
        """
        Register a validated question and its key concepts.
        
        Called after a QA pair passes both Phase 1 and Phase 2.
        
        Args:
            question:  The generated question text.
            concepts:  Optional list of key concepts extracted by
                       Phase 1 (QuestionEvaluator). If None, only
                       the question text is tracked.
        """
        self.covered_questions.append(question)
        
        if concepts:
            new_concepts = [c for c in concepts if c not in self.covered_concepts]
            self.covered_concepts.extend(new_concepts)
            if new_concepts:
                logger.debug(
                    f"Registered {len(new_concepts)} new concepts: {new_concepts}"
                )
    
    def get_diversity_prompt(self) -> str:
        """
        Generate the diversity injection prompt for QuestionGeneratorV3.
        
        Returns an empty string if no prior questions exist in this
        section (first chunk), otherwise returns a formatted block
        listing covered concepts and recent questions.
        
        Returns:
            str — the diversity block to inject, or "" if memory is empty.
        """
        if not self.covered_concepts and not self.covered_questions:
            return ""
        
        parts = []
        parts.append("━━━ MÉMOIRE CONTEXTUELLE ━━━")
        parts.append(f"Section: {self.current_section}")
        
        if self.covered_concepts:
            parts.append("\nConcepts DÉJÀ traités dans cette section:")
            for concept in self.covered_concepts:
                parts.append(f"  • {concept}")
        
        if self.covered_questions:
            parts.append("\nQuestions DÉJÀ posées:")
            recent = self.covered_questions[-self.MAX_QUESTIONS_IN_PROMPT:]
            for q in recent:
                # Truncate long questions for prompt space
                display = q[:100] + "..." if len(q) > 100 else q
                parts.append(f"  • {display}")
        
        parts.append("\n⚠️ CONSIGNE: Générez une question sur un ANGLE NOUVEAU.")
        parts.append("Évitez de reformuler les questions ci-dessus.")
        parts.append("Explorez un aspect non couvert du chunk.")
        parts.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        return "\n".join(parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Return current memory statistics for logging/debugging.
        
        Returns:
            dict with chapter, section, counts.
        """
        return {
            "chapter": self.current_chapter,
            "section": self.current_section,
            "concepts_count": len(self.covered_concepts),
            "questions_count": len(self.covered_questions),
            "chunk_count": self.chunk_count,
        }
    
    def __repr__(self) -> str:
        return (
            f"ScopedMemory("
            f"section='{self.current_section}', "
            f"concepts={len(self.covered_concepts)}, "
            f"questions={len(self.covered_questions)}, "
            f"chunks={self.chunk_count})"
        )
