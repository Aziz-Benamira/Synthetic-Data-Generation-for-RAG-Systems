"""
AnswerGeneratorV3SoG
====================

Extension of AnswerGeneratorV3 that injects SoG (Synthesize-on-Graph)
knowledge graph context into the answer generation prompts.

Two modes:
  - combined   : chunk content + graph entities/relations (default)
  - graph_only : only graph context, no chunk content

Usage:
    gen = AnswerGeneratorV3SoG(llm=llm, ...)
    result = gen.generate(
        question="...",
        chunk={...},
        graph_context="...",   # formatted output from SoGRetriever
        sog_mode="combined",
    )

How it works:
    The subclass sets self._graph_context / self._sog_mode before calling
    super().generate().  Python MRO ensures that _build_first_prompt() and
    _build_regen_prompt() resolve to the overridden versions here, so the
    parent's retry loop automatically uses the enriched prompts.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Imported from the original Agentic_AI package (must be on sys.path).
# sys.path is configured by pipeline_v4_sog.py before this module is imported.
from src.agents.answer_generator_v3 import AnswerGeneratorV3


# ─── Enhanced system prompt ──────────────────────────────────────────────────

_SYSTEM_PROMPT_SOG = """Tu es un assistant pédagogique expert (niveau Master 2).

Tu disposes de DEUX sources d'information complémentaires :
1. Un graphe de connaissances (entités, relations, passages indexés)
2. Le contenu textuel du chapitre source

RÈGLES ABSOLUES:
1. Utilise les deux sources pour construire ta réponse
2. JAMAIS de connaissances externes — même si tu les connais
3. Les entités et relations du graphe enrichissent la réponse — utilise-les
4. Cite des passages EXACTS du texte source entre guillemets
5. Réponse claire, structurée, en français académique
6. 3-6 phrases — complète mais concise

FORMAT DE SORTIE: UN JSON strict
{
  "answer": "La réponse complète",
  "supporting_quotes": ["citation exacte 1", "citation exacte 2"]
}

IMPORTANT: Réponds UNIQUEMENT avec le JSON, rien d'autre."""


_SYSTEM_PROMPT_GRAPH_ONLY = """Tu es un assistant pédagogique expert (niveau Master 2).

Tu disposes d'un graphe de connaissances extrait d'un manuel physique.

RÈGLES ABSOLUES:
1. Utilise UNIQUEMENT les entités, relations et passages du graphe
2. JAMAIS de connaissances externes
3. Construis une réponse cohérente à partir des éléments du graphe
4. Réponse claire, structurée, en français académique
5. 3-6 phrases — complète mais concise

FORMAT DE SORTIE: UN JSON strict
{
  "answer": "La réponse complète",
  "supporting_quotes": ["passage extrait du graphe 1", "passage extrait du graphe 2"]
}

IMPORTANT: Réponds UNIQUEMENT avec le JSON, rien d'autre."""


# ─── Enhanced prompt templates ───────────────────────────────────────────────

_FIRST_ATTEMPT_COMBINED = """Réponds à la question en utilisant les deux sources ci-dessous.

=== QUESTION ===
{question}

=== SOURCE 1 : GRAPHE DE CONNAISSANCES ===
{graph_context}

=== SOURCE 2 : CONTENU TEXTUEL SOURCE ===
Chapitre: {chapter}
Section: {section}
Pages: {pages}

{content}

=== SORTIE ===
{{"answer": "...", "supporting_quotes": ["..."]}}"""


_FIRST_ATTEMPT_GRAPH_ONLY = """Réponds à la question en utilisant UNIQUEMENT le graphe de connaissances.

=== QUESTION ===
{question}

=== GRAPHE DE CONNAISSANCES ===
{graph_context}

=== SORTIE ===
{{"answer": "...", "supporting_quotes": ["..."]}}"""


_REGENERATE_COMBINED = """La réponse précédente a été évaluée par le Critic. Génère une MEILLEURE réponse.

=== QUESTION ===
{question}

=== RÉPONSE PRÉCÉDENTE (tentative {attempt}) ===
{previous_answer}

=== FEEDBACK DU CRITIC ===
{feedback}

=== SOURCE 1 : GRAPHE DE CONNAISSANCES ===
{graph_context}

=== SOURCE 2 : CONTENU TEXTUEL SOURCE ===
Chapitre: {chapter}
Section: {section}

{content}

=== INSTRUCTIONS DE CORRECTION ===
- Corrige PRÉCISÉMENT les problèmes mentionnés dans le feedback
- Utilise les deux sources ci-dessus
- N'INVENTE RIEN — chaque affirmation doit venir des sources
- Cite des passages EXACTS entre guillemets
- Garde les éléments corrects de la réponse précédente

{{"answer": "...", "supporting_quotes": ["..."]}}"""


_REGENERATE_GRAPH_ONLY = """La réponse précédente a été évaluée. Génère une MEILLEURE réponse basée sur le graphe.

=== QUESTION ===
{question}

=== RÉPONSE PRÉCÉDENTE (tentative {attempt}) ===
{previous_answer}

=== FEEDBACK DU CRITIC ===
{feedback}

=== GRAPHE DE CONNAISSANCES ===
{graph_context}

=== INSTRUCTIONS DE CORRECTION ===
- Corrige les problèmes mentionnés dans le feedback
- Utilise uniquement les entités et passages du graphe
- N'invente rien

{{"answer": "...", "supporting_quotes": ["..."]}}"""


# ─── AnswerGeneratorV3SoG ────────────────────────────────────────────────────

class AnswerGeneratorV3SoG(AnswerGeneratorV3):
    """
    AnswerGeneratorV3 with SoG knowledge graph context injection.

    Python MRO guarantees that the parent's retry loop calls the overridden
    _build_first_prompt() and _build_regen_prompt() methods, so no duplication
    of retry / scoring logic is needed here.
    """

    def generate(
        self,
        question: str,
        chunk: Dict[str, Any],
        graph_context: str = "",
        sog_mode: str = "combined",
    ) -> Dict[str, Any]:
        """
        Generate an answer with optional SoG graph context.

        Args:
            question:      Question to answer
            chunk:         Source chunk dict (keys: content, chapter, section, page_range, …)
            graph_context: Formatted output from SoGRetriever.retrieve()["formatted"]
                           Empty string → falls back to standard AnswerGeneratorV3 behaviour
            sog_mode:      "combined"   — chunk content + graph context
                           "graph_only" — only graph context (no chunk text)

        Returns:
            Same dict as AnswerGeneratorV3.generate() (answer, phase2_*)
        """
        if not graph_context:
            return super().generate(question, chunk)

        # Stash SoG state for prompt builders (accessed via self in parent loop)
        self._graph_context = graph_context
        self._sog_mode = sog_mode

        result = super().generate(question, chunk)

        # Clean up state
        self._graph_context = ""
        self._sog_mode = "combined"
        return result

    # ── Overridden prompt builders ────────────────────────────────────────────

    def _build_first_prompt(self, question: str, chunk: Dict[str, Any]) -> str:
        graph_ctx = getattr(self, "_graph_context", "")
        sog_mode  = getattr(self, "_sog_mode", "combined")

        if not graph_ctx:
            return super()._build_first_prompt(question, chunk)

        if sog_mode == "graph_only":
            return _FIRST_ATTEMPT_GRAPH_ONLY.format(
                question=question,
                graph_context=graph_ctx,
            )
        # combined (default)
        return _FIRST_ATTEMPT_COMBINED.format(
            question=question,
            graph_context=graph_ctx,
            chapter=chunk.get("chapter", "N/A"),
            section=chunk.get("section", "N/A"),
            pages=self._format_pages(chunk.get("page_range")),
            content=chunk["content"][:3000],
        )

    def _build_regen_prompt(
        self,
        question: str,
        chunk: Dict[str, Any],
        previous_answer: str,
        feedback: str,
        attempt: int,
    ) -> str:
        graph_ctx = getattr(self, "_graph_context", "")
        sog_mode  = getattr(self, "_sog_mode", "combined")

        if not graph_ctx:
            return super()._build_regen_prompt(
                question, chunk, previous_answer, feedback, attempt
            )

        if sog_mode == "graph_only":
            return _REGENERATE_GRAPH_ONLY.format(
                question=question,
                previous_answer=previous_answer,
                feedback=feedback,
                attempt=attempt,
                graph_context=graph_ctx,
            )
        # combined
        return _REGENERATE_COMBINED.format(
            question=question,
            previous_answer=previous_answer,
            feedback=feedback,
            attempt=attempt,
            graph_context=graph_ctx,
            chapter=chunk.get("chapter", "N/A"),
            section=chunk.get("section", "N/A"),
            content=chunk["content"][:3000],
        )

    # ── _call_llm: override system prompt when SoG is active ─────────────────

    def _call_llm(self, user_prompt: str) -> str:
        graph_ctx = getattr(self, "_graph_context", "")
        sog_mode  = getattr(self, "_sog_mode", "combined")

        if graph_ctx:
            system = _SYSTEM_PROMPT_GRAPH_ONLY if sog_mode == "graph_only" else _SYSTEM_PROMPT_SOG
        else:
            # Fall back to original system prompt (imported from parent module)
            from src.agents.answer_generator_v3 import SYSTEM_PROMPT
            system = SYSTEM_PROMPT

        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response["choices"][0]["message"]["content"]
