"""
Answer Generator Agent
======================

Generates STRICTLY ANCHORED answers from a chunk.
Each answer MUST be derivable ONLY from the provided chunk content.

Design Principles:
- No external knowledge - ONLY the chunk content
- Answers must cite/reference the source text
- Maintains traceability (chunk_id, pages, etc.)
- Supports multiple answer styles (concise, detailed, with citations)

Input: CandidateQuestion + SemanticChunk
Output: GeneratedAnswer object
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import json
import re


class AnswerStyle(Enum):
    """Style of answer to generate"""
    CONCISE = "concise"       # Short, direct answer
    DETAILED = "detailed"     # Full explanation with context
    WITH_QUOTES = "with_quotes"  # Includes exact quotes from source


@dataclass
class GeneratedAnswer:
    """An answer generated from a chunk for a specific question"""
    answer: str
    style: AnswerStyle
    
    # Source traceability (inherited from question)
    question: str
    question_type: str
    chunk_id: str
    source_file: str
    page_range: tuple
    chapter: str
    section: str
    
    # Answer-specific metadata
    supporting_quotes: List[str] = field(default_factory=list)  # Exact quotes from chunk
    confidence: float = 1.0  # 0-1, how well the chunk supports the answer
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "style": self.style.value,
            "question_type": self.question_type,
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "page_range": list(self.page_range),
            "chapter": self.chapter,
            "section": self.section,
            "supporting_quotes": self.supporting_quotes,
            "confidence": self.confidence
        }


# =============================================================================
# PROMPTS FOR ANSWER GENERATION
# =============================================================================

SYSTEM_PROMPT_FR = """Tu es un expert en génération de réponses pour l'évaluation de systèmes RAG.

RÈGLES STRICTES - TRÈS IMPORTANT:
1. Tu ne dois utiliser QUE le contenu fourni pour répondre
2. JAMAIS de connaissances externes, même si tu les connais
3. Si l'information n'est pas dans le contenu, dis "Information non trouvée dans le contexte"
4. Cite des extraits EXACTS du texte source entre guillemets
5. La réponse doit être vérifiable dans le contexte fourni

OBJECTIF:
Générer une réponse qui pourrait être trouvée par un système RAG qui a récupéré ce chunk.
La réponse doit être ENTIÈREMENT dérivable du contenu fourni.

FORMAT DE SORTIE: JSON strict"""

SYSTEM_PROMPT_EN = """You are an expert in generating answers for RAG system evaluation.

STRICT RULES - VERY IMPORTANT:
1. You must use ONLY the provided content to answer
2. NEVER use external knowledge, even if you know it
3. If information is not in the content, say "Information not found in context"
4. Quote EXACT excerpts from the source text in quotation marks
5. The answer must be verifiable in the provided context

OBJECTIVE:
Generate an answer that could be found by a RAG system that retrieved this chunk.
The answer must be ENTIRELY derivable from the provided content.

OUTPUT FORMAT: Strict JSON"""

USER_PROMPT_TEMPLATE = """Génère une réponse à la question suivante en utilisant UNIQUEMENT le contenu fourni.

=== QUESTION ===
{question}

Type de question: {question_type}
Indices attendus: {hints}

=== CONTEXTE SOURCE (seule information autorisée) ===
Chapitre: {chapter}
Section: {section}
Pages: {pages}

Contenu:
---
{content}
---

=== INSTRUCTIONS ===
1. Réponds UNIQUEMENT avec les informations présentes dans le contenu ci-dessus
2. Cite au moins un extrait EXACT du texte (entre guillemets)
3. Si l'information n'est pas dans le contexte, indique-le clairement
4. Évalue ta confiance (0.0-1.0) que la réponse est complètement supportée par le texte

=== FORMAT DE SORTIE (JSON) ===
{{
  "answer": "La réponse complète et précise",
  "supporting_quotes": ["citation exacte 1", "citation exacte 2"],
  "confidence": 0.95,
  "reasoning": "Brève explication de comment la réponse est dérivée du texte"
}}

Génère UNIQUEMENT le JSON, sans commentaires."""


# =============================================================================
# ANSWER GENERATOR CLASS
# =============================================================================

class AnswerGenerator:
    """
    Generates answers from chunks using an LLM.
    
    Design:
    - Stateless: each call is independent
    - Strictly anchored: no external knowledge
    - Traceable: all metadata preserved
    - Verifiable: includes supporting quotes
    """
    
    def __init__(
        self,
        llm_client: Any,
        model_name: str = "llama-3.3-70b-versatile",
        language: str = "fr",
        answer_style: AnswerStyle = AnswerStyle.DETAILED,
        temperature: float = 0.3  # Lower temperature for more factual answers
    ):
        """
        Initialize the answer generator.
        
        Args:
            llm_client: LLM API client (OpenAI, Groq, etc.)
            model_name: Model to use for generation
            language: "fr" or "en" for prompt language
            answer_style: Style of answers to generate
            temperature: LLM temperature (lower = more deterministic)
        """
        self.llm_client = llm_client
        self.model_name = model_name
        self.language = language
        self.answer_style = answer_style
        self.temperature = temperature
        
        # Select system prompt based on language
        self.system_prompt = SYSTEM_PROMPT_FR if language == "fr" else SYSTEM_PROMPT_EN
    
    def generate_answer(
        self,
        question: Any,  # CandidateQuestion
        chunk: Any,     # SemanticChunk
    ) -> GeneratedAnswer:
        """
        Generate an answer for a question using the chunk content.
        
        Args:
            question: CandidateQuestion object
            chunk: SemanticChunk object (source of truth)
            
        Returns:
            GeneratedAnswer object
        """
        # Build the user prompt
        hints = ", ".join(question.expected_answer_hints) if question.expected_answer_hints else "Aucun"
        
        user_prompt = USER_PROMPT_TEMPLATE.format(
            question=question.question,
            question_type=question.question_type.value if hasattr(question.question_type, 'value') else question.question_type,
            hints=hints,
            chapter=chunk.chapter_title,
            section=chunk.section_title,
            pages=f"{chunk.page_range[0]}-{chunk.page_range[1]}",
            content=chunk.content
        )
        
        # Call LLM
        response = self._call_llm(user_prompt)
        
        # Parse response
        answer = self._parse_response(response, question, chunk)
        
        return answer
    
    def _call_llm(self, user_prompt: str) -> str:
        """Call the LLM API."""
        # Groq/OpenAI-style API
        if hasattr(self.llm_client, 'chat'):
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=1000
            )
            return response.choices[0].message.content
        
        # Anthropic-style API
        elif hasattr(self.llm_client, 'messages'):
            response = self.llm_client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.content[0].text
        
        else:
            raise ValueError("Unsupported LLM client type")
    
    def _parse_response(
        self,
        response: str,
        question: Any,
        chunk: Any
    ) -> GeneratedAnswer:
        """Parse LLM response into GeneratedAnswer object."""
        try:
            # Try to parse JSON directly
            data = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except:
                    data = {"answer": response, "supporting_quotes": [], "confidence": 0.5}
            else:
                data = {"answer": response, "supporting_quotes": [], "confidence": 0.5}
        
        # Extract question_type value
        q_type = question.question_type.value if hasattr(question.question_type, 'value') else str(question.question_type)
        
        return GeneratedAnswer(
            answer=data.get("answer", ""),
            style=self.answer_style,
            question=question.question,
            question_type=q_type,
            chunk_id=chunk.chunk_id,
            source_file=chunk.metadata.get("source", ""),
            page_range=chunk.page_range,
            chapter=chunk.chapter_title,
            section=chunk.section_title,
            supporting_quotes=data.get("supporting_quotes", []),
            confidence=float(data.get("confidence", 0.5))
        )
    
    def generate_batch(
        self,
        question_chunk_pairs: List[tuple],  # [(CandidateQuestion, SemanticChunk), ...]
        progress_callback: Optional[callable] = None
    ) -> List[GeneratedAnswer]:
        """
        Generate answers for multiple question-chunk pairs.
        
        Args:
            question_chunk_pairs: List of (question, chunk) tuples
            progress_callback: Optional callback(current, total) for progress
            
        Returns:
            List of GeneratedAnswer objects
        """
        all_answers = []
        total = len(question_chunk_pairs)
        
        for i, (question, chunk) in enumerate(question_chunk_pairs):
            answer = self.generate_answer(question, chunk)
            all_answers.append(answer)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return all_answers
    
    def regenerate_with_feedback(
        self,
        question: Any,  # CandidateQuestion
        chunk: Any,     # SemanticChunk
        previous_answer: str,
        critic_feedback: str
    ) -> GeneratedAnswer:
        """
        Regenerate an answer based on critic feedback.
        
        This is used in the retry loop when the critic rejects a QA pair.
        The feedback contains specific issues to address.
        
        Args:
            question: CandidateQuestion object
            chunk: SemanticChunk object (source context)
            previous_answer: The answer that was rejected
            critic_feedback: Formatted feedback from CriticAgent.format_feedback_for_retry()
            
        Returns:
            New GeneratedAnswer object
        """
        retry_prompt = f"""Tu dois RÉGÉNÉRER une réponse pour corriger les problèmes identifiés par le Critic.

=== QUESTION ===
{question.question}

=== RÉPONSE REJETÉE ===
{previous_answer}

{critic_feedback}

=== CONTEXTE SOURCE (seule information autorisée) ===
Chapitre: {chunk.chapter_title}
Section: {chunk.section_title}
Pages: {chunk.page_range[0]}-{chunk.page_range[1]}

Contenu:
---
{chunk.content}
---

=== INSTRUCTIONS ===
1. Analyse le feedback du Critic
2. Génère UNE NOUVELLE réponse qui évite les erreurs signalées
3. Utilise UNIQUEMENT les informations du chunk ci-dessus
4. N'INVENTE PAS d'exemples ou d'illustrations
5. N'AJOUTE PAS d'informations externes
6. Cite des extraits EXACTS du texte source

=== FORMAT DE SORTIE (JSON) ===
{{
  "answer": "La réponse corrigée, strictement ancrée dans le chunk",
  "supporting_quotes": ["citation exacte 1", "citation exacte 2"],
  "confidence": 0.95,
  "reasoning": "Comment cette réponse corrige les problèmes signalés"
}}

Génère UNIQUEMENT le JSON."""

        # Call LLM with retry prompt
        response = self._call_llm(retry_prompt)
        
        # Parse response (same parsing logic)
        answer = self._parse_response(response, question, chunk)
        
        return answer


# =============================================================================
# QA PAIR CLASS (Combined Question + Answer)
# =============================================================================

@dataclass
class QAPair:
    """
    A complete Question-Answer pair ready for dataset export.
    This is the final format before critic evaluation.
    """
    # Question
    question: str
    question_type: str
    difficulty: str
    
    # Answer
    answer: str
    supporting_quotes: List[str]
    
    # Source traceability
    chunk_id: str
    source_file: str
    page_range: tuple
    chapter: str
    section: str
    
    # Quality metadata
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary format for JSON/HuggingFace."""
        return {
            "question": self.question,
            "answer": self.answer,
            "question_type": self.question_type,
            "difficulty": self.difficulty,
            "supporting_quotes": self.supporting_quotes,
            "confidence": self.confidence,
            "metadata": {
                "chunk_id": self.chunk_id,
                "source_file": self.source_file,
                "page_range": list(self.page_range),
                "chapter": self.chapter,
                "section": self.section
            }
        }
    
    @classmethod
    def from_question_and_answer(
        cls,
        question: Any,  # CandidateQuestion
        answer: GeneratedAnswer
    ) -> 'QAPair':
        """Create QAPair from question and answer objects."""
        return cls(
            question=question.question,
            question_type=question.question_type.value if hasattr(question.question_type, 'value') else str(question.question_type),
            difficulty=question.difficulty.value if hasattr(question.difficulty, 'value') else str(question.difficulty),
            answer=answer.answer,
            supporting_quotes=answer.supporting_quotes,
            chunk_id=answer.chunk_id,
            source_file=answer.source_file,
            page_range=answer.page_range,
            chapter=answer.chapter,
            section=answer.section,
            confidence=answer.confidence
        )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("Answer Generator Module")
    print("=" * 50)
    print()
    print("Answer Styles:")
    for style in AnswerStyle:
        print(f"  - {style.value}")
    print()
    print("To use:")
    print("  from answer_generator import AnswerGenerator, QAPair")
    print("  generator = AnswerGenerator(llm_client, model_name='llama-3.3-70b-versatile')")
    print("  answer = generator.generate_answer(question, chunk)")
    print("  qa_pair = QAPair.from_question_and_answer(question, answer)")
