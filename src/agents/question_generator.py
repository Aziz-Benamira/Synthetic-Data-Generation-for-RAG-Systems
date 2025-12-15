"""
Question Generator Agent
========================

Generates LOCAL questions from a single chunk.
Each question MUST be answerable from the provided chunk alone.

Design Principles:
- No retrieval, no external knowledge
- Questions are anchored to the chunk content
- Multiple question types for diversity
- Metadata preserved for traceability

Input: SemanticChunk with content and metadata
Output: List of CandidateQuestion objects
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import json


class QuestionType(Enum):
    """Types of questions that can be generated"""
    FACTUAL = "factual"           # Direct fact extraction (what, when, who)
    CONCEPTUAL = "conceptual"     # Understanding concepts (what is, explain)
    PROCEDURAL = "procedural"     # How to do something (how, steps)
    COMPARATIVE = "comparative"   # Compare/contrast (difference between)
    CAUSAL = "causal"            # Cause and effect (why, what causes)
    APPLICATION = "application"   # Apply knowledge (calculate, solve)


class DifficultyLevel(Enum):
    """Difficulty levels for questions"""
    EASY = "easy"           # Direct extraction from text
    MEDIUM = "medium"       # Requires understanding/synthesis
    HARD = "hard"          # Requires deeper reasoning


@dataclass
class CandidateQuestion:
    """A question candidate generated from a chunk"""
    question: str
    question_type: QuestionType
    difficulty: DifficultyLevel
    
    # Source traceability
    chunk_id: str
    source_file: str
    page_range: tuple
    chapter: str
    section: str
    
    # Generation metadata
    key_concepts: List[str] = field(default_factory=list)  # Concepts the question tests
    expected_answer_hints: List[str] = field(default_factory=list)  # Hints for answer gen
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "question_type": self.question_type.value,
            "difficulty": self.difficulty.value,
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "page_range": list(self.page_range),
            "chapter": self.chapter,
            "section": self.section,
            "key_concepts": self.key_concepts,
            "expected_answer_hints": self.expected_answer_hints
        }


# =============================================================================
# PROMPTS FOR QUESTION GENERATION
# =============================================================================

SYSTEM_PROMPT_FR = """Tu es un expert en création de questions pédagogiques pour l'évaluation de systèmes RAG.

RÈGLES STRICTES:
1. Chaque question DOIT être répondable UNIQUEMENT avec le contenu fourni
2. Ne JAMAIS faire référence à des connaissances externes
3. Les questions doivent être précises et non ambiguës
4. Varier les types de questions (factuel, conceptuel, procédural, etc.)
5. Adapter la difficulté au contenu

TYPES DE QUESTIONS À GÉNÉRER:
- FACTUAL: Questions sur des faits directs (définitions, valeurs, noms)
- CONCEPTUAL: Questions sur la compréhension de concepts
- PROCEDURAL: Questions sur des méthodes ou étapes
- COMPARATIVE: Questions comparant des éléments
- CAUSAL: Questions sur les causes et effets
- APPLICATION: Questions d'application (calculs, exemples)

FORMAT DE SORTIE: JSON strict"""

SYSTEM_PROMPT_EN = """You are an expert in creating pedagogical questions for RAG system evaluation.

STRICT RULES:
1. Each question MUST be answerable ONLY from the provided content
2. NEVER reference external knowledge
3. Questions must be precise and unambiguous
4. Vary question types (factual, conceptual, procedural, etc.)
5. Adapt difficulty to content

QUESTION TYPES TO GENERATE:
- FACTUAL: Questions about direct facts (definitions, values, names)
- CONCEPTUAL: Questions about understanding concepts
- PROCEDURAL: Questions about methods or steps
- COMPARATIVE: Questions comparing elements
- CAUSAL: Questions about causes and effects
- APPLICATION: Application questions (calculations, examples)

OUTPUT FORMAT: Strict JSON"""

USER_PROMPT_TEMPLATE = """Génère {num_questions} questions à partir du contenu suivant.

=== MÉTADONNÉES ===
Chapitre: {chapter}
Section: {section}
Type sémantique: {semantic_type}
Pages: {pages}

=== CONTENU ===
{content}

=== INSTRUCTIONS ===
1. Analyse le contenu et identifie les concepts clés
2. Génère {num_questions} questions variées (différents types et difficultés)
3. Chaque question doit être auto-suffisante (compréhensible sans le contexte)
4. Inclus des indices pour la réponse attendue

=== FORMAT DE SORTIE (JSON) ===
{{
  "questions": [
    {{
      "question": "La question complète",
      "question_type": "factual|conceptual|procedural|comparative|causal|application",
      "difficulty": "easy|medium|hard",
      "key_concepts": ["concept1", "concept2"],
      "expected_answer_hints": ["indice1", "indice2"]
    }}
  ]
}}

Génère UNIQUEMENT le JSON, sans commentaires."""


# =============================================================================
# QUESTION GENERATOR CLASS
# =============================================================================

class QuestionGenerator:
    """
    Generates questions from semantic chunks using an LLM.
    
    Design:
    - Stateless: each call is independent
    - Configurable: number of questions, types, difficulty distribution
    - Traceable: all metadata preserved
    """
    
    def __init__(
        self,
        llm_client: Any,  # OpenAI/Anthropic client
        model_name: str = "gpt-4o-mini",
        language: str = "fr",
        default_num_questions: int = 3,
        temperature: float = 0.7
    ):
        """
        Initialize the question generator.
        
        Args:
            llm_client: LLM API client (OpenAI, Anthropic, etc.)
            model_name: Model to use for generation
            language: "fr" or "en" for prompt language
            default_num_questions: Default number of questions per chunk
            temperature: LLM temperature (higher = more creative)
        """
        self.llm_client = llm_client
        self.model_name = model_name
        self.language = language
        self.default_num_questions = default_num_questions
        self.temperature = temperature
        
        # Select system prompt based on language
        self.system_prompt = SYSTEM_PROMPT_FR if language == "fr" else SYSTEM_PROMPT_EN
    
    def generate_from_chunk(
        self,
        chunk: Any,  # SemanticChunk
        num_questions: Optional[int] = None
    ) -> List[CandidateQuestion]:
        """
        Generate questions from a single chunk.
        
        Args:
            chunk: SemanticChunk object with content and metadata
            num_questions: Number of questions to generate (uses default if None)
            
        Returns:
            List of CandidateQuestion objects
        """
        num_q = num_questions or self.default_num_questions
        
        # Build the user prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(
            num_questions=num_q,
            chapter=chunk.chapter_title,
            section=chunk.section_title,
            semantic_type=chunk.semantic_type,
            pages=f"{chunk.page_range[0]}-{chunk.page_range[1]}",
            content=chunk.content
        )
        
        # Call LLM
        response = self._call_llm(user_prompt)
        
        # Parse response
        questions = self._parse_response(response, chunk)
        
        return questions
    
    def _call_llm(self, user_prompt: str) -> str:
        """
        Call the LLM API.
        
        This is a placeholder - implement based on your LLM client.
        """
        # OpenAI-style API
        if hasattr(self.llm_client, 'chat'):
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        
        # Anthropic-style API
        elif hasattr(self.llm_client, 'messages'):
            response = self.llm_client.messages.create(
                model=self.model_name,
                max_tokens=2000,
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
        chunk: Any
    ) -> List[CandidateQuestion]:
        """
        Parse LLM response into CandidateQuestion objects.
        """
        try:
            data = json.loads(response)
            questions_data = data.get("questions", [])
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                questions_data = data.get("questions", [])
            else:
                return []
        
        questions = []
        for q_data in questions_data:
            try:
                question = CandidateQuestion(
                    question=q_data["question"],
                    question_type=QuestionType(q_data.get("question_type", "factual")),
                    difficulty=DifficultyLevel(q_data.get("difficulty", "medium")),
                    chunk_id=chunk.chunk_id,
                    source_file=chunk.metadata.get("source", ""),
                    page_range=chunk.page_range,
                    chapter=chunk.chapter_title,
                    section=chunk.section_title,
                    key_concepts=q_data.get("key_concepts", []),
                    expected_answer_hints=q_data.get("expected_answer_hints", [])
                )
                questions.append(question)
            except (KeyError, ValueError) as e:
                # Skip malformed questions
                continue
        
        return questions
    
    def generate_batch(
        self,
        chunks: List[Any],
        num_questions_per_chunk: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> List[CandidateQuestion]:
        """
        Generate questions from multiple chunks.
        
        Args:
            chunks: List of SemanticChunk objects
            num_questions_per_chunk: Questions per chunk
            progress_callback: Optional callback(current, total) for progress
            
        Returns:
            List of all generated CandidateQuestion objects
        """
        all_questions = []
        total = len(chunks)
        
        for i, chunk in enumerate(chunks):
            questions = self.generate_from_chunk(chunk, num_questions_per_chunk)
            all_questions.extend(questions)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return all_questions


# =============================================================================
# QUESTION TYPE STRATEGIES (Optional - for more control)
# =============================================================================

class QuestionTypeStrategy:
    """
    Strategy pattern for generating specific question types.
    Can be extended to have specialized prompts per question type.
    """
    
    @staticmethod
    def get_type_specific_prompt(
        question_type: QuestionType,
        content: str,
        language: str = "fr"
    ) -> str:
        """Get a prompt optimized for a specific question type."""
        
        prompts_fr = {
            QuestionType.FACTUAL: f"""
À partir du contenu suivant, génère une question FACTUELLE.
La réponse doit être un fait direct extrait du texte (nom, définition, valeur, date).

Contenu: {content}

Format: {{"question": "...", "expected_answer_hints": ["..."]}}
""",
            QuestionType.CONCEPTUAL: f"""
À partir du contenu suivant, génère une question CONCEPTUELLE.
La question doit tester la compréhension d'un concept, pas juste la mémorisation.

Contenu: {content}

Format: {{"question": "...", "expected_answer_hints": ["..."]}}
""",
            QuestionType.APPLICATION: f"""
À partir du contenu suivant, génère une question d'APPLICATION.
La question doit demander d'appliquer une formule, méthode ou concept.

Contenu: {content}

Format: {{"question": "...", "expected_answer_hints": ["..."]}}
"""
        }
        
        prompts_en = {
            QuestionType.FACTUAL: f"""
From the following content, generate a FACTUAL question.
The answer must be a direct fact from the text (name, definition, value, date).

Content: {content}

Format: {{"question": "...", "expected_answer_hints": ["..."]}}
""",
            QuestionType.CONCEPTUAL: f"""
From the following content, generate a CONCEPTUAL question.
The question should test understanding of a concept, not just memorization.

Content: {content}

Format: {{"question": "...", "expected_answer_hints": ["..."]}}
""",
            QuestionType.APPLICATION: f"""
From the following content, generate an APPLICATION question.
The question should ask to apply a formula, method, or concept.

Content: {content}

Format: {{"question": "...", "expected_answer_hints": ["..."]}}
"""
        }
        
        prompts = prompts_fr if language == "fr" else prompts_en
        return prompts.get(question_type, prompts[QuestionType.FACTUAL])


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def estimate_question_potential(chunk_content: str) -> Dict[str, int]:
    """
    Estimate how many questions of each type can be generated from content.
    
    Heuristic based on content analysis:
    - Definitions → Factual questions
    - Examples → Application questions
    - Comparisons → Comparative questions
    - Explanations → Conceptual questions
    """
    content_lower = chunk_content.lower()
    
    estimates = {
        "factual": 0,
        "conceptual": 0,
        "application": 0,
        "comparative": 0,
        "causal": 0,
        "procedural": 0
    }
    
    # Factual indicators
    factual_keywords = ["définition", "definition", "est défini", "is defined", 
                        "s'appelle", "is called", "vaut", "equals"]
    estimates["factual"] = sum(1 for kw in factual_keywords if kw in content_lower)
    
    # Application indicators
    app_keywords = ["exemple", "example", "calculer", "calculate", "appliquer", 
                   "apply", "formule", "formula"]
    estimates["application"] = sum(1 for kw in app_keywords if kw in content_lower)
    
    # Conceptual indicators (longer explanatory text)
    if len(chunk_content) > 500:
        estimates["conceptual"] = 2
    
    # Comparative indicators
    comp_keywords = ["contrairement", "unlike", "différence", "difference", 
                    "par rapport", "compared to", "alors que", "whereas"]
    estimates["comparative"] = sum(1 for kw in comp_keywords if kw in content_lower)
    
    # Causal indicators
    causal_keywords = ["car", "because", "donc", "therefore", "implique", 
                      "implies", "cause", "entraîne", "results in"]
    estimates["causal"] = sum(1 for kw in causal_keywords if kw in content_lower)
    
    return estimates


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Demo without actual LLM call
    print("Question Generator Module")
    print("=" * 50)
    print()
    print("Question Types:")
    for qt in QuestionType:
        print(f"  - {qt.value}")
    print()
    print("Difficulty Levels:")
    for dl in DifficultyLevel:
        print(f"  - {dl.value}")
    print()
    print("To use:")
    print("  from question_generator import QuestionGenerator")
    print("  generator = QuestionGenerator(llm_client, model_name='gpt-4o-mini')")
    print("  questions = generator.generate_from_chunk(chunk)")
