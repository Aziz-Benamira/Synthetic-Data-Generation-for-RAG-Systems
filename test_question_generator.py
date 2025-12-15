"""
Test Question Generator with real chunks from our PDFs.
Uses a mock LLM for testing the pipeline structure.
"""

import sys
sys.path.insert(0, 'src/chunking')
sys.path.insert(0, 'src/agents')

from semantic_chunker import SemanticChunker
from question_generator import (
    QuestionGenerator, 
    CandidateQuestion, 
    QuestionType, 
    DifficultyLevel,
    estimate_question_potential,
    USER_PROMPT_TEMPLATE,
    SYSTEM_PROMPT_FR
)
import json


class MockLLMClient:
    """
    Mock LLM client for testing without API calls.
    Returns realistic-looking responses based on chunk content.
    """
    
    def __init__(self):
        self.chat = self  # OpenAI-style interface
        self.completions = self
    
    def create(self, model, messages, temperature=0.7, response_format=None):
        """Mock the OpenAI chat completion API"""
        # Extract the user message content
        user_msg = messages[-1]["content"]
        
        # Generate mock questions based on content analysis
        mock_response = self._generate_mock_response(user_msg)
        
        # Create mock response object
        class MockChoice:
            def __init__(self, content):
                self.message = type('obj', (object,), {'content': content})()
        
        class MockResponse:
            def __init__(self, content):
                self.choices = [MockChoice(content)]
        
        return MockResponse(mock_response)
    
    def _generate_mock_response(self, prompt: str) -> str:
        """Generate realistic mock questions based on prompt content."""
        
        # Check if content is in French or English
        is_french = "Chapitre:" in prompt or "définition" in prompt.lower()
        
        # Generate appropriate mock questions
        if is_french:
            return json.dumps({
                "questions": [
                    {
                        "question": "Quelle est la définition d'une tribu en théorie des probabilités ?",
                        "question_type": "factual",
                        "difficulty": "easy",
                        "key_concepts": ["tribu", "sigma-algèbre", "ensemble"],
                        "expected_answer_hints": ["collection de sous-ensembles", "fermée par complémentation", "fermée par union dénombrable"]
                    },
                    {
                        "question": "Pourquoi une réunion de tribus n'est-elle pas nécessairement une tribu ?",
                        "question_type": "conceptual",
                        "difficulty": "medium",
                        "key_concepts": ["tribu", "réunion", "propriétés"],
                        "expected_answer_hints": ["pas fermée", "contre-exemple possible"]
                    },
                    {
                        "question": "Comment construire la plus petite tribu contenant une famille d'ensembles donnée ?",
                        "question_type": "procedural",
                        "difficulty": "hard",
                        "key_concepts": ["tribu engendrée", "intersection", "construction"],
                        "expected_answer_hints": ["intersection de toutes les tribus", "contenant la famille"]
                    }
                ]
            }, ensure_ascii=False)
        else:
            return json.dumps({
                "questions": [
                    {
                        "question": "What are Einstein's two postulates of special relativity?",
                        "question_type": "factual",
                        "difficulty": "easy",
                        "key_concepts": ["special relativity", "postulates", "light speed"],
                        "expected_answer_hints": ["laws of physics same in all inertial frames", "speed of light constant"]
                    },
                    {
                        "question": "Why is the speed of light the same for all observers regardless of their motion?",
                        "question_type": "conceptual",
                        "difficulty": "medium",
                        "key_concepts": ["postulate 2", "invariance", "reference frames"],
                        "expected_answer_hints": ["fundamental postulate", "independent of source motion"]
                    },
                    {
                        "question": "How does time dilation affect a clock moving at high velocity relative to a stationary observer?",
                        "question_type": "application",
                        "difficulty": "hard",
                        "key_concepts": ["time dilation", "Lorentz factor", "velocity"],
                        "expected_answer_hints": ["moving clock runs slower", "gamma factor"]
                    }
                ]
            })


def test_question_generator():
    """Test the question generator with real chunks."""
    
    print("=" * 70)
    print("TEST: Question Generator")
    print("=" * 70)
    
    # Test with French PDF only (faster)
    print("\n[1] Loading French PDF (M2_cours.pdf)...")
    chunker_fr = SemanticChunker('data/pdfs/M2_cours.pdf')
    chunks_fr = chunker_fr.chunk_document()
    print(f"    Loaded {len(chunks_fr)} chunks")
    
    # Create mock LLM client
    mock_client = MockLLMClient()
    
    # Test French question generation
    print("\n" + "=" * 70)
    print("TEST: French Question Generation")
    print("=" * 70)
    
    # Find a good definition chunk
    definition_chunk = None
    for chunk in chunks_fr:
        if chunk.semantic_type == "definition" and len(chunk.content) > 500:
            definition_chunk = chunk
            break
    
    if definition_chunk:
        print(f"\nChunk selected: {definition_chunk.chunk_id}")
        print(f"  Chapter: {definition_chunk.chapter_title}")
        print(f"  Section: {definition_chunk.section_title}")
        print(f"  Type: {definition_chunk.semantic_type}")
        print(f"  Size: {len(definition_chunk.content)} chars")
        print(f"\nContent preview:")
        print(f"  {definition_chunk.content[:300]}...")
        
        # Estimate question potential
        print(f"\nQuestion potential estimate:")
        potential = estimate_question_potential(definition_chunk.content)
        for qtype, count in potential.items():
            if count > 0:
                print(f"  - {qtype}: {count}")
        
        # Generate questions
        generator_fr = QuestionGenerator(
            llm_client=mock_client,
            model_name="gpt-4o-mini",
            language="fr",
            default_num_questions=3
        )
        
        questions_fr = generator_fr.generate_from_chunk(definition_chunk)
        
        print(f"\nGenerated {len(questions_fr)} questions:")
        for i, q in enumerate(questions_fr, 1):
            print(f"\n  Q{i}: {q.question}")
            print(f"      Type: {q.question_type.value}")
            print(f"      Difficulty: {q.difficulty.value}")
            print(f"      Concepts: {', '.join(q.key_concepts)}")
    
    # Show prompt that would be sent to LLM
    print("\n" + "=" * 70)
    print("PROMPT PREVIEW (what gets sent to LLM)")
    print("=" * 70)
    
    if definition_chunk:
        prompt = USER_PROMPT_TEMPLATE.format(
            num_questions=3,
            chapter=definition_chunk.chapter_title,
            section=definition_chunk.section_title,
            semantic_type=definition_chunk.semantic_type,
            pages=f"{definition_chunk.page_range[0]}-{definition_chunk.page_range[1]}",
            content=definition_chunk.content[:500] + "..."
        )
        print(prompt)
    
    # Show JSON output format
    print("\n" + "=" * 70)
    print("OUTPUT FORMAT (CandidateQuestion → JSON)")
    print("=" * 70)
    
    if questions_fr:
        print(json.dumps(questions_fr[0].to_dict(), indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_question_generator()
