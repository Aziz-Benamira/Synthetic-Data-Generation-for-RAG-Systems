"""
Generation Strategies Module for Synthesize-on-Graph (SoG)

This module implements the two complementary generation strategies:
1. Chain-of-Thought (CoT) Generation
2. Contrastive Clarifying (CC) Generation
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from sampling_strategy import SamplingPath
from context_graph import Paragraph


class GenerationStrategy(Enum):
    """Enumeration of generation strategies."""
    CHAIN_OF_THOUGHT = "cot"
    CONTRASTIVE_CLARIFYING = "cc"


@dataclass
class SyntheticSample:
    """
    Represents a generated synthetic sample.
    """
    question: str
    answer: str
    context: List[str]  # List of paragraph texts used
    reasoning_type: str  # e.g., "multi-hop", "comparison"
    strategy_used: GenerationStrategy
    source_path: Optional[SamplingPath] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            'question': self.question,
            'answer': self.answer,
            'context': self.context,
            'reasoning_type': self.reasoning_type,
            'strategy': self.strategy_used.value,
        }


class LLMInterface:
    """
    Abstract interface for LLM interaction.
    In practice, this should be implemented with actual LLM APIs (GPT-4, Claude, etc.)
    """
    
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        raise NotImplementedError("Implement this with actual LLM API")
    
    def embed(self, text: str) -> List[float]:
        """
        Generate embeddings for text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        raise NotImplementedError("Implement this with actual embedding API")


class MockLLMInterface(LLMInterface):
    """
    Mock LLM interface for testing and demonstration.
    """
    
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate mock responses."""
        if "chain-of-thought" in prompt.lower() or "narrative" in prompt.lower():
            return self._generate_mock_cot_response()
        elif "contrastive" in prompt.lower() or "compare" in prompt.lower():
            return self._generate_mock_cc_response()
        else:
            return "Mock LLM response. Replace with actual LLM implementation."
    
    def _generate_mock_cot_response(self) -> str:
        return """Question: How do the concepts in the provided documents connect to form a coherent understanding?

Answer: Let me walk through this step by step:
1. First, we examine the initial concept which provides the foundation...
2. Building on that, we can see how the second element relates through...
3. This connection leads us to understand that...
4. Finally, bringing it all together, we can conclude that...

Therefore, the answer demonstrates multi-hop reasoning across the provided contexts."""
    
    def _generate_mock_cc_response(self) -> str:
        return """Question: What are the key differences and similarities between the concepts presented in the documents?

Answer: Comparing the provided information:
- The first concept emphasizes... while the second focuses on...
- A key similarity is that both share the characteristic of...
- However, they differ significantly in terms of...
- The distinctive feature of the first is... whereas the second is characterized by...

In conclusion, while related, these concepts serve different purposes in the broader context."""


class ChainOfThoughtGenerator:
    """
    Implements Chain-of-Thought (CoT) generation strategy.
    
    Creates step-by-step narratives connecting fragments across documents
    with logical flow through distinct phases:
    - Initiation
    - Development
    - Turning points
    - Conclusion
    """
    
    def __init__(self, llm: LLMInterface):
        self.llm = llm
    
    def _construct_narrative_prompt(self, path: SamplingPath) -> str:
        """
        Construct a prompt to guide the LLM in creating a narrative.
        
        Args:
            path: Sampling path with connected entities and paragraphs
            
        Returns:
            Prompt string
        """
        paragraphs = path.get_paragraphs()
        entities = path.get_entities()
        
        prompt = """You are a knowledge synthesis expert. Given the following text fragments from different documents, create a coherent narrative that connects them through logical relationships.

The narrative should follow these phases:
1. INITIATION: Introduce the core concept from the first fragment
2. DEVELOPMENT: Build upon the concept with information from subsequent fragments
3. TURNING POINTS: Highlight key transitions or contrasts between ideas
4. CONCLUSION: Synthesize the information into a unified understanding

Text Fragments:
"""
        
        for i, (entity, para) in enumerate(zip(entities, paragraphs)):
            prompt += f"\n--- Fragment {i+1} (Entity: {entity}) ---\n{para.text}\n"
        
        prompt += """
Now, create a chain-of-thought narrative that connects these fragments, then generate:
1. A question that requires multi-hop reasoning across these fragments
2. An answer that demonstrates step-by-step reasoning through the narrative

Format your response as:
Question: [Your question]

Answer: [Step-by-step answer with clear reasoning chain]
"""
        
        return prompt
    
    def generate(self, path: SamplingPath) -> Optional[SyntheticSample]:
        """
        Generate a synthetic sample using CoT strategy.
        
        Args:
            path: Sampling path to generate from
            
        Returns:
            Generated synthetic sample or None if generation fails
        """
        if len(path) < 2:
            return None  # Need at least 2 hops for meaningful CoT
        
        # Construct prompt
        prompt = self._construct_narrative_prompt(path)
        
        # Generate response
        response = self.llm.generate(prompt, max_tokens=1500, temperature=0.7)
        
        # Parse response
        question, answer = self._parse_response(response)
        
        if not question or not answer:
            return None
        
        # Create synthetic sample
        paragraphs = path.get_paragraphs()
        context = [para.text for para in paragraphs]
        
        return SyntheticSample(
            question=question,
            answer=answer,
            context=context,
            reasoning_type="multi-hop",
            strategy_used=GenerationStrategy.CHAIN_OF_THOUGHT,
            source_path=path
        )
    
    def _parse_response(self, response: str) -> Tuple[str, str]:
        """
        Parse question and answer from LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            Tuple of (question, answer)
        """
        question = ""
        answer = ""
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.lower().startswith('question:'):
                current_section = 'question'
                question = line[9:].strip()
            elif line.lower().startswith('answer:'):
                current_section = 'answer'
                answer = line[7:].strip()
            elif current_section == 'question' and line:
                question += ' ' + line
            elif current_section == 'answer' and line:
                answer += ' ' + line
        
        return question.strip(), answer.strip()


class ContrastiveClarifyingGenerator:
    """
    Implements Contrastive Clarifying (CC) generation strategy.
    
    Designed for sparse entities with limited graph connections.
    Generates comparative analyses that contrast and compare multiple
    text fragments, highlighting discriminative information and nuances.
    """
    
    def __init__(self, llm: LLMInterface):
        self.llm = llm
    
    def _construct_comparison_prompt(self, path: SamplingPath) -> str:
        """
        Construct a prompt to guide the LLM in creating a comparative analysis.
        
        Args:
            path: Sampling path with entities and paragraphs to compare
            
        Returns:
            Prompt string
        """
        paragraphs = path.get_paragraphs()
        entities = path.get_entities()
        
        prompt = """You are a comparative analysis expert. Given the following text fragments, create a detailed comparison that highlights similarities, differences, and discriminative features.

Your analysis should:
1. Identify key similarities across the fragments
2. Highlight important differences and contrasts
3. Clarify nuances and distinctive features
4. Provide discriminative information that distinguishes each concept

Text Fragments:
"""
        
        for i, (entity, para) in enumerate(zip(entities, paragraphs)):
            prompt += f"\n--- Fragment {i+1} (Entity: {entity}) ---\n{para.text}\n"
        
        prompt += """
Now, create a comparative analysis and generate:
1. A question that asks about the relationships, similarities, or differences between these concepts
2. An answer that provides a detailed comparison with clear discriminative features

Format your response as:
Question: [Your question]

Answer: [Detailed comparative answer]
"""
        
        return prompt
    
    def generate(self, path: SamplingPath) -> Optional[SyntheticSample]:
        """
        Generate a synthetic sample using CC strategy.
        
        Args:
            path: Sampling path to generate from
            
        Returns:
            Generated synthetic sample or None if generation fails
        """
        if len(path) < 2:
            return None  # Need at least 2 entities to compare
        
        # Construct prompt
        prompt = self._construct_comparison_prompt(path)
        
        # Generate response
        response = self.llm.generate(prompt, max_tokens=1500, temperature=0.7)
        
        # Parse response
        question, answer = self._parse_response(response)
        
        if not question or not answer:
            return None
        
        # Create synthetic sample
        paragraphs = path.get_paragraphs()
        context = [para.text for para in paragraphs]
        
        return SyntheticSample(
            question=question,
            answer=answer,
            context=context,
            reasoning_type="comparison",
            strategy_used=GenerationStrategy.CONTRASTIVE_CLARIFYING,
            source_path=path
        )
    
    def _parse_response(self, response: str) -> Tuple[str, str]:
        """
        Parse question and answer from LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            Tuple of (question, answer)
        """
        question = ""
        answer = ""
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.lower().startswith('question:'):
                current_section = 'question'
                question = line[9:].strip()
            elif line.lower().startswith('answer:'):
                current_section = 'answer'
                answer = line[7:].strip()
            elif current_section == 'question' and line:
                question += ' ' + line
            elif current_section == 'answer' and line:
                answer += ' ' + line
        
        return question.strip(), answer.strip()


class AdaptiveStrategySelector:
    """
    Dynamically selects between CoT and CC generation strategies
    based on entity graph connectivity and utilization rates.
    """
    
    def __init__(self, 
                 cot_generator: ChainOfThoughtGenerator,
                 cc_generator: ContrastiveClarifyingGenerator,
                 sparse_threshold: int = 3,
                 utilization_threshold: float = 0.1):
        """
        Initialize strategy selector.
        
        Args:
            cot_generator: CoT generator instance
            cc_generator: CC generator instance
            sparse_threshold: Threshold for sparse entity detection
            utilization_threshold: Threshold for low utilization rate
        """
        self.cot_generator = cot_generator
        self.cc_generator = cc_generator
        self.sparse_threshold = sparse_threshold
        self.utilization_threshold = utilization_threshold
    
    def select_strategy(self, path: SamplingPath, context_graph) -> GenerationStrategy:
        """
        Select the appropriate generation strategy for a path.
        
        Strategy selection criteria:
        - Use CC if path contains sparse entities (degree < threshold)
        - Use CC if entity utilization rate is below threshold
        - Otherwise use CoT
        
        Args:
            path: Sampling path
            context_graph: Context graph for entity analysis
            
        Returns:
            Selected generation strategy
        """
        entities = path.get_entities()
        
        # Check for sparse entities
        for entity in entities:
            degree = context_graph.get_entity_degree(entity)
            if degree < self.sparse_threshold:
                return GenerationStrategy.CONTRASTIVE_CLARIFYING
        
        # Check utilization rates (would need total_samples from generation process)
        # For now, use a simplified heuristic based on entity frequency
        long_tail_entities = context_graph.get_long_tail_entities()
        if any(entity in long_tail_entities for entity in entities):
            return GenerationStrategy.CONTRASTIVE_CLARIFYING
        
        return GenerationStrategy.CHAIN_OF_THOUGHT
    
    def generate(self, path: SamplingPath, context_graph) -> Optional[SyntheticSample]:
        """
        Generate a synthetic sample using the selected strategy.
        
        Args:
            path: Sampling path
            context_graph: Context graph for strategy selection
            
        Returns:
            Generated synthetic sample
        """
        strategy = self.select_strategy(path, context_graph)
        
        if strategy == GenerationStrategy.CHAIN_OF_THOUGHT:
            return self.cot_generator.generate(path)
        else:
            return self.cc_generator.generate(path)
