"""
Entity Extraction Utilities for SoG Pipeline

This module provides utilities for extracting entities from text.
Can be integrated with NER models like spaCy, Stanford NER, or LLM-based extraction.
"""

from typing import List, Set, Dict
import re
from collections import Counter


class EntityExtractor:
    """
    Abstract base class for entity extraction.
    """
    
    def extract_entities(self, text: str) -> List[str]:
        """
        Extract entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of entity names
        """
        raise NotImplementedError("Implement this with actual NER model")


class SimpleEntityExtractor(EntityExtractor):
    """
    Simple rule-based entity extractor for demonstration.
    
    In production, replace with spaCy, Flair, or LLM-based extraction.
    """
    
    def __init__(self):
        # Simple pattern to extract capitalized words and phrases
        self.pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        
        # Common words to filter out
        self.stopwords = {
            'The', 'A', 'An', 'This', 'That', 'These', 'Those',
            'In', 'On', 'At', 'To', 'For', 'With', 'By', 'From',
            'About', 'As', 'Into', 'Through', 'During', 'Before',
            'After', 'Above', 'Below', 'Between', 'Under', 'Since',
            'However', 'Therefore', 'Thus', 'Hence', 'Moreover',
            'Furthermore', 'Nevertheless', 'Nonetheless',
            'First', 'Second', 'Third', 'Finally', 'Also'
        }
    
    def extract_entities(self, text: str) -> List[str]:
        """
        Extract entities using simple pattern matching.
        
        Args:
            text: Input text
            
        Returns:
            List of unique entity names
        """
        # Find all matches
        matches = re.findall(self.pattern, text)
        
        # Filter out stopwords
        entities = [m for m in matches if m not in self.stopwords]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)
        
        return unique_entities


class SpaCyEntityExtractor(EntityExtractor):
    """
    Entity extractor using spaCy NER.
    
    Requires: pip install spacy
    And: python -m spacy download en_core_web_sm
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize spaCy extractor.
        
        Args:
            model_name: spaCy model name
        """
        try:
            import spacy
            self.nlp = spacy.load(model_name)
        except ImportError:
            raise ImportError("spaCy not installed. Install with: pip install spacy")
        except OSError:
            raise OSError(f"spaCy model '{model_name}' not found. "
                        f"Download with: python -m spacy download {model_name}")
    
    def extract_entities(self, text: str) -> List[str]:
        """
        Extract entities using spaCy NER.
        
        Args:
            text: Input text
            
        Returns:
            List of entity names
        """
        doc = self.nlp(text)
        entities = [ent.text for ent in doc.ents]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)
        
        return unique_entities


class LLMEntityExtractor(EntityExtractor):
    """
    Entity extractor using LLM-based extraction.
    
    Can work with OpenAI, Anthropic, or other LLM APIs.
    """
    
    def __init__(self, llm_interface):
        """
        Initialize LLM extractor.
        
        Args:
            llm_interface: LLM interface for extraction
        """
        self.llm = llm_interface
    
    def extract_entities(self, text: str) -> List[str]:
        """
        Extract entities using LLM.
        
        Args:
            text: Input text
            
        Returns:
            List of entity names
        """
        prompt = f"""Extract all named entities (people, organizations, locations, concepts, etc.) from the following text.
Return only the entity names, one per line, without any additional formatting or explanation.

Text: {text}

Entities:"""
        
        response = self.llm.generate(prompt, max_tokens=500, temperature=0.3)
        
        # Parse response
        lines = response.strip().split('\n')
        entities = [line.strip() for line in lines if line.strip()]
        
        # Remove duplicates
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)
        
        return unique_entities


def extract_entities_from_corpus(paragraphs: List[str], 
                                extractor: EntityExtractor) -> List[List[str]]:
    """
    Extract entities from all paragraphs in a corpus.
    
    Args:
        paragraphs: List of paragraph texts
        extractor: Entity extractor instance
        
    Returns:
        List of entity lists, one per paragraph
    """
    entities_per_paragraph = []
    
    for para in paragraphs:
        entities = extractor.extract_entities(para)
        entities_per_paragraph.append(entities)
    
    return entities_per_paragraph


def analyze_entity_distribution(entities_per_paragraph: List[List[str]]) -> Dict:
    """
    Analyze the distribution of entities in a corpus.
    
    Args:
        entities_per_paragraph: List of entity lists
        
    Returns:
        Dictionary with analysis results
    """
    # Flatten all entities
    all_entities = [e for entities in entities_per_paragraph for e in entities]
    
    # Count frequencies
    entity_counts = Counter(all_entities)
    
    # Calculate statistics
    total_entities = len(all_entities)
    unique_entities = len(entity_counts)
    
    # Get long-tail entities (appear only once or twice)
    long_tail = sum(1 for count in entity_counts.values() if count <= 2)
    
    # Get top entities
    top_entities = entity_counts.most_common(10)
    
    return {
        'total_entities': total_entities,
        'unique_entities': unique_entities,
        'long_tail_entities': long_tail,
        'long_tail_percentage': (long_tail / unique_entities * 100) if unique_entities > 0 else 0,
        'avg_entities_per_paragraph': total_entities / len(entities_per_paragraph) if entities_per_paragraph else 0,
        'top_entities': top_entities
    }
