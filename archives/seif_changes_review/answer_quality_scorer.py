"""
AnswerQualityScorer - Catch Hallucinations and Verify Answer Quality
======================================================================

Purpose:
--------
Independent quality assessment of generated answers to detect:
1. Hallucinations (facts not in the source chunk)
2. Incomplete answers (missing key information)
3. Poor grounding (answer doesn't cite the chunk)
4. Factual inconsistencies

This is SEPARATE from the CriticAgent - it focuses specifically on answer
quality and factual grounding, while CriticAgent evaluates overall QA pair quality.

Key Features:
-------------
- Entity overlap checking (answer mentions chunk entities)
- Factual grounding score (how well answer is supported by chunk)
- Completeness check (does answer address the question fully)
- Hallucination detection (claims not in source)
- Citation presence verification

Usage Example:
--------------
>>> from answer_quality_scorer import AnswerQualityScorer
>>> 
>>> scorer = AnswerQualityScorer()
>>> score = scorer.score_answer(
...     question="Qu'est-ce qu'une tribu?",
...     answer="Une tribu est une collection de sous-ensembles...",
...     chunk_content="1.1 Définition: Une tribu (ou σ-algèbre)..."
... )
>>> 
>>> print(f"Score: {score.overall_score:.2f}")
>>> print(f"Grounded: {score.is_grounded}")
>>> print(f"Issues: {score.issues}")

Author: Seif
Date: 2025
"""

import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import Counter
import warnings

# Try to import spaCy for NER (optional)
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


@dataclass
class AnswerQualityScore:
    """Quality score for a generated answer."""
    # Overall
    overall_score: float  # 0.0-1.0
    is_grounded: bool  # True if answer is well-grounded in chunk
    
    # Component scores
    entity_overlap_score: float  # How many chunk entities appear in answer
    keyword_overlap_score: float  # Keyword overlap between answer and chunk
    length_score: float  # Is answer appropriate length?
    completeness_score: float  # Does it address the question?
    citation_score: float  # Does it reference the source?
    
    # Issues detected
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Details
    chunk_entities: List[str] = field(default_factory=list)
    answer_entities: List[str] = field(default_factory=list)
    missing_entities: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "overall_score": self.overall_score,
            "is_grounded": self.is_grounded,
            "entity_overlap_score": self.entity_overlap_score,
            "keyword_overlap_score": self.keyword_overlap_score,
            "length_score": self.length_score,
            "completeness_score": self.completeness_score,
            "citation_score": self.citation_score,
            "issues": self.issues,
            "warnings": self.warnings,
            "chunk_entities": self.chunk_entities,
            "answer_entities": self.answer_entities,
            "missing_entities": self.missing_entities
        }


class AnswerQualityScorer:
    """
    Scores answer quality focusing on factual grounding and completeness.
    
    Attributes:
        min_length: Minimum acceptable answer length (words)
        max_length: Maximum acceptable answer length (words)
        entity_overlap_threshold: Minimum entity overlap for grounding
        keyword_overlap_threshold: Minimum keyword overlap for grounding
    """
    
    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 500,
        entity_overlap_threshold: float = 0.3,
        keyword_overlap_threshold: float = 0.4,
        use_spacy: bool = True
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.entity_overlap_threshold = entity_overlap_threshold
        self.keyword_overlap_threshold = keyword_overlap_threshold
        
        # Try to load spaCy model for NER
        self.nlp = None
        if use_spacy and SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("fr_core_news_sm")
                print("✅ Loaded spaCy model for entity recognition")
            except:
                print("⚠️  spaCy model not found. Install with: python -m spacy download fr_core_news_sm")
                print("   Falling back to pattern-based entity extraction")
        else:
            if use_spacy and not SPACY_AVAILABLE:
                print("⚠️  spaCy not installed. Falling back to pattern-based entity extraction")
    
    def score_answer(
        self,
        question: str,
        answer: str,
        chunk_content: str
    ) -> AnswerQualityScore:
        """
        Score answer quality and detect potential issues.
        
        Args:
            question: The question being answered
            answer: The generated answer
            chunk_content: The source chunk
            
        Returns:
            AnswerQualityScore with detailed breakdown
        """
        issues = []
        warnings_list = []
        
        # 1. Length check
        length_score, length_issue = self._check_length(answer)
        if length_issue:
            issues.append(length_issue)
        
        # 2. Entity overlap (hallucination detection)
        chunk_entities = self._extract_entities(chunk_content)
        answer_entities = self._extract_entities(answer)
        entity_overlap_score, entity_issues = self._check_entity_overlap(
            chunk_entities, answer_entities
        )
        issues.extend(entity_issues)
        
        # 3. Keyword overlap (grounding check)
        keyword_overlap_score, keyword_warnings = self._check_keyword_overlap(
            chunk_content, answer
        )
        warnings_list.extend(keyword_warnings)
        
        # 4. Completeness check (does it answer the question?)
        completeness_score, completeness_issues = self._check_completeness(
            question, answer
        )
        issues.extend(completeness_issues)
        
        # 5. Citation check (does it reference the source?)
        citation_score, citation_warnings = self._check_citations(answer)
        warnings_list.extend(citation_warnings)
        
        # Calculate overall score (weighted average)
        weights = {
            'entity': 0.30,
            'keyword': 0.25,
            'length': 0.15,
            'completeness': 0.20,
            'citation': 0.10
        }
        
        overall_score = (
            entity_overlap_score * weights['entity'] +
            keyword_overlap_score * weights['keyword'] +
            length_score * weights['length'] +
            completeness_score * weights['completeness'] +
            citation_score * weights['citation']
        )
        
        # Determine if answer is grounded
        is_grounded = (
            entity_overlap_score >= self.entity_overlap_threshold and
            keyword_overlap_score >= self.keyword_overlap_threshold and
            len(issues) == 0
        )
        
        # Find missing entities (potential hallucination indicators)
        missing_entities = [e for e in answer_entities if e not in chunk_entities]
        
        return AnswerQualityScore(
            overall_score=overall_score,
            is_grounded=is_grounded,
            entity_overlap_score=entity_overlap_score,
            keyword_overlap_score=keyword_overlap_score,
            length_score=length_score,
            completeness_score=completeness_score,
            citation_score=citation_score,
            issues=issues,
            warnings=warnings_list,
            chunk_entities=chunk_entities,
            answer_entities=answer_entities,
            missing_entities=missing_entities
        )
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities and key terms from text."""
        if self.nlp:
            # Use spaCy NER
            doc = self.nlp(text)
            entities = [ent.text for ent in doc.ents]
        else:
            # Fallback: pattern-based extraction
            entities = []
            
            # 1. Capitalized terms (likely proper nouns)
            capitalized = re.findall(r'\b[A-ZÀ-Ÿ][a-zà-ÿ]+(?:\s+[A-ZÀ-Ÿ][a-zà-ÿ]+)*\b', text)
            entities.extend(capitalized)
            
            # 2. Mathematical terms (Greek letters, variables)
            math_terms = re.findall(r'[α-ωΑ-Ω]|\\[a-zA-Z]+', text)
            entities.extend(math_terms)
            
            # 3. Technical terms (words with special chars)
            technical = re.findall(r'\b\w+[-_]\w+\b', text)
            entities.extend(technical)
        
        # Remove duplicates and normalize
        entities = list(set([e.strip() for e in entities if len(e.strip()) > 2]))
        return entities
    
    def _check_entity_overlap(
        self,
        chunk_entities: List[str],
        answer_entities: List[str]
    ) -> Tuple[float, List[str]]:
        """Check how many answer entities come from the chunk."""
        if len(answer_entities) == 0:
            return 1.0, []  # No entities in answer = no hallucination risk
        
        # Count entities in answer that appear in chunk
        overlap_count = sum(1 for e in answer_entities if e in chunk_entities)
        overlap_score = overlap_count / len(answer_entities)
        
        issues = []
        if overlap_score < self.entity_overlap_threshold:
            missing = [e for e in answer_entities if e not in chunk_entities]
            issues.append(
                f"Low entity overlap ({overlap_score:.0%}). "
                f"Potential hallucinations: {', '.join(missing[:3])}"
            )
        
        return overlap_score, issues
    
    def _check_keyword_overlap(
        self,
        chunk_content: str,
        answer: str
    ) -> Tuple[float, List[str]]:
        """Check keyword overlap between answer and chunk."""
        # Extract keywords (words 4+ chars, not stopwords)
        stopwords = {'dans', 'avec', 'pour', 'cette', 'sont', 'être', 'peut', 
                     'plus', 'leur', 'mais', 'tout', 'tous', 'fait', 'très'}
        
        def extract_keywords(text):
            words = re.findall(r'\b\w{4,}\b', text.lower())
            return [w for w in words if w not in stopwords]
        
        chunk_keywords = set(extract_keywords(chunk_content))
        answer_keywords = set(extract_keywords(answer))
        
        if len(answer_keywords) == 0:
            return 0.5, ["Answer has very few keywords"]
        
        overlap = len(answer_keywords & chunk_keywords)
        overlap_score = overlap / len(answer_keywords)
        
        warnings_list = []
        if overlap_score < self.keyword_overlap_threshold:
            warnings_list.append(
                f"Low keyword overlap ({overlap_score:.0%}). "
                f"Answer may not be well-grounded in source."
            )
        
        return overlap_score, warnings_list
    
    def _check_length(self, answer: str) -> Tuple[float, Optional[str]]:
        """Check if answer length is appropriate."""
        word_count = len(answer.split())
        
        if word_count < self.min_length:
            return 0.3, f"Answer too short ({word_count} words, min: {self.min_length})"
        elif word_count > self.max_length:
            return 0.7, f"Answer too long ({word_count} words, max: {self.max_length})"
        else:
            # Score based on optimal range (50-200 words)
            if 50 <= word_count <= 200:
                return 1.0, None
            elif word_count < 50:
                return 0.7 + (word_count - self.min_length) / (50 - self.min_length) * 0.3, None
            else:
                return 0.8, None
    
    def _check_completeness(
        self,
        question: str,
        answer: str
    ) -> Tuple[float, List[str]]:
        """Check if answer addresses the question."""
        issues = []
        
        # Extract question type
        question_lower = question.lower()
        
        # Check for question-specific requirements
        if question_lower.startswith(('qu\'est-ce', 'quelle est', 'quel est')):
            # Definition question - should contain "est" or "sont"
            if 'est' not in answer.lower() and 'sont' not in answer.lower():
                issues.append("Definition question but answer doesn't contain 'est/sont'")
        
        elif 'pourquoi' in question_lower or 'comment' in question_lower:
            # Explanation question - should be longer
            if len(answer.split()) < 30:
                issues.append("Explanation question but answer is too brief")
        
        elif 'différence' in question_lower or 'comparer' in question_lower:
            # Comparison question - should mention both elements
            if 'et' not in answer.lower():
                issues.append("Comparison question but answer doesn't clearly compare")
        
        # Check for empty/trivial answers
        if len(answer.strip()) < 20:
            issues.append("Answer is too trivial or empty")
        
        # Score based on issues
        completeness_score = 1.0 - (len(issues) * 0.3)
        completeness_score = max(0.0, completeness_score)
        
        return completeness_score, issues
    
    def _check_citations(self, answer: str) -> Tuple[float, List[str]]:
        """Check if answer cites or references the source."""
        warnings_list = []
        
        # Check for citation markers
        citation_patterns = [
            r'selon\s+',
            r'd\'après\s+',
            r'dans\s+le\s+(texte|document|cours|chapitre)',
            r'il\s+est\s+(indiqué|mentionné|écrit|précisé)',
            r'on\s+(lit|trouve|voit)',
            r'définit?\s+comme',
        ]
        
        has_citation = any(re.search(pattern, answer.lower()) for pattern in citation_patterns)
        
        if has_citation:
            citation_score = 1.0
        else:
            citation_score = 0.5
            warnings_list.append("Answer doesn't explicitly cite the source")
        
        return citation_score, warnings_list
    
    def score_batch(
        self,
        qa_pairs: List[Tuple[str, str, str]]
    ) -> List[AnswerQualityScore]:
        """
        Score multiple answers in batch.
        
        Args:
            qa_pairs: List of (question, answer, chunk_content) tuples
            
        Returns:
            List of AnswerQualityScore objects
        """
        return [
            self.score_answer(q, a, chunk)
            for q, a, chunk in qa_pairs
        ]
    
    def get_statistics(self, scores: List[AnswerQualityScore]) -> Dict:
        """Get statistics for a batch of scores."""
        if not scores:
            return {}
        
        return {
            'average_overall_score': sum(s.overall_score for s in scores) / len(scores),
            'average_entity_overlap': sum(s.entity_overlap_score for s in scores) / len(scores),
            'average_keyword_overlap': sum(s.keyword_overlap_score for s in scores) / len(scores),
            'grounded_count': sum(1 for s in scores if s.is_grounded),
            'grounded_percentage': sum(1 for s in scores if s.is_grounded) / len(scores) * 100,
            'total_issues': sum(len(s.issues) for s in scores),
            'total_warnings': sum(len(s.warnings) for s in scores),
            'common_issues': Counter([issue for s in scores for issue in s.issues]).most_common(5)
        }


# ============================================================================
# TEST SUITE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ANSWER QUALITY SCORER - TEST")
    print("=" * 70)
    print()
    
    scorer = AnswerQualityScorer()
    
    # Test data
    chunk_text = """
    1.1 Définition d'une Tribu
    
    Une tribu (ou σ-algèbre) est une collection de sous-ensembles d'un ensemble Ω
    qui satisfait les propriétés suivantes:
    1. Ω appartient à la tribu
    2. Si A appartient à la tribu, alors son complémentaire aussi
    3. La tribu est stable par union dénombrable
    
    Les tribus sont fondamentales en théorie de la mesure et en probabilités.
    Elles permettent de définir rigoureusement les événements mesurables.
    """
    
    test_cases = [
        {
            "name": "Good Answer (Well-grounded)",
            "question": "Qu'est-ce qu'une tribu?",
            "answer": "Une tribu (ou σ-algèbre) est une collection de sous-ensembles "
                     "qui satisfait trois propriétés: elle contient Ω, est stable par "
                     "complémentaire, et stable par union dénombrable. C'est un concept "
                     "fondamental en théorie de la mesure."
        },
        {
            "name": "Hallucination (Facts not in chunk)",
            "question": "Qu'est-ce qu'une tribu?",
            "answer": "Une tribu est une structure algébrique inventée par Borel en 1895 "
                     "pour formaliser la théorie des ensembles. Elle utilise des axiomes "
                     "de Zermelo-Fraenkel et est équivalente à une monade en théorie des catégories."
        },
        {
            "name": "Too Short",
            "question": "Qu'est-ce qu'une tribu?",
            "answer": "Une collection de sous-ensembles."
        },
        {
            "name": "Incomplete (Missing key info)",
            "question": "Qu'est-ce qu'une tribu?",
            "answer": "C'est quelque chose qui contient Ω et qui est stable."
        },
        {
            "name": "Good with Citation",
            "question": "Pourquoi les tribus sont-elles importantes?",
            "answer": "Selon le texte, les tribus sont fondamentales en théorie de la mesure "
                     "et en probabilités car elles permettent de définir rigoureusement "
                     "les événements mesurables."
        }
    ]
    
    print("📊 Testing Answer Quality Scorer:")
    print("-" * 70)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. {test['name']}")
        print(f"   Question: {test['question'][:60]}...")
        print(f"   Answer: {test['answer'][:60]}...")
        
        score = scorer.score_answer(
            question=test['question'],
            answer=test['answer'],
            chunk_content=chunk_text
        )
        
        # Determine status
        if score.overall_score >= 0.7 and score.is_grounded:
            status = "✅ GOOD"
        elif score.overall_score >= 0.5:
            status = "⚠️  WARNING"
        else:
            status = "❌ POOR"
        
        print(f"   {status} | Score: {score.overall_score:.2f} | Grounded: {score.is_grounded}")
        print(f"   Components: entity={score.entity_overlap_score:.2f}, "
              f"keyword={score.keyword_overlap_score:.2f}, "
              f"length={score.length_score:.2f}, "
              f"complete={score.completeness_score:.2f}")
        
        if score.issues:
            print(f"   🚨 Issues: {'; '.join(score.issues)}")
        if score.warnings:
            print(f"   ⚠️  Warnings: {'; '.join(score.warnings)}")
        if score.missing_entities:
            print(f"   🔍 Potential hallucinations: {', '.join(score.missing_entities[:3])}")
    
    print()
    print("=" * 70)
    print("✅ ANSWER QUALITY SCORER READY TO USE")
    print("=" * 70)
