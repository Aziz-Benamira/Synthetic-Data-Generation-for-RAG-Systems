"""
Chain-of-Thought Validator - Verify Reasoning Quality
======================================================

Purpose:
--------
Validates the logical structure and reasoning quality of answers, especially
for explanatory and analytical questions. Detects:
1. Logical jumps (missing intermediate steps)
2. Circular reasoning
3. Unsupported claims
4. Missing causality in "why/how" answers
5. Poor argument structure

This is particularly important for:
- Explanation questions ("Comment...", "Pourquoi...")
- Analysis questions ("Analyser...", "Évaluer...")
- Application questions ("Appliquer...")

Key Features:
-------------
- Reasoning step extraction
- Logical flow validation
- Causality checking (for why/how questions)
- Argument structure analysis
- Coherence scoring

Usage Example:
--------------
>>> from chain_of_thought_validator import ChainOfThoughtValidator
>>> 
>>> validator = ChainOfThoughtValidator()
>>> result = validator.validate(
...     question="Pourquoi les tribus sont importantes en probabilités?",
...     answer="Les tribus permettent de définir les événements mesurables..."
... )
>>> 
>>> print(f"Valid: {result.is_valid}")
>>> print(f"Reasoning steps: {len(result.reasoning_steps)}")
>>> print(f"Issues: {result.issues}")

Author: Seif
Date: 2025
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class ReasoningType(Enum):
    """Type of reasoning in the answer."""
    CAUSAL = "causal"  # Because X, therefore Y
    SEQUENTIAL = "sequential"  # First X, then Y, finally Z
    COMPARATIVE = "comparative"  # X is better/different than Y because...
    DEDUCTIVE = "deductive"  # From premise X, we conclude Y
    EXPLANATORY = "explanatory"  # X works by doing Y
    NONE = "none"  # No clear reasoning structure


@dataclass
class ReasoningStep:
    """A single step in a reasoning chain."""
    step_number: int
    content: str
    connective: Optional[str] = None  # donc, ainsi, car, parce que, etc.
    is_supported: bool = True  # Is this step logically supported?


@dataclass
class ChainOfThoughtValidation:
    """Result of chain-of-thought validation."""
    # Overall
    is_valid: bool
    overall_score: float  # 0.0-1.0
    
    # Reasoning structure
    reasoning_type: ReasoningType
    reasoning_steps: List[ReasoningStep]
    num_steps: int
    
    # Quality metrics
    has_causality: bool  # For why/how questions
    has_logical_flow: bool
    has_circular_reasoning: bool
    has_unsupported_claims: bool
    
    # Component scores
    structure_score: float  # Is there clear structure?
    causality_score: float  # Are causal links present?
    coherence_score: float  # Does it flow logically?
    completeness_score: float  # Are all steps present?
    
    # Issues detected
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "is_valid": self.is_valid,
            "overall_score": self.overall_score,
            "reasoning_type": self.reasoning_type.value,
            "num_steps": self.num_steps,
            "has_causality": self.has_causality,
            "has_logical_flow": self.has_logical_flow,
            "has_circular_reasoning": self.has_circular_reasoning,
            "has_unsupported_claims": self.has_unsupported_claims,
            "structure_score": self.structure_score,
            "causality_score": self.causality_score,
            "coherence_score": self.coherence_score,
            "completeness_score": self.completeness_score,
            "issues": self.issues,
            "warnings": self.warnings,
            "reasoning_steps": [
                {"step": s.step_number, "content": s.content, "connective": s.connective}
                for s in self.reasoning_steps
            ]
        }


class ChainOfThoughtValidator:
    """
    Validates reasoning quality in answers.
    
    Attributes:
        min_steps_for_explanation: Minimum reasoning steps for explanation questions
        causality_keywords: Keywords that indicate causal reasoning
        logical_connectives: Words that connect reasoning steps
    """
    
    # Causal connectives
    CAUSALITY_KEYWORDS = {
        'fr': ['car', 'parce que', 'puisque', 'donc', 'ainsi', 'par conséquent',
               'c\'est pourquoi', 'en effet', 'grâce à', 'à cause de', 'permet de'],
        'en': ['because', 'since', 'therefore', 'thus', 'hence', 'consequently',
               'as a result', 'due to', 'thanks to', 'enables']
    }
    
    # Sequential connectives
    SEQUENTIAL_KEYWORDS = {
        'fr': ['d\'abord', 'ensuite', 'puis', 'enfin', 'finalement', 'premièrement',
               'deuxièmement', 'd\'une part', 'd\'autre part'],
        'en': ['first', 'then', 'next', 'finally', 'firstly', 'secondly',
               'on one hand', 'on the other hand']
    }
    
    # Comparative connectives
    COMPARATIVE_KEYWORDS = {
        'fr': ['contrairement à', 'par rapport à', 'tandis que', 'alors que',
               'en revanche', 'au contraire', 'plus que', 'moins que'],
        'en': ['compared to', 'unlike', 'whereas', 'while', 'however',
               'on the contrary', 'more than', 'less than']
    }
    
    def __init__(
        self,
        language: str = "fr",
        min_steps_for_explanation: int = 2,
        strict_mode: bool = False
    ):
        self.language = language
        self.min_steps_for_explanation = min_steps_for_explanation
        self.strict_mode = strict_mode
        
        # Get language-specific keywords
        self.causality_keywords = self.CAUSALITY_KEYWORDS.get(language, self.CAUSALITY_KEYWORDS['fr'])
        self.sequential_keywords = self.SEQUENTIAL_KEYWORDS.get(language, self.SEQUENTIAL_KEYWORDS['fr'])
        self.comparative_keywords = self.COMPARATIVE_KEYWORDS.get(language, self.COMPARATIVE_KEYWORDS['fr'])
    
    def validate(
        self,
        question: str,
        answer: str,
        question_type: Optional[str] = None
    ) -> ChainOfThoughtValidation:
        """
        Validate reasoning in an answer.
        
        Args:
            question: The question being answered
            answer: The answer to validate
            question_type: Optional question type (explanation, analysis, etc.)
            
        Returns:
            ChainOfThoughtValidation with detailed analysis
        """
        issues = []
        warnings = []
        
        # 1. Detect reasoning type
        reasoning_type = self._detect_reasoning_type(question, answer)
        
        # 2. Extract reasoning steps
        reasoning_steps = self._extract_reasoning_steps(answer, reasoning_type)
        
        # 3. Check if question requires reasoning
        requires_reasoning = self._requires_reasoning(question, question_type)
        
        # 4. Check causality (for why/how questions)
        has_causality = self._check_causality(answer)
        if requires_reasoning and reasoning_type == ReasoningType.CAUSAL and not has_causality:
            issues.append("Causal question but answer lacks causal connectives (car, parce que, donc)")
        
        # 5. Check logical flow
        has_logical_flow, flow_issues = self._check_logical_flow(reasoning_steps)
        issues.extend(flow_issues)
        
        # 6. Check for circular reasoning
        has_circular = self._detect_circular_reasoning(answer)
        if has_circular:
            issues.append("Circular reasoning detected (conclusion restates premise)")
        
        # 7. Check for unsupported claims
        has_unsupported, unsupported_issues = self._check_unsupported_claims(answer)
        issues.extend(unsupported_issues)
        
        # 8. Calculate component scores
        structure_score = self._score_structure(reasoning_steps, requires_reasoning)
        causality_score = 1.0 if has_causality or not requires_reasoning else 0.3
        coherence_score = 1.0 if has_logical_flow else 0.5
        completeness_score = self._score_completeness(reasoning_steps, requires_reasoning)
        
        # 9. Calculate overall score
        weights = {
            'structure': 0.25,
            'causality': 0.25,
            'coherence': 0.30,
            'completeness': 0.20
        }
        
        overall_score = (
            structure_score * weights['structure'] +
            causality_score * weights['causality'] +
            coherence_score * weights['coherence'] +
            completeness_score * weights['completeness']
        )
        
        # 10. Determine validity
        is_valid = (
            overall_score >= 0.6 and
            len(issues) == 0 and
            (has_causality or not requires_reasoning)
        )
        
        return ChainOfThoughtValidation(
            is_valid=is_valid,
            overall_score=overall_score,
            reasoning_type=reasoning_type,
            reasoning_steps=reasoning_steps,
            num_steps=len(reasoning_steps),
            has_causality=has_causality,
            has_logical_flow=has_logical_flow,
            has_circular_reasoning=has_circular,
            has_unsupported_claims=has_unsupported,
            structure_score=structure_score,
            causality_score=causality_score,
            coherence_score=coherence_score,
            completeness_score=completeness_score,
            issues=issues,
            warnings=warnings
        )
    
    def _detect_reasoning_type(self, question: str, answer: str) -> ReasoningType:
        """Detect the type of reasoning expected."""
        question_lower = question.lower()
        answer_lower = answer.lower()
        
        # Causal reasoning (why/how questions)
        if 'pourquoi' in question_lower or 'comment' in question_lower:
            if any(kw in answer_lower for kw in self.causality_keywords):
                return ReasoningType.CAUSAL
        
        # Sequential reasoning (process/steps)
        if any(kw in answer_lower for kw in self.sequential_keywords):
            return ReasoningType.SEQUENTIAL
        
        # Comparative reasoning
        if 'différence' in question_lower or 'comparer' in question_lower:
            if any(kw in answer_lower for kw in self.comparative_keywords):
                return ReasoningType.COMPARATIVE
        
        # Check for causal keywords in answer
        if any(kw in answer_lower for kw in self.causality_keywords):
            return ReasoningType.CAUSAL
        
        return ReasoningType.EXPLANATORY
    
    def _extract_reasoning_steps(
        self,
        answer: str,
        reasoning_type: ReasoningType
    ) -> List[ReasoningStep]:
        """Extract individual reasoning steps from answer."""
        steps = []
        
        # Split by sentences
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Find connectives and build steps
        for i, sentence in enumerate(sentences):
            connective = None
            
            # Check for connectives
            for kw in self.causality_keywords + self.sequential_keywords + self.comparative_keywords:
                if kw in sentence.lower():
                    connective = kw
                    break
            
            step = ReasoningStep(
                step_number=i + 1,
                content=sentence,
                connective=connective
            )
            steps.append(step)
        
        return steps
    
    def _requires_reasoning(self, question: str, question_type: Optional[str] = None) -> bool:
        """Check if question requires reasoning (not just facts)."""
        question_lower = question.lower()
        
        # Explicit reasoning questions
        reasoning_keywords = ['pourquoi', 'comment', 'expliquer', 'justifier',
                             'analyser', 'évaluer', 'démontrer', 'prouver']
        
        if any(kw in question_lower for kw in reasoning_keywords):
            return True
        
        # Check question type
        if question_type in ['explanation', 'analysis', 'application']:
            return True
        
        return False
    
    def _check_causality(self, answer: str) -> bool:
        """Check if answer contains causal reasoning."""
        answer_lower = answer.lower()
        return any(kw in answer_lower for kw in self.causality_keywords)
    
    def _check_logical_flow(self, steps: List[ReasoningStep]) -> Tuple[bool, List[str]]:
        """Check if reasoning steps flow logically."""
        issues = []
        
        if len(steps) < 2:
            return True, []  # Single sentence doesn't need flow checking
        
        # Check for abrupt jumps (steps without connectives)
        unconnected_count = sum(1 for s in steps[1:] if s.connective is None)
        
        if unconnected_count > len(steps) * 0.5:
            issues.append(
                f"Reasoning lacks connectives ({unconnected_count}/{len(steps)} steps unconnected)"
            )
        
        has_flow = len(issues) == 0
        return has_flow, issues
    
    def _detect_circular_reasoning(self, answer: str) -> bool:
        """Detect circular reasoning (conclusion restates premise)."""
        sentences = [s.strip() for s in re.split(r'[.!?]+', answer) if s.strip()]
        
        if len(sentences) < 2:
            return False
        
        # Simple heuristic: check if first and last sentence are very similar
        first = set(sentences[0].lower().split())
        last = set(sentences[-1].lower().split())
        
        # Remove stopwords
        stopwords = {'le', 'la', 'les', 'un', 'une', 'des', 'est', 'sont', 'et', 'ou'}
        first = first - stopwords
        last = last - stopwords
        
        if len(first) == 0 or len(last) == 0:
            return False
        
        # Check overlap
        overlap = len(first & last) / min(len(first), len(last))
        
        return overlap > 0.7  # 70% word overlap = likely circular
    
    def _check_unsupported_claims(self, answer: str) -> Tuple[bool, List[str]]:
        """Check for unsupported claims (strong statements without justification)."""
        issues = []
        
        # Strong claim words
        strong_claims = [
            r'\b(toujours|jamais|tous|aucun|nécessairement|certainement)\b',
            r'\b(always|never|all|none|necessarily|certainly)\b'
        ]
        
        for pattern in strong_claims:
            matches = re.findall(pattern, answer.lower())
            if matches:
                # Check if followed by justification
                for match in matches:
                    # Simple check: is there a "car" or "parce que" nearby?
                    context = answer.lower()
                    match_pos = context.find(match)
                    nearby = context[match_pos:match_pos+100]
                    
                    if not any(kw in nearby for kw in ['car', 'parce que', 'puisque']):
                        issues.append(f"Strong claim '{match}' without justification")
        
        has_unsupported = len(issues) > 0
        return has_unsupported, issues
    
    def _score_structure(self, steps: List[ReasoningStep], requires_reasoning: bool) -> float:
        """Score the structure quality."""
        if not requires_reasoning:
            return 1.0
        
        if len(steps) == 0:
            return 0.0
        
        # Check for connectives
        connected_count = sum(1 for s in steps if s.connective)
        connection_ratio = connected_count / len(steps)
        
        # Check for adequate length
        if len(steps) < self.min_steps_for_explanation:
            length_score = len(steps) / self.min_steps_for_explanation
        else:
            length_score = 1.0
        
        # Combine
        structure_score = (connection_ratio * 0.6 + length_score * 0.4)
        
        return structure_score
    
    def _score_completeness(self, steps: List[ReasoningStep], requires_reasoning: bool) -> float:
        """Score completeness of reasoning."""
        if not requires_reasoning:
            return 1.0
        
        # Check if reasoning is developed enough
        if len(steps) < self.min_steps_for_explanation:
            return len(steps) / self.min_steps_for_explanation * 0.7
        
        # Check average step length (should be substantive)
        avg_length = sum(len(s.content.split()) for s in steps) / len(steps)
        
        if avg_length < 5:
            return 0.6  # Steps too brief
        elif avg_length > 30:
            return 0.8  # Steps might be too verbose
        else:
            return 1.0
    
    def validate_batch(
        self,
        qa_pairs: List[Tuple[str, str]]
    ) -> List[ChainOfThoughtValidation]:
        """Validate multiple answers in batch."""
        return [self.validate(q, a) for q, a in qa_pairs]


# ============================================================================
# TEST SUITE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CHAIN-OF-THOUGHT VALIDATOR - TEST")
    print("=" * 70)
    print()
    
    validator = ChainOfThoughtValidator()
    
    test_cases = [
        {
            "name": "Good Causal Reasoning",
            "question": "Pourquoi les tribus sont-elles importantes en probabilités?",
            "answer": "Les tribus sont importantes car elles permettent de définir rigoureusement "
                     "les événements mesurables. En effet, sans tribu, on ne peut pas appliquer "
                     "la théorie de la mesure. Par conséquent, elles sont fondamentales pour "
                     "construire une théorie des probabilités cohérente."
        },
        {
            "name": "Missing Causality",
            "question": "Pourquoi utilise-t-on des tribus?",
            "answer": "On utilise des tribus. Elles sont dans la théorie de la mesure. "
                     "Les probabilités les utilisent aussi."
        },
        {
            "name": "Circular Reasoning",
            "question": "Qu'est-ce qu'une tribu?",
            "answer": "Une tribu est une tribu qui satisfait les propriétés d'une tribu."
        },
        {
            "name": "Good Sequential Reasoning",
            "question": "Comment construire une tribu?",
            "answer": "D'abord, on prend un ensemble Ω. Ensuite, on choisit des sous-ensembles "
                     "qui satisfont les propriétés. Puis, on vérifie la stabilité par complémentaire. "
                     "Enfin, on vérifie la stabilité par union dénombrable."
        },
        {
            "name": "Unsupported Strong Claim",
            "question": "Les tribus sont-elles utiles?",
            "answer": "Les tribus sont toujours nécessaires et aucune théorie ne peut s'en passer."
        }
    ]
    
    print("🔍 Testing Chain-of-Thought Validator:")
    print("-" * 70)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. {test['name']}")
        print(f"   Question: {test['question']}")
        print(f"   Answer: {test['answer'][:80]}...")
        
        result = validator.validate(
            question=test['question'],
            answer=test['answer']
        )
        
        # Status
        if result.is_valid:
            status = "✅ VALID"
        elif result.overall_score >= 0.5:
            status = "⚠️  WARNING"
        else:
            status = "❌ INVALID"
        
        print(f"   {status} | Score: {result.overall_score:.2f}")
        print(f"   Type: {result.reasoning_type.value} | Steps: {result.num_steps}")
        print(f"   Scores: structure={result.structure_score:.2f}, "
              f"causality={result.causality_score:.2f}, "
              f"coherence={result.coherence_score:.2f}")
        print(f"   Causality: {result.has_causality}, "
              f"Flow: {result.has_logical_flow}, "
              f"Circular: {result.has_circular_reasoning}")
        
        if result.issues:
            print(f"   🚨 Issues: {'; '.join(result.issues)}")
    
    print()
    print("=" * 70)
    print("✅ CHAIN-OF-THOUGHT VALIDATOR READY TO USE")
    print("=" * 70)
