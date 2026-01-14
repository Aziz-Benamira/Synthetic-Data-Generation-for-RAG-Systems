"""
Difficulty Estimator
====================

Estimates question difficulty using multiple heuristics.
No LLM needed - fast rule-based system.

Difficulty Levels:
- easy: Simple recall, short questions (e.g., "Qu'est-ce que X?")
- medium: Requires understanding, moderate complexity
- hard: Requires analysis, synthesis, or complex reasoning
"""

from typing import Tuple, List, Dict
import re
from collections import Counter


class DifficultyEstimator:
    """
    Estimates question difficulty based on multiple factors:
    1. Question complexity (length, structure)
    2. Question type (factoid=easy, analysis=hard)
    3. Multi-part questions
    4. Technical term density
    5. Cognitive verbs (Bloom's taxonomy indicators)
    """
    
    # Question type difficulty weights
    TYPE_DIFFICULTY = {
        "factoid": 0.0,        # Easiest (simple recall)
        "definition": 0.2,
        "comparison": 0.6,     # Medium (requires understanding)
        "explanation": 0.7,
        "application": 0.8,    # Harder (requires transfer)
        "calculation": 0.9,
        "analysis": 1.0        # Hardest (critical thinking)
    }
    
    # Cognitive verbs indicating difficulty
    COGNITIVE_VERBS = {
        "low": [  # Remember, Understand
            "définir", "lister", "nommer", "identifier", "reconnaître",
            "décrire", "expliquer", "résumer", "classer", "donner"
        ],
        "medium": [  # Apply, Analyze
            "appliquer", "utiliser", "démontrer", "calculer",
            "analyser", "comparer", "distinguer", "examiner"
        ],
        "high": [  # Evaluate, Create
            "évaluer", "juger", "critiquer", "justifier", "argumenter",
            "concevoir", "créer", "proposer", "synthétiser", "formuler"
        ]
    }
    
    def __init__(self, classifier=None):
        """
        Initialize difficulty estimator.
        
        Args:
            classifier: Optional QuestionTypeClassifier instance
        """
        self.classifier = classifier
    
    def estimate(self, question: str, question_type: str = None) -> Tuple[str, float, Dict[str, float]]:
        """
        Estimate question difficulty.
        
        Args:
            question: Question text
            question_type: Pre-classified type (optional, will classify if None)
            
        Returns:
            Tuple of (difficulty_label, confidence_score, factor_scores)
            - difficulty_label: "easy" | "medium" | "hard"
            - confidence_score: 0-1 (how confident we are)
            - factor_scores: Dict of individual factor contributions
        """
        if not question or not question.strip():
            return "medium", 0.5, {}
        
        # Get question type if not provided
        if question_type is None and self.classifier:
            question_type = self.classifier.classify(question)
        
        # Calculate individual factors
        factors = {}
        
        # Factor 1: Question length/complexity (0-1)
        factors["length"] = self._length_complexity(question)
        
        # Factor 2: Question type difficulty (0-1)
        factors["type"] = self.TYPE_DIFFICULTY.get(question_type, 0.5) if question_type else 0.5
        
        # Factor 3: Cognitive verb level (0-1)
        factors["cognitive"] = self._cognitive_complexity(question)
        
        # Factor 4: Multi-part question (0-1)
        factors["multipart"] = self._multipart_complexity(question)
        
        # Factor 5: Technical term density (0-1)
        factors["technical"] = self._technical_density(question)
        
        # Factor 6: Syntactic complexity (0-1)
        factors["syntax"] = self._syntactic_complexity(question)
        
        # Weighted average (type and cognitive are most important)
        weights = {
            "length": 0.15,
            "type": 0.30,       # Most important
            "cognitive": 0.25,  # Second most important
            "multipart": 0.10,
            "technical": 0.10,
            "syntax": 0.10
        }
        
        weighted_score = sum(factors[k] * weights[k] for k in factors.keys())
        
        # Normalize to 0-1
        normalized = max(0.0, min(1.0, weighted_score))
        
        # Map to difficulty labels with thresholds
        if normalized < 0.35:
            difficulty = "easy"
            confidence = 1.0 - abs(normalized - 0.2)  # Distance from center of range
        elif normalized < 0.65:
            difficulty = "medium"
            confidence = 1.0 - abs(normalized - 0.5)
        else:
            difficulty = "hard"
            confidence = 1.0 - abs(normalized - 0.8)
        
        confidence = max(0.5, min(1.0, confidence))  # Keep confidence reasonable
        
        return difficulty, confidence, factors
    
    def _length_complexity(self, question: str) -> float:
        """
        Estimate complexity based on question length.
        Longer questions are generally more complex.
        """
        word_count = len(question.split())
        
        if word_count < 8:
            return 0.0  # Very short = easy
        elif word_count < 15:
            return 0.4
        elif word_count < 25:
            return 0.7
        else:
            return 1.0  # Very long = complex
    
    def _cognitive_complexity(self, question: str) -> float:
        """
        Detect cognitive complexity based on verbs (Bloom's taxonomy).
        """
        question_lower = question.lower()
        
        # Check for high-level verbs first
        for verb in self.COGNITIVE_VERBS["high"]:
            if verb in question_lower:
                return 1.0
        
        # Then medium-level
        for verb in self.COGNITIVE_VERBS["medium"]:
            if verb in question_lower:
                return 0.6
        
        # Then low-level
        for verb in self.COGNITIVE_VERBS["low"]:
            if verb in question_lower:
                return 0.2
        
        return 0.5  # Default if no verbs detected
    
    def _multipart_complexity(self, question: str) -> float:
        """
        Check if question has multiple parts (increases difficulty).
        Examples: "Définir X et Y", "Comparer A, B et C", "Pourquoi X et comment Y?"
        """
        # Count conjunctions and multiple question marks
        conjunctions = len(re.findall(r'\bet\b|\bou\b|,', question))
        multiple_questions = question.count('?')
        
        score = 0.0
        
        if conjunctions >= 3 or multiple_questions > 1:
            score = 1.0  # Very multi-part
        elif conjunctions >= 1:
            score = 0.5  # Somewhat multi-part
        
        # Also check for explicit multi-part patterns
        multipart_patterns = [
            r'\bet\s+(ensuite|puis|également)',
            r'\bd\'une part.+d\'autre part',
            r'\bpremièrement.+deuxièmement',
            r'\bquels?.+et\s+quels?'
        ]
        
        for pattern in multipart_patterns:
            if re.search(pattern, question.lower()):
                score = max(score, 0.8)
        
        return score
    
    def _technical_density(self, question: str) -> float:
        """
        Estimate density of technical/specialized terms.
        More technical terms = harder question.
        """
        words = question.split()
        
        # Heuristics for technical terms:
        # 1. Capitalized words (excluding start of sentence)
        # 2. Words with Greek letters or special characters
        # 3. Long, uncommon words (>12 chars)
        # 4. Mathematical notation
        
        technical_count = 0
        
        for i, word in enumerate(words):
            word_clean = re.sub(r'[^\w]', '', word)
            
            # Skip first word and short words
            if i == 0 or len(word_clean) < 4:
                continue
            
            # Check if capitalized (proper noun / technical term)
            if word_clean and word_clean[0].isupper():
                technical_count += 1
            
            # Check if very long word
            elif len(word_clean) > 12:
                technical_count += 0.5
        
        # Check for mathematical notation
        if re.search(r'[∫∑∏√±×÷≤≥≠∞∂∇]|[α-ωΑ-Ω]|\^|\d+/\d+', question):
            technical_count += 2
        
        # Normalize by question length
        if len(words) > 0:
            density = technical_count / len(words)
            return min(1.0, density * 3)  # Scale up (density is usually < 0.3)
        
        return 0.0
    
    def _syntactic_complexity(self, question: str) -> float:
        """
        Estimate syntactic complexity (subordinate clauses, nesting).
        """
        # Count subordinate clause markers
        subordinate_markers = [
            'qui', 'que', 'dont', 'où', 'lequel', 'laquelle',
            'parce que', 'puisque', 'bien que', 'quoique',
            'si', 'lorsque', 'quand', 'tandis que', 'alors que'
        ]
        
        question_lower = question.lower()
        complexity = 0
        
        for marker in subordinate_markers:
            complexity += question_lower.count(marker)
        
        # Count parentheses and commas (indicators of complex structure)
        complexity += question.count('(') * 0.5
        complexity += question.count(',') * 0.3
        
        # Normalize
        return min(1.0, complexity / 3)
    
    def estimate_batch(self, questions: List[str], question_types: List[str] = None) -> List[Tuple[str, float]]:
        """
        Estimate difficulty for multiple questions.
        
        Returns:
            List of (difficulty_label, confidence) tuples
        """
        if question_types is None:
            question_types = [None] * len(questions)
        
        results = []
        for q, qtype in zip(questions, question_types):
            difficulty, confidence, _ = self.estimate(q, qtype)
            results.append((difficulty, confidence))
        
        return results
    
    def get_distribution(self, questions: List[str], question_types: List[str] = None) -> Dict[str, int]:
        """
        Get difficulty distribution for a list of questions.
        
        Returns:
            Dict mapping difficulty -> count
        """
        results = self.estimate_batch(questions, question_types)
        counts = Counter(diff for diff, _ in results)
        return dict(counts)
    
    def format_distribution_report(self, questions: List[str], question_types: List[str] = None) -> str:
        """
        Create a human-readable distribution report.
        """
        distribution = self.get_distribution(questions, question_types)
        total = len(questions)
        
        report = ["Question Difficulty Distribution:", "=" * 50]
        
        # Target distribution: 30% easy, 50% medium, 20% hard
        targets = {"easy": 0.30, "medium": 0.50, "hard": 0.20}
        
        for difficulty in ["easy", "medium", "hard"]:
            count = distribution.get(difficulty, 0)
            proportion = count / total if total > 0 else 0.0
            target = targets[difficulty]
            
            bar = "█" * int(proportion * 50)
            status = "✓" if abs(proportion - target) < 0.10 else "⚠"
            
            report.append(f"{difficulty:8s}: {count:3d} ({proportion:6.1%}) {bar} {status} (target: {target:.0%})")
        
        report.append(f"\nTotal: {total} questions")
        
        # Recommendations
        report.append("\n💡 Recommendations:")
        for difficulty in ["easy", "medium", "hard"]:
            count = distribution.get(difficulty, 0)
            proportion = count / total if total > 0 else 0.0
            target = targets[difficulty]
            
            if proportion < target - 0.10:
                report.append(f"   • Need more '{difficulty}' questions (current: {proportion:.0%}, target: {target:.0%})")
            elif proportion > target + 0.10:
                report.append(f"   • Too many '{difficulty}' questions (current: {proportion:.0%}, target: {target:.0%})")
        
        return "\n".join(report)
    
    def explain_difficulty(self, question: str, question_type: str = None) -> str:
        """
        Provide detailed explanation of why a question has its difficulty rating.
        """
        difficulty, confidence, factors = self.estimate(question, question_type)
        
        report = [
            f"Question: {question}",
            f"",
            f"Difficulty: {difficulty.upper()} (confidence: {confidence:.0%})",
            f"",
            f"Factor Breakdown:"
        ]
        
        # Sort factors by contribution
        sorted_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)
        
        for factor_name, score in sorted_factors:
            bar = "█" * int(score * 20)
            impact = "HIGH" if score > 0.7 else "MED" if score > 0.4 else "LOW"
            report.append(f"  • {factor_name:12s}: {score:4.2f} {bar:20s} ({impact})")
        
        # Explanation
        report.append(f"\nExplanation:")
        
        if factors["type"] > 0.7:
            report.append(f"  - Question type '{question_type}' requires higher-order thinking")
        if factors["cognitive"] > 0.7:
            report.append(f"  - Uses complex cognitive verbs (analyze, evaluate, create)")
        if factors["multipart"] > 0.5:
            report.append(f"  - Multi-part question (multiple sub-questions)")
        if factors["technical"] > 0.5:
            report.append(f"  - High density of technical terminology")
        if factors["length"] > 0.7:
            report.append(f"  - Long, complex phrasing")
        if factors["syntax"] > 0.5:
            report.append(f"  - Complex sentence structure with subordinate clauses")
        
        return "\n".join(report)


# Testing
if __name__ == "__main__":
    print("=" * 70)
    print("DIFFICULTY ESTIMATOR - TEST")
    print("=" * 70)
    print()
    
    # Import classifier
    import sys
    sys.path.insert(0, '.')
    from question_type_classifier import QuestionTypeClassifier
    
    classifier = QuestionTypeClassifier(language="french")
    estimator = DifficultyEstimator(classifier=classifier)
    
    # Test cases
    test_questions = [
        # Easy questions
        "Qu'est-ce qu'une tribu?",
        "Qui a découvert la radioactivité?",
        "Combien de chromosomes?",
        
        # Medium questions
        "Quelle est la différence entre une tribu et une σ-algèbre?",
        "Expliquer le principe d'incertitude de Heisenberg.",
        "Comment appliquer le théorème de Bayes?",
        
        # Hard questions
        "Analyser les limites épistémologiques du déterminisme laplacien et discuter leurs implications pour la mécanique quantique.",
        "Évaluer de manière critique les hypothèses sous-jacentes au modèle standard et proposer des extensions théoriques cohérentes.",
        "Comparer les approches axiomatique de Kolmogorov et fréquentiste de von Mises, puis justifier laquelle est plus adaptée aux systèmes chaotiques."
    ]
    
    print("1️⃣ Individual Difficulty Estimates:")
    print("-" * 70)
    
    for i, question in enumerate(test_questions, 1):
        qtype = classifier.classify(question)
        difficulty, confidence, factors = estimator.estimate(question, qtype)
        
        # Color coding
        emoji = "🟢" if difficulty == "easy" else "🟡" if difficulty == "medium" else "🔴"
        
        print(f"{i:2d}. {emoji} [{difficulty:6s}] {confidence:.0%} - {question[:50]}...")
        print(f"     Type: {qtype:12s} | Top factors: {', '.join(f'{k}={v:.2f}' for k, v in sorted(factors.items(), key=lambda x: x[1], reverse=True)[:3])}")
        print()
    
    print("2️⃣ Distribution Report:")
    print("-" * 70)
    print(estimator.format_distribution_report(test_questions))
    print()
    
    print("3️⃣ Detailed Explanation (Sample):")
    print("-" * 70)
    sample_q = test_questions[-1]  # Hardest question
    print(estimator.explain_difficulty(sample_q, classifier.classify(sample_q)))
    print()
    
    print("=" * 70)
    print("✅ DIFFICULTY ESTIMATOR READY TO USE")
    print("=" * 70)
