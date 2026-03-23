"""
Question Type Classifier
========================

Rule-based classifier for categorizing questions into 7 types.
Fast, deterministic, no LLM calls needed.

Question Types:
1. factoid: Simple fact-based questions (qui, quoi, où, quand)
2. definition: Asking for definitions or meanings
3. comparison: Comparing concepts or entities
4. explanation: Why/how questions requiring reasoning
5. application: Using concepts in practice
6. calculation: Mathematical/quantitative problems
7. analysis: Critical thinking, evaluation, interpretation
"""

from typing import Dict, List, Tuple
from collections import defaultdict
import re


class QuestionTypeClassifier:
    """
    Lightweight rule-based classifier for question types.
    Uses keyword matching and pattern detection.
    """
    
    # Keyword patterns for each question type
    PATTERNS = {
        "factoid": {
            "french": [
                r"\bqui\b", r"\bquoi\b", r"\boù\b", r"\bquand\b",
                r"\bcombien\b", r"\bquel(le)?s?\b(?!\s+est)",
                r"\bquelle est la valeur\b", r"\bnommer\b", r"\bciter\b"
            ],
            "english": [
                r"\bwho\b", r"\bwhat\b", r"\bwhere\b", r"\bwhen\b",
                r"\bhow many\b", r"\bhow much\b", r"\bwhich\b",
                r"\bname\b", r"\blist\b", r"\bidentify\b"
            ]
        },
        "definition": {
            "french": [
                r"qu'est-ce que", r"qu'est-ce qu'", r"\bdéfinir\b", r"\bdéfinition\b",
                r"\bsignifie\b", r"\bdésigne\b", r"\bc'est quoi\b",
                r"quelle est la définition", r"\bsens de\b", r"\bveut dire\b"
            ],
            "english": [
                r"what is", r"what are", r"\bdefine\b", r"\bdefinition\b",
                r"\bmeans?\b", r"\bmeaning\b", r"what does .+ mean"
            ]
        },
        "comparison": {
            "french": [
                r"\bdifférence\b", r"\bcomparer\b", r"\bcomparaison\b",
                r"\bdistinction\b", r"\bversus\b", r"\bvs\b", r"\bcontraire\b",
                r"\bcontrairement\b", r"\bopposé\b", r"\bsimilitude\b",
                r"\bentre .+ et\b", r"\bplutôt que\b"
            ],
            "english": [
                r"\bdifference\b", r"\bcompare\b", r"\bcomparison\b",
                r"\bdistinguish\b", r"\bversus\b", r"\bvs\b", r"\bcontrast\b",
                r"\brather than\b", r"\binstead of\b", r"\bbetween .+ and\b"
            ]
        },
        "explanation": {
            "french": [
                r"\bpourquoi\b", r"\bcomment\b(?!\scalculer)", r"\bexpliquer\b",
                r"\braison\b", r"\bcause\b", r"\bconséquence\b",
                r"\bentraîne\b", r"\bproduit\b", r"\bprovoque\b",
                r"\ben raison de\b", r"\bcar\b", r"\bparce que\b"
            ],
            "english": [
                r"\bwhy\b", r"\bhow(?!\smany|\smuch)\b", r"\bexplain\b",
                r"\breason\b", r"\bcause\b", r"\bconsequence\b",
                r"\bresult in\b", r"\blead to\b", r"\bdue to\b"
            ]
        },
        "application": {
            "french": [
                r"\bappliquer\b", r"\butiliser\b", r"\bexemple\b",
                r"\bcas\b", r"\bpratique\b", r"\bdémontrer\b",
                r"\billustrer\b", r"\ben pratique\b", r"\bdans le cas\b",
                r"\bcomment utiliser\b", r"\bmettre en œuvre\b"
            ],
            "english": [
                r"\bapply\b", r"\buse\b", r"\bexample\b", r"\bcase\b",
                r"\bpractice\b", r"\bdemonstrate\b", r"\billustrate\b",
                r"\bin practice\b", r"\bhow to use\b", r"\bimplement\b"
            ]
        },
        "calculation": {
            "french": [
                r"\bcalculer\b", r"\bdéterminer la valeur\b", r"\btrouver\b",
                r"\brésoudre\b", r"\béquation\b", r"\bvaleur de\b",
                r"\bresultat\b", r"\bchiffrer\b", r"\bmesurer\b",
                r"\bestimer\b", r"\bquelle est la valeur\b"
            ],
            "english": [
                r"\bcalculate\b", r"\bcompute\b", r"\bfind the value\b",
                r"\bsolve\b", r"\bequation\b", r"\bvalue of\b",
                r"\bmeasure\b", r"\bestimate\b", r"\bwhat is the value\b"
            ]
        },
        "analysis": {
            "french": [
                r"\banalyser\b", r"\bévaluer\b", r"\bcritique\b",
                r"\bjuger\b", r"\binterpréter\b", r"\bexaminer\b",
                r"\bdiscuter\b", r"\bargumenter\b", r"\bjustifier\b",
                r"\bapprécier\b", r"\bpertinence\b", r"\blimite\b"
            ],
            "english": [
                r"\banalyze\b", r"\bevaluate\b", r"\bcritique\b",
                r"\bjudge\b", r"\binterpret\b", r"\bexamine\b",
                r"\bdiscuss\b", r"\bargue\b", r"\bjustify\b",
                r"\bassess\b", r"\blimitation\b"
            ]
        }
    }
    
    # Type priorities (if multiple matches, use this order)
    TYPE_PRIORITY = [
        "calculation",    # Most specific
        "definition",
        "comparison",
        "analysis",
        "explanation",
        "application",
        "factoid"        # Most general (fallback)
    ]
    
    def __init__(self, language: str = "french"):
        """
        Initialize classifier.
        
        Args:
            language: "french" or "english"
        """
        self.language = language
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency"""
        self.compiled_patterns = {}
        for qtype, lang_patterns in self.PATTERNS.items():
            if self.language in lang_patterns:
                patterns = [re.compile(p, re.IGNORECASE) for p in lang_patterns[self.language]]
                self.compiled_patterns[qtype] = patterns
    
    def classify(self, question: str) -> str:
        """
        Classify question into one of 7 types.
        
        Args:
            question: Question text
            
        Returns:
            Question type: "factoid" | "definition" | "comparison" | "explanation" |
                          "application" | "calculation" | "analysis"
        """
        if not question or not question.strip():
            return "factoid"  # Default
        
        question_lower = question.lower()
        
        # Special case: "comment appliquer/utiliser" = application, not explanation
        if re.search(r"comment\s+(appliquer|utiliser|mettre en œuvre)", question_lower):
            return "application"
        
        # Check each type for pattern matches
        matches = {}
        for qtype, patterns in self.compiled_patterns.items():
            match_count = sum(1 for pattern in patterns if pattern.search(question))
            if match_count > 0:
                matches[qtype] = match_count
        
        # If no matches, return factoid as default
        if not matches:
            return "factoid"
        
        # If single match, return it
        if len(matches) == 1:
            return list(matches.keys())[0]
        
        # Multiple matches: use priority order
        for qtype in self.TYPE_PRIORITY:
            if qtype in matches:
                return qtype
        
        # Fallback
        return max(matches, key=matches.get)
    
    def classify_batch(self, questions: List[str]) -> List[str]:
        """Classify multiple questions"""
        return [self.classify(q) for q in questions]
    
    def get_distribution(self, questions: List[str]) -> Dict[str, float]:
        """
        Get type distribution for a list of questions.
        
        Returns:
            Dict mapping type -> proportion (0-1)
        """
        if not questions:
            return {}
        
        types = self.classify_batch(questions)
        counts = defaultdict(int)
        for qtype in types:
            counts[qtype] += 1
        
        total = len(questions)
        return {qtype: count / total for qtype, count in counts.items()}
    
    def suggest_underrepresented_type(self, questions: List[str]) -> Tuple[str, float]:
        """
        Find the most underrepresented question type.
        
        Returns:
            (type_name, current_proportion)
        """
        distribution = self.get_distribution(questions)
        
        # Target: roughly equal distribution (1/7 ≈ 0.143)
        target_proportion = 1.0 / 7
        
        # Find type with lowest proportion
        all_types = ["factoid", "definition", "comparison", "explanation", 
                     "application", "calculation", "analysis"]
        
        min_type = all_types[0]
        min_proportion = distribution.get(min_type, 0.0)
        
        for qtype in all_types[1:]:
            proportion = distribution.get(qtype, 0.0)
            if proportion < min_proportion:
                min_proportion = proportion
                min_type = qtype
        
        return min_type, min_proportion
    
    def get_type_examples(self, qtype: str) -> List[str]:
        """
        Get example questions for a given type.
        Useful for prompting the generator.
        """
        examples = {
            "factoid": [
                "Qui a découvert la radioactivité?",
                "Quelle est la capitale de la France?",
                "Combien de chromosomes possède un être humain?",
                "Où se situe le Mont Blanc?"
            ],
            "definition": [
                "Qu'est-ce qu'une fonction continue?",
                "Définir le concept d'entropie.",
                "Que signifie le terme 'isotope'?",
                "Quelle est la définition d'une matrice inversible?"
            ],
            "comparison": [
                "Quelle est la différence entre mitose et méiose?",
                "Comparer les approches bayésienne et fréquentiste.",
                "En quoi l'ARN diffère-t-il de l'ADN?",
                "Distinguer corrélation et causalité."
            ],
            "explanation": [
                "Pourquoi le ciel est-il bleu?",
                "Comment fonctionne la photosynthèse?",
                "Expliquer le principe de la relativité restreinte.",
                "Pour quelle raison les métaux conduisent-ils l'électricité?"
            ],
            "application": [
                "Comment appliquer le théorème de Pythagore dans ce cas?",
                "Donner un exemple d'utilisation de la dérivée en physique.",
                "Illustrer le concept de sélection naturelle avec un cas concret.",
                "Dans quelles situations utilise-t-on un test de Student?"
            ],
            "calculation": [
                "Calculer la dérivée de f(x) = x² + 3x.",
                "Quelle est la valeur de l'intégrale ∫₀¹ x dx?",
                "Déterminer la probabilité d'obtenir deux fois 6 en lançant deux dés.",
                "Résoudre l'équation 2x + 5 = 13."
            ],
            "analysis": [
                "Analyser les limites du modèle newtonien.",
                "Évaluer la pertinence de cette approximation.",
                "Discuter des implications philosophiques du déterminisme.",
                "Interpréter les résultats de cette expérience."
            ]
        }
        
        if self.language == "english":
            examples = {
                "factoid": [
                    "Who discovered radioactivity?",
                    "What is the capital of France?",
                    "How many chromosomes does a human have?",
                    "Where is Mount Blanc located?"
                ],
                "definition": [
                    "What is a continuous function?",
                    "Define the concept of entropy.",
                    "What does the term 'isotope' mean?",
                    "What is the definition of an invertible matrix?"
                ],
                "comparison": [
                    "What is the difference between mitosis and meiosis?",
                    "Compare Bayesian and frequentist approaches.",
                    "How does RNA differ from DNA?",
                    "Distinguish between correlation and causation."
                ],
                "explanation": [
                    "Why is the sky blue?",
                    "How does photosynthesis work?",
                    "Explain the principle of special relativity.",
                    "Why do metals conduct electricity?"
                ],
                "application": [
                    "How to apply the Pythagorean theorem in this case?",
                    "Give an example of derivative use in physics.",
                    "Illustrate natural selection with a concrete case.",
                    "In which situations would you use a Student's t-test?"
                ],
                "calculation": [
                    "Calculate the derivative of f(x) = x² + 3x.",
                    "What is the value of the integral ∫₀¹ x dx?",
                    "Determine the probability of rolling two sixes with two dice.",
                    "Solve the equation 2x + 5 = 13."
                ],
                "analysis": [
                    "Analyze the limitations of the Newtonian model.",
                    "Evaluate the relevance of this approximation.",
                    "Discuss the philosophical implications of determinism.",
                    "Interpret the results of this experiment."
                ]
            }
        
        return examples.get(qtype, [])
    
    def format_distribution_report(self, questions: List[str]) -> str:
        """
        Create a human-readable distribution report.
        """
        distribution = self.get_distribution(questions)
        
        report = ["Question Type Distribution:", "=" * 50]
        
        for qtype in self.TYPE_PRIORITY:
            count = sum(1 for q in questions if self.classify(q) == qtype)
            proportion = distribution.get(qtype, 0.0)
            bar = "█" * int(proportion * 50)
            report.append(f"{qtype:15s}: {count:3d} ({proportion:6.1%}) {bar}")
        
        total = len(questions)
        report.append(f"\nTotal: {total} questions")
        
        # Add recommendation
        underrep_type, underrep_prop = self.suggest_underrepresented_type(questions)
        target = 1.0 / 7
        report.append(f"\n💡 Recommendation: Generate more '{underrep_type}' questions")
        report.append(f"   Current: {underrep_prop:.1%}, Target: {target:.1%}")
        
        return "\n".join(report)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("QUESTION TYPE CLASSIFIER - TEST")
    print("=" * 70)
    print()
    
    # Initialize classifier
    classifier = QuestionTypeClassifier(language="french")
    
    # Test cases
    test_questions = [
        "Qu'est-ce qu'une tribu en théorie des probabilités?",  # definition
        "Quelle est la différence entre une tribu et une σ-algèbre?",  # comparison
        "Pourquoi une tribu doit-elle contenir l'ensemble vide?",  # explanation
        "Combien d'éléments contient une tribu finie?",  # factoid
        "Calculer la probabilité de l'événement A∪B.",  # calculation
        "Comment appliquer le théorème de Bayes dans ce contexte?",  # application
        "Analyser les propriétés de stabilité d'une tribu.",  # analysis
        "Qui a introduit le concept de σ-algèbre?",  # factoid
        "Définir la notion de mesure de probabilité.",  # definition
        "Comparer les approches fréquentiste et bayésienne.",  # comparison
    ]
    
    print("1️⃣ Individual Classifications:")
    print("-" * 70)
    for i, question in enumerate(test_questions, 1):
        qtype = classifier.classify(question)
        print(f"{i:2d}. [{qtype:12s}] {question[:50]}...")
    print()
    
    print("2️⃣ Distribution Report:")
    print("-" * 70)
    print(classifier.format_distribution_report(test_questions))
    print()
    
    print("3️⃣ Type Examples:")
    print("-" * 70)
    underrep_type, _ = classifier.suggest_underrepresented_type(test_questions)
    examples = classifier.get_type_examples(underrep_type)
    print(f"Examples for '{underrep_type}' type:")
    for ex in examples[:3]:
        print(f"  • {ex}")
    print()
    
    print("=" * 70)
    print("✅ CLASSIFIER READY TO USE")
    print("=" * 70)
