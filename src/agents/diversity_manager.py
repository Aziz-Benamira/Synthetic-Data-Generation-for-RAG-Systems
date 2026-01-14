"""
DiversityManager Agent - Semantic Duplicate Detection & Distribution Balancing
===============================================================================

Purpose:
--------
Ensures the generated dataset has high diversity by:
1. Detecting near-duplicate questions using semantic similarity
2. Tracking type/difficulty distribution and recommending underrepresented types
3. Maintaining a history of generated questions for comparison

Key Features:
-------------
- Semantic similarity using sentence-transformers (MiniLM)
- Configurable similarity thresholds (default: 0.85 for duplicates)
- Lightweight model (120MB) that works offline
- Distribution tracking across types and difficulties
- Suggestions for next question to maintain balance

Usage Example:
--------------
>>> from diversity_manager import DiversityManager
>>> 
>>> manager = DiversityManager(similarity_threshold=0.85)
>>> 
>>> # Add questions as they're generated
>>> is_unique = manager.add_question(
...     question="Qu'est-ce qu'une tribu?",
...     question_type="definition",
...     difficulty="easy"
... )
>>> 
>>> # Check if new question is too similar
>>> is_duplicate, similarity, similar_question = manager.check_similarity(
...     "Quelle est la définition d'une tribu?"
... )
>>> 
>>> # Get recommendations for balanced dataset
>>> recommendations = manager.get_next_question_suggestions()
>>> print(recommendations['recommended_type'])  # e.g., "analysis"
>>> print(recommendations['recommended_difficulty'])  # e.g., "hard"

Author: Seif
Date: 2025
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import warnings

# Suppress sentence-transformers warnings
warnings.filterwarnings('ignore')

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  Warning: sentence-transformers not installed. Using fallback mode.")
    print("   Install with: pip install sentence-transformers")


class DiversityManager:
    """
    Manages question diversity using semantic similarity and distribution tracking.
    
    Attributes:
        similarity_threshold: Float threshold for duplicate detection (0.0-1.0)
        model: SentenceTransformer model for embeddings
        history: List of {question, embedding, type, difficulty}
        type_counts: Counter for question types
        difficulty_counts: Counter for difficulties
    """
    
    # Target distributions (from IMPROVEMENT_PROPOSALS.md)
    TARGET_TYPE_DISTRIBUTION = {
        'factoid': 0.14,      # 14% - Simple facts
        'definition': 0.14,   # 14% - What is X?
        'comparison': 0.14,   # 14% - A vs B
        'explanation': 0.14,  # 14% - How/Why
        'application': 0.14,  # 14% - How to use
        'calculation': 0.14,  # 14% - Quantitative
        'analysis': 0.16,     # 16% - Critical thinking
    }
    
    TARGET_DIFFICULTY_DISTRIBUTION = {
        'easy': 0.30,    # 30% - Basic recall
        'medium': 0.50,  # 50% - Understanding
        'hard': 0.20,    # 20% - Complex reasoning
    }
    
    def __init__(self, similarity_threshold: float = 0.85, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the DiversityManager.
        
        Args:
            similarity_threshold: Cosine similarity threshold for duplicates (default: 0.85)
            model_name: SentenceTransformer model to use (default: all-MiniLM-L6-v2, 120MB)
        """
        self.similarity_threshold = similarity_threshold
        self.history = []
        self.type_counts = Counter()
        self.difficulty_counts = Counter()
        
        # Load sentence transformer model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                print(f"📥 Loading embedding model: {model_name}...")
                self.model = SentenceTransformer(model_name)
                print(f"✅ Model loaded successfully!")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                print("   Falling back to simple string matching.")
                self.model = None
        else:
            self.model = None
    
    def _compute_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Compute embedding for a text string.
        
        Args:
            text: Input text
            
        Returns:
            Numpy array embedding or None if model unavailable
        """
        if self.model is None:
            return None
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            print(f"⚠️  Error computing embedding: {e}")
            return None
    
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            
        Returns:
            Cosine similarity score (0.0-1.0)
        """
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _simple_similarity(self, text1: str, text2: str) -> float:
        """
        Fallback similarity using Jaccard index on words (no ML model needed).
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Jaccard similarity (0.0-1.0)
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def check_similarity(self, question: str) -> Tuple[bool, float, Optional[str]]:
        """
        Check if a question is too similar to existing questions.
        
        Args:
            question: New question to check
            
        Returns:
            Tuple of (is_duplicate, max_similarity, most_similar_question)
        """
        if len(self.history) == 0:
            return False, 0.0, None
        
        # Compute embedding for new question
        new_embedding = self._compute_embedding(question)
        
        max_similarity = 0.0
        most_similar_question = None
        
        for entry in self.history:
            if self.model is not None and new_embedding is not None:
                # Use semantic similarity
                similarity = self._cosine_similarity(new_embedding, entry['embedding'])
            else:
                # Fallback to simple similarity
                similarity = self._simple_similarity(question, entry['question'])
            
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_question = entry['question']
        
        is_duplicate = max_similarity >= self.similarity_threshold
        
        return is_duplicate, max_similarity, most_similar_question
    
    def add_question(
        self, 
        question: str, 
        question_type: str, 
        difficulty: str,
        force: bool = False
    ) -> bool:
        """
        Add a question to the history (with duplicate checking).
        
        Args:
            question: Question text
            question_type: Type classification
            difficulty: Difficulty level
            force: Skip duplicate check if True
            
        Returns:
            True if added, False if duplicate rejected
        """
        # Check for duplicates
        if not force:
            is_duplicate, similarity, similar_q = self.check_similarity(question)
            if is_duplicate:
                print(f"🚫 Duplicate detected (similarity: {similarity:.2%})")
                print(f"   New: {question[:60]}...")
                print(f"   Existing: {similar_q[:60]}...")
                return False
        
        # Compute embedding
        embedding = self._compute_embedding(question)
        
        # Add to history
        self.history.append({
            'question': question,
            'embedding': embedding,
            'type': question_type,
            'difficulty': difficulty
        })
        
        # Update counters
        self.type_counts[question_type] += 1
        self.difficulty_counts[difficulty] += 1
        
        return True
    
    def get_type_distribution(self) -> Dict[str, float]:
        """Get current type distribution as percentages."""
        total = sum(self.type_counts.values())
        if total == 0:
            return {}
        
        return {
            qtype: count / total
            for qtype, count in self.type_counts.items()
        }
    
    def get_difficulty_distribution(self) -> Dict[str, float]:
        """Get current difficulty distribution as percentages."""
        total = sum(self.difficulty_counts.values())
        if total == 0:
            return {}
        
        return {
            difficulty: count / total
            for difficulty, count in self.difficulty_counts.items()
        }
    
    def get_next_question_suggestions(self) -> Dict:
        """
        Suggest type/difficulty for next question to balance distribution.
        
        Returns:
            Dict with recommended_type, recommended_difficulty, and explanations
        """
        current_types = self.get_type_distribution()
        current_difficulties = self.get_difficulty_distribution()
        
        # Find most underrepresented type
        type_gaps = {}
        for qtype, target_pct in self.TARGET_TYPE_DISTRIBUTION.items():
            current_pct = current_types.get(qtype, 0.0)
            gap = target_pct - current_pct
            type_gaps[qtype] = gap
        
        recommended_type = max(type_gaps, key=type_gaps.get)
        
        # Find most underrepresented difficulty
        difficulty_gaps = {}
        for difficulty, target_pct in self.TARGET_DIFFICULTY_DISTRIBUTION.items():
            current_pct = current_difficulties.get(difficulty, 0.0)
            gap = target_pct - current_pct
            difficulty_gaps[difficulty] = gap
        
        recommended_difficulty = max(difficulty_gaps, key=difficulty_gaps.get)
        
        return {
            'recommended_type': recommended_type,
            'recommended_difficulty': recommended_difficulty,
            'type_gap': type_gaps[recommended_type],
            'difficulty_gap': difficulty_gaps[recommended_difficulty],
            'current_type_distribution': current_types,
            'current_difficulty_distribution': current_difficulties,
            'total_questions': len(self.history)
        }
    
    def format_distribution_report(self) -> str:
        """Generate a human-readable distribution report."""
        type_dist = self.get_type_distribution()
        difficulty_dist = self.get_difficulty_distribution()
        
        report = []
        report.append("=" * 70)
        report.append("DIVERSITY REPORT")
        report.append("=" * 70)
        report.append(f"Total Questions: {len(self.history)}")
        report.append("")
        
        # Type distribution
        report.append("Question Type Distribution:")
        report.append("-" * 70)
        for qtype in sorted(self.TARGET_TYPE_DISTRIBUTION.keys()):
            current = type_dist.get(qtype, 0.0)
            target = self.TARGET_TYPE_DISTRIBUTION[qtype]
            count = self.type_counts.get(qtype, 0)
            bar = "█" * int(current * 100 / 2)
            status = "✅" if abs(current - target) < 0.05 else "⚠️"
            report.append(f"{qtype:12} : {count:3} ({current:5.1%}) {bar:20} {status} (target: {target:.0%})")
        report.append("")
        
        # Difficulty distribution
        report.append("Difficulty Distribution:")
        report.append("-" * 70)
        for difficulty in ['easy', 'medium', 'hard']:
            current = difficulty_dist.get(difficulty, 0.0)
            target = self.TARGET_DIFFICULTY_DISTRIBUTION[difficulty]
            count = self.difficulty_counts.get(difficulty, 0)
            bar = "█" * int(current * 100 / 2)
            status = "✅" if abs(current - target) < 0.05 else "⚠️"
            report.append(f"{difficulty:12} : {count:3} ({current:5.1%}) {bar:20} {status} (target: {target:.0%})")
        report.append("")
        
        # Recommendations
        suggestions = self.get_next_question_suggestions()
        report.append("💡 Next Question Recommendations:")
        report.append(f"   • Type: {suggestions['recommended_type']} (gap: {suggestions['type_gap']:+.1%})")
        report.append(f"   • Difficulty: {suggestions['recommended_difficulty']} (gap: {suggestions['difficulty_gap']:+.1%})")
        report.append("")
        
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def clear_history(self):
        """Clear all history and reset counters."""
        self.history = []
        self.type_counts = Counter()
        self.difficulty_counts = Counter()
    
    def export_history(self) -> List[Dict]:
        """Export history without embeddings (for JSON serialization)."""
        return [
            {
                'question': entry['question'],
                'type': entry['type'],
                'difficulty': entry['difficulty']
            }
            for entry in self.history
        ]
    
    def import_history(self, questions: List[Dict]):
        """
        Import questions from previous runs.
        
        Args:
            questions: List of {question, type, difficulty} dicts
        """
        for q in questions:
            self.add_question(
                question=q['question'],
                question_type=q['type'],
                difficulty=q['difficulty'],
                force=False  # Still check for duplicates
            )


# ============================================================================
# TEST SUITE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DIVERSITY MANAGER - TEST")
    print("=" * 70)
    print()
    
    # Initialize manager
    manager = DiversityManager(similarity_threshold=0.85)
    
    # Test 1: Add various questions
    print("1️⃣ Adding Questions to History:")
    print("-" * 70)
    
    test_questions = [
        ("Qu'est-ce qu'une tribu?", "definition", "easy"),
        ("Qui a découvert la radioactivité?", "factoid", "easy"),
        ("Quelle est la différence entre une tribu et une σ-algèbre?", "comparison", "medium"),
        ("Expliquer le principe d'incertitude de Heisenberg.", "explanation", "easy"),
        ("Comment appliquer le théorème de Bayes?", "application", "medium"),
        ("Analyser les limites épistémologiques du déterminisme.", "analysis", "hard"),
        ("Calculer la probabilité P(A ∪ B) sachant P(A) et P(B).", "calculation", "medium"),
        ("Évaluer de manière critique les hypothèses sous-jacentes.", "analysis", "hard"),
    ]
    
    for i, (question, qtype, difficulty) in enumerate(test_questions, 1):
        success = manager.add_question(question, qtype, difficulty)
        status = "✅ Added" if success else "❌ Rejected"
        print(f" {i}. {status}: {question[:50]}... [{qtype}, {difficulty}]")
    
    print()
    
    # Test 2: Check for duplicates
    print("2️⃣ Duplicate Detection:")
    print("-" * 70)
    
    duplicate_tests = [
        "Qu'est-ce qu'une tribu?",  # Exact duplicate
        "Quelle est la définition d'une tribu?",  # Semantic duplicate
        "Qu'est-ce qu'un espace vectoriel?",  # New question
    ]
    
    for i, test_q in enumerate(duplicate_tests, 1):
        is_dup, sim, similar_q = manager.check_similarity(test_q)
        status = "🚫 DUPLICATE" if is_dup else "✅ UNIQUE"
        print(f" {i}. {status} (similarity: {sim:.2%})")
        print(f"    Test: {test_q}")
        if similar_q:
            print(f"    Most similar: {similar_q[:50]}...")
        print()
    
    # Test 3: Distribution report
    print("3️⃣ Distribution Report:")
    print("-" * 70)
    print(manager.format_distribution_report())
    
    # Test 4: Suggestions
    print()
    print("4️⃣ Next Question Suggestions:")
    print("-" * 70)
    suggestions = manager.get_next_question_suggestions()
    print(f"Recommended Type: {suggestions['recommended_type']}")
    print(f"Recommended Difficulty: {suggestions['recommended_difficulty']}")
    print(f"Type Gap: {suggestions['type_gap']:+.1%}")
    print(f"Difficulty Gap: {suggestions['difficulty_gap']:+.1%}")
    print()
    
    print("=" * 70)
    print("✅ DIVERSITY MANAGER READY TO USE")
    print("=" * 70)
