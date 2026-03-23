"""
Dataset Generation Pipeline - Orchestrator
===========================================

Orchestrates the full pipeline: PDF → Chunks → Questions → Answers → Critic → Dataset

Pipeline Flow:
1. Parse PDF into semantic chunks
2. For each chunk:
   a. Generate candidate questions
   b. Generate answers for each question
   c. Evaluate QA pairs with Critic
   d. Keep only PASSED pairs
3. Export to HuggingFace format

Design Principles:
- Simple linear flow (no LangGraph needed for sequential pipeline)
- Progress tracking and logging
- Resume capability (checkpoint after each chunk)
- Export in multiple formats
"""

import os
import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Callable
from datetime import datetime
from enum import Enum

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'chunking'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'agents'))

from semantic_chunker import SemanticChunker, SemanticChunk
from question_generator import QuestionGenerator, CandidateQuestion, QuestionType, DifficultyLevel
from answer_generator import AnswerGenerator, QAPair
from critic_agent import CriticAgent, CriticEvaluation, FinalDecision
from question_type_classifier import QuestionTypeClassifier
from difficulty_estimator import DifficultyEstimator
from diversity_manager import DiversityManager


class PipelineStatus(Enum):
    """Status of the pipeline"""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineConfig:
    """Configuration for the pipeline"""
    # Input
    pdf_path: str
    output_dir: str = "output"
    
    # Chunk selection
    max_chunks: Optional[int] = None  # None = all chunks
    chunk_types: Optional[List[str]] = None  # None = all types
    min_chunk_length: int = 200  # Minimum chars
    
    # Question generation
    questions_per_chunk: int = 2
    question_types: Optional[List[str]] = None  # None = all types
    difficulty_distribution: Dict[str, float] = field(default_factory=lambda: {
        "easy": 0.3, "medium": 0.5, "hard": 0.2
    })
    
    # LLM settings (Ollama Local)
    generator_model: str = "mistral:latest"  # Mistral 7B Instruct (~4.5GB)
    critic_model: str = "llama3:8b"  # Llama 3 8B - STRICT! (~4.7GB)
    temperature: float = 0.7
    
    # Retry settings (AGENTIC WORKFLOW)
    max_retries: int = 2  # Max retries when critic rejects (0 = no retry)
    
    # Diversity settings (NEW)
    enable_diversity_check: bool = True  # Check for duplicate questions
    diversity_threshold: float = 0.85  # Similarity threshold for duplicates
    
    # Language
    language: str = "fr"  # "fr" or "en"
    
    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_frequency: int = 10  # Save every N chunks
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pdf_path": self.pdf_path,
            "output_dir": self.output_dir,
            "max_chunks": self.max_chunks,
            "chunk_types": self.chunk_types,
            "min_chunk_length": self.min_chunk_length,
            "questions_per_chunk": self.questions_per_chunk,
            "question_types": self.question_types,
            "difficulty_distribution": self.difficulty_distribution,
            "generator_model": self.generator_model,
            "critic_model": self.critic_model,
            "temperature": self.temperature,
            "max_retries": self.max_retries,
            "language": self.language,
            "save_checkpoints": self.save_checkpoints,
            "checkpoint_frequency": self.checkpoint_frequency
        }


@dataclass
class PipelineStats:
    """Statistics for the pipeline run"""
    # Timing
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: float = 0
    
    # Counts
    total_chunks: int = 0
    processed_chunks: int = 0
    total_questions_generated: int = 0
    total_qa_pairs: int = 0
    passed_qa_pairs: int = 0
    rejected_qa_pairs: int = 0
    
    # Retry stats (AGENTIC WORKFLOW)
    total_retries: int = 0
    passed_after_retry: int = 0
    
    # Diversity stats (NEW)
    duplicates_detected: int = 0
    
    # Rates
    pass_rate: float = 0.0
    questions_per_chunk_avg: float = 0.0
    
    # Errors
    errors: List[str] = field(default_factory=list)
    
    # Critic breakdown
    rejection_reasons: Dict[str, int] = field(default_factory=dict)
    criterion_averages: Dict[str, float] = field(default_factory=dict)
    
    # Distribution stats (NEW)
    type_distribution: Dict[str, int] = field(default_factory=dict)
    difficulty_distribution: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass 
class DatasetEntry:
    """Single entry in the final dataset"""
    # Question-Answer
    question: str
    answer: str
    
    # Source info
    source_file: str
    chunk_id: str
    page_range: Tuple[int, int]
    
    # Metadata (ENHANCED)
    question_type: str  # Classified by QuestionTypeClassifier
    difficulty: str  # Estimated by DifficultyEstimator
    
    # Quality scores
    critic_score: float
    criterion_scores: Dict[str, float]
    
    # Supporting evidence
    supporting_quotes: List[str]
    chunk_content: str  # Full chunk for reference
    
    # Optional fields (with defaults)
    chapter: Optional[str] = None
    section: Optional[str] = None
    question_type_confidence: Optional[float] = None  # NEW
    difficulty_confidence: Optional[float] = None  # NEW
    difficulty_factors: Optional[Dict[str, float]] = None  # NEW: Factor breakdown
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "source_file": self.source_file,
            "chunk_id": self.chunk_id,
            "page_range": list(self.page_range),
            "chapter": self.chapter,
            "section": self.section,
            "question_type": self.question_type,
            "question_type_confidence": self.question_type_confidence,
            "difficulty": self.difficulty,
            "difficulty_confidence": self.difficulty_confidence,
            "difficulty_factors": self.difficulty_factors,
            "critic_score": self.critic_score,
            "criterion_scores": self.criterion_scores,
            "supporting_quotes": self.supporting_quotes,
            "chunk_content": self.chunk_content
        }
    
    def to_huggingface_format(self) -> Dict[str, Any]:
        """Convert to HuggingFace datasets format"""
        return {
            "question": self.question,
            "answer": self.answer,
            "context": self.chunk_content,
            "source": self.source_file,
            "metadata": {
                "chunk_id": self.chunk_id,
                "page_range": list(self.page_range),
                "chapter": self.chapter,
                "section": self.section,
                "question_type": self.question_type,
                "difficulty": self.difficulty,
                "quality_score": self.critic_score
            }
        }


class DatasetPipeline:
    """
    Main orchestrator for dataset generation.
    
    Usage:
        pipeline = DatasetPipeline(config, llm_client)
        dataset = pipeline.run()
        pipeline.export("output/dataset.json")
    """
    
    def __init__(
        self,
        config: PipelineConfig,
        llm_client: Any,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration
            llm_client: LLM API client (e.g., Groq client)
            progress_callback: Optional callback(stage, current, total)
        """
        self.config = config
        self.llm_client = llm_client
        self.progress_callback = progress_callback
        
        # State
        self.status = PipelineStatus.NOT_STARTED
        self.stats = PipelineStats()
        self.dataset: List[DatasetEntry] = []
        self.checkpoint_data: Dict[str, Any] = {}
        
        # Initialize components
        self._init_components()
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
    
    def _init_components(self):
        """Initialize all pipeline components with separate models."""
        # Question Generator (uses generator_model)
        self.question_generator = QuestionGenerator(
            llm_client=self.llm_client,
            model_name=self.config.generator_model,
            language=self.config.language,
            temperature=self.config.temperature
        )
        
        # Answer Generator (uses generator_model)
        self.answer_generator = AnswerGenerator(
            llm_client=self.llm_client,
            model_name=self.config.generator_model,
            language=self.config.language,
            temperature=self.config.temperature
        )
        
        # Critic Agent (uses DIFFERENT model to avoid self-evaluation bias!)
        self.critic = CriticAgent(
            llm_client=self.llm_client,
            model_name=self.config.critic_model,  # Different model!
            language=self.config.language,
            temperature=0.2,  # Lower for consistent evaluation
            strict_mode=True
        )
        
        # Enhancement Agents (NEW - No LLM needed!)
        self._log("🎯 Initializing enhancement agents...")
        self.type_classifier = QuestionTypeClassifier()
        self.difficulty_estimator = DifficultyEstimator(classifier=self.type_classifier)
        
        if self.config.enable_diversity_check:
            self.diversity_manager = DiversityManager(
                similarity_threshold=self.config.diversity_threshold
            )
        else:
            self.diversity_manager = None
        
        self._log("   ✓ QuestionTypeClassifier ready")
        self._log("   ✓ DifficultyEstimator ready")
        if self.diversity_manager:
            self._log("   ✓ DiversityManager ready")
    
    def _log(self, message: str):
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def _report_progress(self, stage: str, current: int, total: int):
        """Report progress to callback if available."""
        if self.progress_callback:
            self.progress_callback(stage, current, total)
    
    def run(self, resume_from_checkpoint: bool = False) -> List[DatasetEntry]:
        """
        Run the full pipeline.
        
        Args:
            resume_from_checkpoint: If True, try to resume from last checkpoint
            
        Returns:
            List of DatasetEntry objects (passed QA pairs)
        """
        self._log("=" * 60)
        self._log("DÉMARRAGE DU PIPELINE DE GÉNÉRATION DE DATASET")
        self._log("=" * 60)
        self._log(f"🔄 Mode AGENTIC: Retry loop activé (max {self.config.max_retries} retries)")
        self._log(f"   🤖 Generator: {self.config.generator_model}")
        self._log(f"   🔍 Critic: {self.config.critic_model} (different model!)")
        
        self.status = PipelineStatus.RUNNING
        self.stats.start_time = datetime.now().isoformat()
        
        try:
            # Step 1: Parse PDF into chunks
            chunks = self._step_parse_pdf()
            
            # Step 2: Filter chunks
            chunks = self._step_filter_chunks(chunks)
            
            # Step 3: Process each chunk
            self._step_process_chunks(chunks)
            
            # Step 4: Finalize stats
            self._finalize_stats()
            
            self.status = PipelineStatus.COMPLETED
            self._log(f"\n✅ PIPELINE TERMINÉ: {len(self.dataset)} QA pairs dans le dataset")
            
        except Exception as e:
            self.status = PipelineStatus.FAILED
            self.stats.errors.append(str(e))
            self._log(f"\n❌ ERREUR: {e}")
            raise
        
        return self.dataset
    
    def _step_parse_pdf(self) -> List[SemanticChunk]:
        """Step 1: Parse PDF into semantic chunks."""
        self._log(f"\n📄 ÉTAPE 1: Parsing du PDF...")
        self._log(f"   Fichier: {self.config.pdf_path}")
        
        chunker = SemanticChunker(self.config.pdf_path)
        chunks = chunker.chunk_document()
        
        self.stats.total_chunks = len(chunks)
        self._log(f"   ✓ {len(chunks)} chunks extraits")
        
        return chunks
    
    def _step_filter_chunks(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """Step 2: Filter chunks based on config."""
        self._log(f"\n🔍 ÉTAPE 2: Filtrage des chunks...")
        
        filtered = chunks
        
        # Filter by type
        if self.config.chunk_types:
            filtered = [c for c in filtered if c.semantic_type in self.config.chunk_types]
            self._log(f"   - Types {self.config.chunk_types}: {len(filtered)} chunks")
        
        # Filter by length
        filtered = [c for c in filtered if len(c.content) >= self.config.min_chunk_length]
        self._log(f"   - Longueur min {self.config.min_chunk_length}: {len(filtered)} chunks")
        
        # Limit number
        if self.config.max_chunks:
            filtered = filtered[:self.config.max_chunks]
            self._log(f"   - Limite {self.config.max_chunks}: {len(filtered)} chunks")
        
        self._log(f"   ✓ {len(filtered)} chunks sélectionnés pour traitement")
        
        return filtered
    
    def _step_process_chunks(self, chunks: List[SemanticChunk]):
        """Step 3: Process each chunk through the pipeline."""
        self._log(f"\n⚙️  ÉTAPE 3: Traitement des chunks...")
        
        total = len(chunks)
        
        for i, chunk in enumerate(chunks):
            self._log(f"\n{'─'*50}")
            self._log(f"Chunk {i+1}/{total}: {chunk.chunk_id}")
            self._log(f"Type: {chunk.semantic_type} | Pages: {chunk.page_range}")
            
            self._report_progress("chunks", i + 1, total)
            
            try:
                # 3a. Generate questions
                questions = self._generate_questions(chunk)
                
                if not questions:
                    self._log(f"   ⚠️  Aucune question générée, skip")
                    continue
                
                # 3b. Generate answers
                qa_pairs = self._generate_answers(questions, chunk)
                
                if not qa_pairs:
                    self._log(f"   ⚠️  Aucune réponse générée, skip")
                    continue
                
                # 3c. Evaluate with Critic
                passed_pairs = self._evaluate_qa_pairs(qa_pairs, chunk)
                
                # 3d. Add to dataset
                for qa_pair, evaluation in passed_pairs:
                    entry = self._create_dataset_entry(qa_pair, chunk, evaluation)
                    self.dataset.append(entry)
                
                self.stats.processed_chunks += 1
                
                # Checkpoint
                if self.config.save_checkpoints and (i + 1) % self.config.checkpoint_frequency == 0:
                    self._save_checkpoint(i + 1)
                    
            except Exception as e:
                self._log(f"   ❌ Erreur: {e}")
                self.stats.errors.append(f"Chunk {chunk.chunk_id}: {e}")
                continue
    
    def _generate_questions(self, chunk: SemanticChunk) -> List[CandidateQuestion]:
        """Generate questions for a chunk (with diversity checking)."""
        self._log(f"   📝 Génération de questions...")
        
        questions = self.question_generator.generate_from_chunk(
            chunk=chunk,
            num_questions=self.config.questions_per_chunk
        )
        
        # Filter duplicates with DiversityManager (if enabled)
        if self.diversity_manager:
            unique_questions = []
            for q in questions:
                is_dup, sim, similar_q = self.diversity_manager.check_similarity(q.question)
                if is_dup:
                    self.stats.duplicates_detected += 1
                    self._log(f"      🚫 Duplicate skipped ({sim:.0%} similar)")
                else:
                    unique_questions.append(q)
            questions = unique_questions
        
        self.stats.total_questions_generated += len(questions)
        self._log(f"      → {len(questions)} questions générées")
        
        return questions
    
    def _generate_answers(
        self, 
        questions: List[CandidateQuestion], 
        chunk: SemanticChunk
    ) -> List[QAPair]:
        """Generate answers for questions (with classification & estimation)."""
        self._log(f"   💬 Génération de réponses...")
        
        qa_pairs = []
        for q in questions:
            try:
                # Generate answer (returns GeneratedAnswer)
                answer = self.answer_generator.generate_answer(q, chunk)
                # Convert to QAPair
                qa_pair = QAPair.from_question_and_answer(q, answer)
                
                # Classify question type (NEW)
                qtype, confidence = self.type_classifier.classify(q.question)
                qa_pair._classified_type = qtype
                qa_pair._type_confidence = confidence
                
                # Estimate difficulty (NEW)
                difficulty, diff_conf, factors = self.difficulty_estimator.estimate(q.question, qtype)
                qa_pair._estimated_difficulty = difficulty
                qa_pair._difficulty_confidence = diff_conf
                qa_pair._difficulty_factors = factors
                
                qa_pairs.append(qa_pair)
                
                # Add to diversity history (if enabled)
                if self.diversity_manager:
                    self.diversity_manager.add_question(
                        question=q.question,
                        question_type=qtype,
                        difficulty=difficulty,
                        force=True  # Already checked in _generate_questions
                    )
                
            except Exception as e:
                self._log(f"      ⚠️ Erreur réponse: {e}")
                continue
        
        self.stats.total_qa_pairs += len(qa_pairs)
        self._log(f"      → {len(qa_pairs)} QA pairs générés")
        
        return qa_pairs
    
    def _evaluate_qa_pairs(
        self, 
        qa_pairs: List[QAPair], 
        chunk: SemanticChunk
    ) -> List[Tuple[QAPair, CriticEvaluation]]:
        """
        Evaluate QA pairs with Critic and RETRY if rejected (AGENTIC WORKFLOW).
        
        This implements the multi-agent feedback loop:
        1. Critic evaluates QA pair
        2. If REJECT: format feedback → regenerate Q+A (max N retries)
        3. If PASS or max retries exceeded: continue
        """
        self._log(f"   🔍 Évaluation par le Critic (max {self.config.max_retries} retries)...")
        
        passed = []
        
        for qa in qa_pairs:
            # Track retry attempts
            current_qa = qa
            current_question = None  # Will need CandidateQuestion for retry
            attempt = 0
            max_attempts = self.config.max_retries + 1  # Initial + retries
            
            while attempt < max_attempts:
                attempt += 1
                
                # Evaluate with Critic
                evaluation = self.critic.evaluate(current_qa, chunk)
                
                if evaluation.decision == FinalDecision.PASS:
                    # SUCCESS!
                    passed.append((current_qa, evaluation))
                    self.stats.passed_qa_pairs += 1
                    
                    # Track if success came after retry
                    if attempt > 1:
                        self.stats.passed_after_retry += 1
                        self._log(f"      ✅ PASS (après {attempt-1} retry): {current_qa.question[:40]}...")
                    else:
                        self._log(f"      ✅ PASS: {current_qa.question[:40]}...")
                    break
                    
                else:
                    # REJECTED - should we retry?
                    if attempt < max_attempts:
                        # INCREMENT RETRY COUNTER
                        self.stats.total_retries += 1
                        
                        # Format feedback and retry
                        self._log(f"      🔄 RETRY {attempt}/{self.config.max_retries}: {current_qa.question[:40]}...")
                        feedback = self.critic.format_feedback_for_retry(evaluation)
                        
                        # Regenerate BOTH question and answer
                        new_question = self.question_generator.regenerate_with_feedback(
                            chunk=chunk,
                            previous_question=current_qa.question,
                            critic_feedback=feedback
                        )
                        
                        if new_question:
                            # Generate new answer for new question
                            try:
                                new_answer = self.answer_generator.regenerate_with_feedback(
                                    question=new_question,
                                    chunk=chunk,
                                    previous_answer=current_qa.answer,
                                    critic_feedback=feedback
                                )
                                # Create new QAPair
                                current_qa = QAPair.from_question_and_answer(new_question, new_answer)
                            except Exception as e:
                                self._log(f"         ⚠️ Erreur régénération: {e}")
                                break  # Stop retry loop on error
                        else:
                            self._log(f"         ⚠️ Échec régénération question")
                            break  # Stop retry loop
                    else:
                        # Max retries exceeded - REJECT definitively
                        self.stats.rejected_qa_pairs += 1
                        for criterion in evaluation.failed_criteria:
                            self.stats.rejection_reasons[criterion] = \
                                self.stats.rejection_reasons.get(criterion, 0) + 1
                        self._log(f"      ❌ REJECT (après {self.config.max_retries} retries): {current_qa.question[:40]}...")
        
        self._log(f"      → {len(passed)}/{len(qa_pairs)} acceptés (avec retries)")
        
        return passed
    
    def _create_dataset_entry(
        self, 
        qa_pair: QAPair, 
        chunk: SemanticChunk,
        evaluation: CriticEvaluation
    ) -> DatasetEntry:
        """Create a dataset entry from QA pair (with enhanced metadata)."""
        # Get classified/estimated metadata (with fallback)
        qtype = getattr(qa_pair, '_classified_type', qa_pair.question_type)
        type_conf = getattr(qa_pair, '_type_confidence', None)
        difficulty = getattr(qa_pair, '_estimated_difficulty', qa_pair.difficulty)
        diff_conf = getattr(qa_pair, '_difficulty_confidence', None)
        diff_factors = getattr(qa_pair, '_difficulty_factors', None)
        
        # Update stats distributions
        self.stats.type_distribution[qtype] = self.stats.type_distribution.get(qtype, 0) + 1
        self.stats.difficulty_distribution[difficulty] = self.stats.difficulty_distribution.get(difficulty, 0) + 1
        
        return DatasetEntry(
            question=qa_pair.question,
            answer=qa_pair.answer,
            source_file=os.path.basename(self.config.pdf_path),
            chunk_id=chunk.chunk_id,
            page_range=chunk.page_range,
            chapter=chunk.chapter_title,
            section=chunk.section_title,
            question_type=qtype,
            question_type_confidence=type_conf,
            difficulty=difficulty,
            difficulty_confidence=diff_conf,
            difficulty_factors=diff_factors,
            critic_score=evaluation.overall_score,
            criterion_scores={
                name: eval.score 
                for name, eval in evaluation.criteria_evaluations.items()
            },
            supporting_quotes=qa_pair.supporting_quotes,
            chunk_content=chunk.content
        )
    
    def _finalize_stats(self):
        """Finalize statistics after pipeline completion."""
        self.stats.end_time = datetime.now().isoformat()
        
        if self.stats.start_time and self.stats.end_time:
            start = datetime.fromisoformat(self.stats.start_time)
            end = datetime.fromisoformat(self.stats.end_time)
            self.stats.duration_seconds = (end - start).total_seconds()
        
        if self.stats.total_qa_pairs > 0:
            self.stats.pass_rate = self.stats.passed_qa_pairs / self.stats.total_qa_pairs
        
        if self.stats.processed_chunks > 0:
            self.stats.questions_per_chunk_avg = \
                self.stats.total_questions_generated / self.stats.processed_chunks
    
    def _save_checkpoint(self, chunk_index: int):
        """Save checkpoint for resuming."""
        checkpoint_path = os.path.join(
            self.config.output_dir, 
            f"checkpoint_{chunk_index}.json"
        )
        
        checkpoint = {
            "chunk_index": chunk_index,
            "dataset_size": len(self.dataset),
            "stats": self.stats.to_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)
        
        self._log(f"   💾 Checkpoint sauvegardé: {checkpoint_path}")
    
    # =========================================================================
    # EXPORT METHODS
    # =========================================================================
    
    def export_json(self, filepath: Optional[str] = None) -> str:
        """Export dataset to JSON file."""
        if filepath is None:
            filepath = os.path.join(self.config.output_dir, "dataset.json")
        
        output = {
            "metadata": {
                "source_file": os.path.basename(self.config.pdf_path),
                "generation_date": datetime.now().isoformat(),
                "total_entries": len(self.dataset),
                "config": self.config.to_dict(),
                "stats": self.stats.to_dict()
            },
            "data": [entry.to_dict() for entry in self.dataset]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        self._log(f"📁 Dataset exporté: {filepath}")
        return filepath
    
    def export_huggingface(self, filepath: Optional[str] = None) -> str:
        """Export dataset in HuggingFace format (JSONL)."""
        if filepath is None:
            filepath = os.path.join(self.config.output_dir, "dataset_hf.jsonl")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for entry in self.dataset:
                f.write(json.dumps(entry.to_huggingface_format(), ensure_ascii=False) + "\n")
        
        self._log(f"📁 Dataset HuggingFace exporté: {filepath}")
        return filepath
    
    def export_csv(self, filepath: Optional[str] = None) -> str:
        """Export dataset to CSV file."""
        import csv
        
        if filepath is None:
            filepath = os.path.join(self.config.output_dir, "dataset.csv")
        
        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "question", "answer", "source_file", "chunk_id", 
                "chapter", "section", "question_type", "difficulty", "quality_score"
            ])
            
            for entry in self.dataset:
                writer.writerow([
                    entry.question,
                    entry.answer,
                    entry.source_file,
                    entry.chunk_id,
                    entry.chapter,
                    entry.section,
                    entry.question_type,
                    entry.difficulty,
                    entry.critic_score
                ])
        
        self._log(f"📁 Dataset CSV exporté: {filepath}")
        return filepath
    
    def print_summary(self):
        """Print a summary of the pipeline run (with diversity report)."""
        print("\n" + "=" * 60)
        print("RÉSUMÉ DU PIPELINE")
        print("=" * 60)
        print(f"\n📄 Source: {os.path.basename(self.config.pdf_path)}")
        print(f"⏱️  Durée: {self.stats.duration_seconds:.1f} secondes")
        print(f"\n📊 STATISTIQUES:")
        print(f"   Chunks traités: {self.stats.processed_chunks}/{self.stats.total_chunks}")
        print(f"   Questions générées: {self.stats.total_questions_generated}")
        print(f"   QA pairs évalués: {self.stats.total_qa_pairs}")
        print(f"   ✅ Acceptés: {self.stats.passed_qa_pairs} ({100*self.stats.pass_rate:.1f}%)")
        print(f"   ❌ Rejetés: {self.stats.rejected_qa_pairs}")
        
        if self.config.enable_diversity_check:
            print(f"   🚫 Duplicates détectés: {self.stats.duplicates_detected}")
        
        if self.stats.rejection_reasons:
            print(f"\n📉 Raisons de rejet:")
            for criterion, count in sorted(self.stats.rejection_reasons.items(), key=lambda x: -x[1]):
                print(f"   - {criterion}: {count}")
        
        # Distribution report (NEW)
        if self.stats.type_distribution:
            print(f"\n🎯 Distribution des types:")
            total = sum(self.stats.type_distribution.values())
            for qtype, count in sorted(self.stats.type_distribution.items()):
                pct = 100 * count / total
                bar = "█" * int(pct / 5)
                print(f"   {qtype:12} : {count:2} ({pct:5.1f}%) {bar}")
        
        if self.stats.difficulty_distribution:
            print(f"\n📊 Distribution des difficultés:")
            total = sum(self.stats.difficulty_distribution.values())
            for diff in ['easy', 'medium', 'hard']:
                count = self.stats.difficulty_distribution.get(diff, 0)
                pct = 100 * count / total if total > 0 else 0
                bar = "█" * int(pct / 5)
                print(f"   {diff:12} : {count:2} ({pct:5.1f}%) {bar}")
        
        if self.stats.errors:
            print(f"\n⚠️  Erreurs ({len(self.stats.errors)}):")
            for err in self.stats.errors[:5]:
                print(f"   - {err[:80]}...")
        
        print(f"\n📁 Dataset final: {len(self.dataset)} entrées")
        print("=" * 60)


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def generate_dataset(
    pdf_path: str,
    llm_client: Any,
    output_dir: str = "output",
    max_chunks: Optional[int] = None,
    questions_per_chunk: int = 2,
    language: str = "fr"
) -> Tuple[List[DatasetEntry], PipelineStats]:
    """
    Convenience function to generate a dataset from a PDF.
    
    Args:
        pdf_path: Path to PDF file
        llm_client: LLM API client
        output_dir: Output directory
        max_chunks: Maximum chunks to process (None = all)
        questions_per_chunk: Questions per chunk
        language: "fr" or "en"
        
    Returns:
        (dataset entries, statistics)
    """
    config = PipelineConfig(
        pdf_path=pdf_path,
        output_dir=output_dir,
        max_chunks=max_chunks,
        questions_per_chunk=questions_per_chunk,
        language=language
    )
    
    pipeline = DatasetPipeline(config, llm_client)
    dataset = pipeline.run()
    
    # Export all formats
    pipeline.export_json()
    pipeline.export_huggingface()
    pipeline.export_csv()
    
    pipeline.print_summary()
    
    return dataset, pipeline.stats


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Dataset Generation Pipeline")
    print("=" * 60)
    print()
    print("Usage:")
    print("  from pipeline import DatasetPipeline, PipelineConfig")
    print()
    print("  config = PipelineConfig(")
    print("      pdf_path='document.pdf',")
    print("      output_dir='output',")
    print("      max_chunks=10,")
    print("      questions_per_chunk=2")
    print("  )")
    print()
    print("  pipeline = DatasetPipeline(config, llm_client)")
    print("  dataset = pipeline.run()")
    print("  pipeline.export_json()")
    print("  pipeline.export_huggingface()")
    print()
    print("Or use the convenience function:")
    print("  from pipeline import generate_dataset")
    print("  dataset, stats = generate_dataset('document.pdf', llm_client)")
