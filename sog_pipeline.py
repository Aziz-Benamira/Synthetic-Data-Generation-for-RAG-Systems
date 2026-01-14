"""
Synthesize-on-Graph (SoG) Pipeline

This is the main orchestrator that brings together all components of the SoG framework:
- Context Graph Construction
- Two-Stage Sampling Strategy
- Adaptive Generation (CoT + CC)
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path
import logging
from tqdm import tqdm

from context_graph import ContextGraph, Paragraph
from sampling_strategy import SamplingStrategy, SamplingPath, EmbeddingFunction
from generation_strategies import (
    ChainOfThoughtGenerator,
    ContrastiveClarifyingGenerator,
    AdaptiveStrategySelector,
    SyntheticSample,
    LLMInterface,
    MockLLMInterface
)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SoGConfig:
    """Configuration for the SoG pipeline."""
    
    # Sampling parameters
    max_depth: int = 3
    top_k_neighbors: int = 5
    long_tail_boost: float = 1.5
    
    # Generation parameters
    sparse_threshold: int = 3
    utilization_threshold: float = 0.1
    
    # LLM parameters
    max_tokens: int = 1500
    temperature: float = 0.7
    
    # Output parameters
    num_samples: int = 100
    random_seed: Optional[int] = 42
    output_path: Optional[str] = "synthetic_data.jsonl"
    
    # Processing parameters
    batch_size: int = 10
    verbose: bool = True


@dataclass
class Document:
    """Represents a document with its paragraphs."""
    doc_id: int
    title: str
    paragraphs: List[str]
    entities_per_paragraph: List[List[str]]
    
    def validate(self) -> bool:
        """Validate document structure."""
        return len(self.paragraphs) == len(self.entities_per_paragraph)


class SynthesizeOnGraph:
    """
    Main SoG pipeline orchestrator.
    
    Integrates all components to generate synthetic data from a corpus:
    1. Build context graph from documents
    2. Sample paths using two-stage strategy
    3. Generate synthetic samples using adaptive strategies
    """
    
    def __init__(self, 
                 config: Optional[SoGConfig] = None,
                 llm: Optional[LLMInterface] = None,
                 embedding_func: Optional[EmbeddingFunction] = None):
        """
        Initialize the SoG pipeline.
        
        Args:
            config: Configuration object
            llm: LLM interface for generation
            embedding_func: Embedding function for similarity computation
        """
        self.config = config or SoGConfig()
        self.llm = llm or MockLLMInterface()
        self.embedding_func = embedding_func
        
        # Initialize components
        self.context_graph = ContextGraph()
        self.sampling_strategy = None  # Initialized after graph construction
        
        # Initialize generators
        self.cot_generator = ChainOfThoughtGenerator(self.llm)
        self.cc_generator = ContrastiveClarifyingGenerator(self.llm)
        self.strategy_selector = AdaptiveStrategySelector(
            self.cot_generator,
            self.cc_generator,
            sparse_threshold=self.config.sparse_threshold,
            utilization_threshold=self.config.utilization_threshold
        )
        
        # Statistics
        self.stats = {
            'total_samples_generated': 0,
            'cot_samples': 0,
            'cc_samples': 0,
            'failed_generations': 0,
            'avg_path_length': 0.0
        }
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the context graph.
        
        Args:
            documents: List of documents to add
        """
        logger.info(f"Adding {len(documents)} documents to context graph...")
        
        for doc in tqdm(documents, disable=not self.config.verbose):
            if not doc.validate():
                logger.warning(f"Skipping invalid document {doc.doc_id}")
                continue
            
            self.context_graph.add_document(
                doc.doc_id,
                doc.paragraphs,
                doc.entities_per_paragraph
            )
        
        # Log graph statistics
        stats = self.context_graph.get_graph_statistics()
        logger.info(f"Context graph built: {stats['num_entities']} entities, "
                   f"{stats['num_edges']} edges, {stats['num_paragraphs']} paragraphs")
        logger.info(f"Graph density: {stats['density']:.4f}, "
                   f"Avg degree: {stats['avg_degree']:.2f}")
        
        # Initialize sampling strategy after graph construction
        self.sampling_strategy = SamplingStrategy(
            self.context_graph,
            embedding_func=self.embedding_func,
            max_depth=self.config.max_depth,
            top_k_neighbors=self.config.top_k_neighbors,
            long_tail_boost=self.config.long_tail_boost
        )
    
    def generate_synthetic_data(self, num_samples: Optional[int] = None) -> List[SyntheticSample]:
        """
        Generate synthetic data samples.
        
        Args:
            num_samples: Number of samples to generate (overrides config)
            
        Returns:
            List of generated synthetic samples
        """
        if self.sampling_strategy is None:
            raise RuntimeError("No documents added. Call add_documents() first.")
        
        num_samples = num_samples or self.config.num_samples
        logger.info(f"Generating {num_samples} synthetic samples...")
        
        # Sample paths
        logger.info("Sampling paths from context graph...")
        paths = self.sampling_strategy.sample_paths(
            num_samples,
            random_seed=self.config.random_seed
        )
        
        if not paths:
            logger.warning("No paths sampled from context graph")
            return []
        
        logger.info(f"Sampled {len(paths)} paths")
        
        # Generate synthetic samples
        samples = []
        path_lengths = []
        
        logger.info("Generating synthetic samples from paths...")
        for path in tqdm(paths, disable=not self.config.verbose):
            try:
                # Generate sample using adaptive strategy
                sample = self.strategy_selector.generate(path, self.context_graph)
                
                if sample is not None:
                    samples.append(sample)
                    path_lengths.append(len(path))
                    
                    # Update statistics
                    self.stats['total_samples_generated'] += 1
                    if sample.strategy_used.value == 'cot':
                        self.stats['cot_samples'] += 1
                    else:
                        self.stats['cc_samples'] += 1
                else:
                    self.stats['failed_generations'] += 1
                    
            except Exception as e:
                logger.warning(f"Failed to generate sample: {e}")
                self.stats['failed_generations'] += 1
        
        # Update statistics
        if path_lengths:
            self.stats['avg_path_length'] = sum(path_lengths) / len(path_lengths)
        
        logger.info(f"Successfully generated {len(samples)} samples")
        self._log_statistics()
        
        return samples
    
    def save_samples(self, samples: List[SyntheticSample], output_path: Optional[str] = None) -> None:
        """
        Save synthetic samples to file.
        
        Args:
            samples: List of synthetic samples
            output_path: Output file path (overrides config)
        """
        output_path = output_path or self.config.output_path
        
        if output_path is None:
            logger.warning("No output path specified. Samples not saved.")
            return
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving {len(samples)} samples to {output_path}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + '\n')
        
        logger.info(f"Samples saved to {output_path}")
    
    def _log_statistics(self) -> None:
        """Log generation statistics."""
        logger.info("=== Generation Statistics ===")
        logger.info(f"Total samples generated: {self.stats['total_samples_generated']}")
        logger.info(f"CoT samples: {self.stats['cot_samples']}")
        logger.info(f"CC samples: {self.stats['cc_samples']}")
        logger.info(f"Failed generations: {self.stats['failed_generations']}")
        logger.info(f"Average path length: {self.stats['avg_path_length']:.2f}")
        
        if self.stats['total_samples_generated'] > 0:
            cot_ratio = self.stats['cot_samples'] / self.stats['total_samples_generated']
            cc_ratio = self.stats['cc_samples'] / self.stats['total_samples_generated']
            logger.info(f"CoT ratio: {cot_ratio:.2%}")
            logger.info(f"CC ratio: {cc_ratio:.2%}")
    
    def get_statistics(self) -> Dict:
        """
        Get generation statistics.
        
        Returns:
            Dictionary of statistics
        """
        graph_stats = self.context_graph.get_graph_statistics()
        return {
            'generation': self.stats,
            'graph': graph_stats
        }
    
    def run_pipeline(self, documents: List[Document], 
                    num_samples: Optional[int] = None,
                    output_path: Optional[str] = None) -> List[SyntheticSample]:
        """
        Run the complete SoG pipeline.
        
        Args:
            documents: List of documents to process
            num_samples: Number of samples to generate
            output_path: Output file path
            
        Returns:
            List of generated synthetic samples
        """
        # Step 1: Build context graph
        self.add_documents(documents)
        
        # Step 2: Generate synthetic data
        samples = self.generate_synthetic_data(num_samples)
        
        # Step 3: Save samples
        if output_path or self.config.output_path:
            self.save_samples(samples, output_path)
        
        return samples


def load_documents_from_json(json_path: str) -> List[Document]:
    """
    Load documents from a JSON file.
    
    Expected JSON format:
    [
        {
            "doc_id": 0,
            "title": "Document Title",
            "paragraphs": ["Para 1 text", "Para 2 text", ...],
            "entities_per_paragraph": [["entity1", "entity2"], ["entity3"], ...]
        },
        ...
    ]
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        List of Document objects
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    for item in data:
        doc = Document(
            doc_id=item['doc_id'],
            title=item.get('title', f"Document {item['doc_id']}"),
            paragraphs=item['paragraphs'],
            entities_per_paragraph=item['entities_per_paragraph']
        )
        documents.append(doc)
    
    return documents


def load_documents_from_jsonl(jsonl_path: str) -> List[Document]:
    """
    Load documents from a JSONL file (one document per line).
    
    Args:
        jsonl_path: Path to JSONL file
        
    Returns:
        List of Document objects
    """
    documents = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            doc = Document(
                doc_id=item['doc_id'],
                title=item.get('title', f"Document {item['doc_id']}"),
                paragraphs=item['paragraphs'],
                entities_per_paragraph=item['entities_per_paragraph']
            )
            documents.append(doc)
    
    return documents
