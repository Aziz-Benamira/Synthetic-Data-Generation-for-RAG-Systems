"""
Sampling Strategy Module for Synthesize-on-Graph (SoG)

This module implements the two-stage sampling strategy:
1. Graph Traversal with Similarity-Based Selection (BFS)
2. Secondary Sampling for Long-Tail Entity Mitigation
"""

from typing import List, Tuple, Optional, Callable, Set
from dataclasses import dataclass
from collections import deque
import numpy as np
from context_graph import ContextGraph, Paragraph
import random


@dataclass
class SamplingPath:
    """
    Represents a sampling path P = [(e_root, q^(0)), (e_1, c_1), ..., (e_n, c_n)]
    where n ≤ D (maximum depth).
    """
    path: List[Tuple[str, Paragraph]]  # List of (entity, paragraph) pairs
    
    def __len__(self):
        return len(self.path)
    
    def get_root_entity(self) -> str:
        """Get the root entity of the path."""
        return self.path[0][0] if self.path else None
    
    def get_root_paragraph(self) -> Paragraph:
        """Get the root paragraph of the path."""
        return self.path[0][1] if self.path else None
    
    def get_entities(self) -> List[str]:
        """Get all entities in the path."""
        return [entity for entity, _ in self.path]
    
    def get_paragraphs(self) -> List[Paragraph]:
        """Get all paragraphs in the path."""
        return [para for _, para in self.path]
    
    def contains_long_tail_entity(self, long_tail_entities: Set[str]) -> bool:
        """Check if the path contains any long-tail entities."""
        return any(entity in long_tail_entities for entity in self.get_entities())


class EmbeddingFunction:
    """
    Abstract embedding function for similarity computation.
    In practice, this should be replaced with actual embedding models.
    """
    
    def embed(self, text: str) -> np.ndarray:
        """
        Embed text into a vector representation.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        raise NotImplementedError("This should be implemented with an actual embedding model")
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts.
        
        F_sim(q^(0), c) = dot(embed(q^(0)), embed(c))
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score
        """
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        return float(np.dot(emb1, emb2))


class MockEmbeddingFunction(EmbeddingFunction):
    """
    Mock embedding function for testing.
    Uses simple hashing for reproducibility.
    """
    
    def __init__(self, dim: int = 128):
        self.dim = dim
        self.cache = {}
    
    def embed(self, text: str) -> np.ndarray:
        """Generate a mock embedding based on text hash."""
        if text in self.cache:
            return self.cache[text]
        
        # Simple hash-based embedding for testing
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self.dim)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        self.cache[text] = embedding
        return embedding


class SamplingStrategy:
    """
    Implements the two-stage sampling strategy for SoG framework.
    """
    
    def __init__(self, 
                 context_graph: ContextGraph,
                 embedding_func: Optional[EmbeddingFunction] = None,
                 max_depth: int = 3,
                 top_k_neighbors: int = 5,
                 long_tail_boost: float = 1.5):
        """
        Initialize sampling strategy.
        
        Args:
            context_graph: The context graph to sample from
            embedding_func: Function to compute embeddings for similarity
            max_depth: Maximum depth D for BFS traversal
            top_k_neighbors: Number of top neighbors to consider at each step
            long_tail_boost: Boost factor for paths containing long-tail entities
        """
        self.graph = context_graph
        self.embedding_func = embedding_func or MockEmbeddingFunction()
        self.max_depth = max_depth
        self.top_k_neighbors = top_k_neighbors
        self.long_tail_boost = long_tail_boost
        self.long_tail_entities = self.graph.get_long_tail_entities()
    
    def _compute_similarity_scores(self, root_para: Paragraph, 
                                   candidate_paras: List[Paragraph]) -> List[float]:
        """
        Compute similarity scores between root paragraph and candidates.
        
        Args:
            root_para: Root paragraph q^(0)
            candidate_paras: List of candidate paragraphs
            
        Returns:
            List of similarity scores
        """
        root_text = root_para.text
        scores = []
        
        for para in candidate_paras:
            score = self.embedding_func.similarity(root_text, para.text)
            scores.append(score)
        
        return scores
    
    def _select_top_k_paragraphs(self, root_para: Paragraph, 
                                candidate_paras: List[Paragraph],
                                k: Optional[int] = None) -> List[Paragraph]:
        """
        Select top-k most similar paragraphs.
        
        Args:
            root_para: Root paragraph for similarity computation
            candidate_paras: List of candidate paragraphs
            k: Number of paragraphs to select (defaults to top_k_neighbors)
            
        Returns:
            List of selected paragraphs
        """
        if not candidate_paras:
            return []
        
        k = k or self.top_k_neighbors
        scores = self._compute_similarity_scores(root_para, candidate_paras)
        
        # Sort by score (descending) and take top-k
        sorted_indices = np.argsort(scores)[::-1][:k]
        return [candidate_paras[i] for i in sorted_indices]
    
    def bfs_traversal(self, root_entity: str, root_paragraph: Paragraph) -> List[SamplingPath]:
        """
        Perform BFS traversal starting from root entity up to max_depth.
        
        This implements Stage 1: Graph Traversal with Similarity-Based Selection
        
        Args:
            root_entity: Starting entity e_root
            root_paragraph: Starting paragraph q^(0)
            
        Returns:
            List of sampling paths
        """
        paths = []
        queue = deque([(root_entity, root_paragraph, [(root_entity, root_paragraph)], {root_entity}, 0)])
        
        while queue:
            current_entity, current_para, current_path, visited_entities, depth = queue.popleft()
            
            # If we've reached max depth, save this path
            if depth >= self.max_depth:
                paths.append(SamplingPath(current_path))
                continue
            
            # Get neighboring entities
            neighbors = self.graph.get_neighbors(current_entity)
            
            # Filter out already visited entities
            unvisited_neighbors = [n for n in neighbors if n not in visited_entities]
            
            if not unvisited_neighbors:
                # No more neighbors to explore, save this path
                paths.append(SamplingPath(current_path))
                continue
            
            # Get candidate paragraphs from neighboring entities
            candidate_paras = []
            entity_to_paras = {}
            
            for neighbor in unvisited_neighbors:
                neighbor_paras = list(self.graph.get_entity_paragraphs(neighbor))
                if neighbor_paras:
                    entity_to_paras[neighbor] = neighbor_paras
                    candidate_paras.extend([(neighbor, para) for para in neighbor_paras])
            
            if not candidate_paras:
                paths.append(SamplingPath(current_path))
                continue
            
            # Select top-k most similar paragraphs
            paras_only = [para for _, para in candidate_paras]
            selected_paras = self._select_top_k_paragraphs(root_paragraph, paras_only, k=self.top_k_neighbors)
            
            # Map back to (entity, paragraph) pairs
            selected_pairs = [(entity, para) for entity, para in candidate_paras if para in selected_paras]
            
            # Add selected neighbors to queue
            for next_entity, next_para in selected_pairs:
                new_path = current_path + [(next_entity, next_para)]
                new_visited = visited_entities | {next_entity}
                queue.append((next_entity, next_para, new_path, new_visited, depth + 1))
        
        return paths
    
    def secondary_sampling(self, paths: List[SamplingPath], num_samples: int) -> List[SamplingPath]:
        """
        Perform secondary sampling to prioritize paths with long-tail entities.
        
        This implements Stage 2: Secondary Sampling and Controlled Allocation
        
        Args:
            paths: List of candidate paths from BFS traversal
            num_samples: Number of paths to sample
            
        Returns:
            Selected paths with balanced entity distribution
        """
        if not paths or num_samples == 0:
            return []
        
        # Compute weights for each path
        weights = []
        for path in paths:
            weight = 1.0
            
            # Boost weight if path contains long-tail entities
            if path.contains_long_tail_entity(self.long_tail_entities):
                weight *= self.long_tail_boost
            
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Sample paths according to weights
        num_samples = min(num_samples, len(paths))
        selected_indices = np.random.choice(len(paths), size=num_samples, replace=False, p=weights)
        
        return [paths[i] for i in selected_indices]
    
    def sample_paths(self, num_paths: int, random_seed: Optional[int] = None) -> List[SamplingPath]:
        """
        Sample paths from the context graph using the two-stage strategy.
        
        Args:
            num_paths: Number of paths to sample
            random_seed: Random seed for reproducibility
            
        Returns:
            List of sampled paths
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Select random root entities and paragraphs
        all_entities = list(self.graph.entity_mapping.keys())
        if not all_entities:
            return []
        
        all_paths = []
        
        # Generate more paths than needed for secondary sampling
        num_roots = min(len(all_entities), num_paths * 2)
        root_entities = random.sample(all_entities, num_roots)
        
        for root_entity in root_entities:
            # Get a random paragraph containing this entity
            entity_paras = list(self.graph.get_entity_paragraphs(root_entity))
            if not entity_paras:
                continue
            
            root_para = random.choice(entity_paras)
            
            # Perform BFS traversal
            paths = self.bfs_traversal(root_entity, root_para)
            all_paths.extend(paths)
        
        # Secondary sampling to balance entity distribution
        selected_paths = self.secondary_sampling(all_paths, num_paths)
        
        return selected_paths
