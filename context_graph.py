"""
Context Graph Construction Module for Synthesize-on-Graph (SoG)

This module implements the context graph construction component of the SoG framework,
which builds a graph where nodes represent entities and edges represent cross-document
knowledge associations.
"""

from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
import networkx as nx
from collections import defaultdict
import numpy as np


@dataclass
class Paragraph:
    """Represents a paragraph with its document and paragraph indices."""
    doc_id: int
    para_id: int
    text: str
    entities: List[str] = field(default_factory=list)
    
    def __hash__(self):
        return hash((self.doc_id, self.para_id))
    
    def __eq__(self, other):
        return (self.doc_id, self.para_id) == (other.doc_id, other.para_id)


@dataclass
class Entity:
    """Represents an entity with its associated paragraphs."""
    name: str
    paragraphs: Set[Paragraph] = field(default_factory=set)
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        return self.name == other.name


class ContextGraph:
    """
    Context Graph implementation for SoG framework.
    
    The graph G = (E, ε) where:
    - E: nodes representing entities
    - ε: edges representing cross-document knowledge associations
    
    Edge (e_x, e_y) exists if ∃i,j such that e_x, e_y ∈ E_{i,j}
    where E_{i,j} is the set of entities extracted from paragraph j of document i.
    """
    
    def __init__(self):
        self.graph = nx.Graph()
        self.entity_mapping: Dict[str, Entity] = {}  # M: e_k → P_k
        self.paragraphs: List[Paragraph] = []
        self.entity_frequency: Dict[str, int] = defaultdict(int)
        
    def add_document(self, doc_id: int, paragraphs_text: List[str], 
                    paragraphs_entities: List[List[str]]) -> None:
        """
        Add a document to the context graph.
        
        Args:
            doc_id: Document identifier
            paragraphs_text: List of paragraph texts
            paragraphs_entities: List of entity lists, one per paragraph
        """
        for para_id, (text, entities) in enumerate(zip(paragraphs_text, paragraphs_entities)):
            # Create paragraph object
            paragraph = Paragraph(doc_id, para_id, text, entities)
            self.paragraphs.append(paragraph)
            
            # Update entity-context mapping
            for entity_name in entities:
                if entity_name not in self.entity_mapping:
                    entity = Entity(entity_name)
                    self.entity_mapping[entity_name] = entity
                    self.graph.add_node(entity_name)
                
                self.entity_mapping[entity_name].paragraphs.add(paragraph)
                self.entity_frequency[entity_name] += 1
            
            # Create edges between co-occurring entities in the same paragraph
            # This represents cross-document associations
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i+1:]:
                    if not self.graph.has_edge(entity1, entity2):
                        self.graph.add_edge(entity1, entity2, weight=1)
                    else:
                        # Increment edge weight for co-occurrence frequency
                        self.graph[entity1][entity2]['weight'] += 1
    
    def get_entity_paragraphs(self, entity_name: str) -> Set[Paragraph]:
        """
        Get all paragraphs associated with an entity.
        
        Args:
            entity_name: Name of the entity
            
        Returns:
            Set of paragraphs containing the entity
        """
        if entity_name in self.entity_mapping:
            return self.entity_mapping[entity_name].paragraphs
        return set()
    
    def get_neighbors(self, entity_name: str) -> List[str]:
        """
        Get neighboring entities in the context graph.
        
        Args:
            entity_name: Name of the entity
            
        Returns:
            List of neighboring entity names
        """
        if entity_name in self.graph:
            return list(self.graph.neighbors(entity_name))
        return []
    
    def get_cross_document_paragraphs(self, entity1: str, entity2: str) -> Tuple[Set[Paragraph], Set[Paragraph]]:
        """
        Get paragraphs from different documents containing two entities.
        
        Args:
            entity1: First entity name
            entity2: Second entity name
            
        Returns:
            Tuple of (entity1_paragraphs, entity2_paragraphs)
        """
        paras1 = self.get_entity_paragraphs(entity1)
        paras2 = self.get_entity_paragraphs(entity2)
        
        # Filter to keep only paragraphs from different documents
        cross_doc_paras1 = {p1 for p1 in paras1 if any(p1.doc_id != p2.doc_id for p2 in paras2)}
        cross_doc_paras2 = {p2 for p2 in paras2 if any(p2.doc_id != p1.doc_id for p1 in paras1)}
        
        return cross_doc_paras1, cross_doc_paras2
    
    def get_entity_degree(self, entity_name: str) -> int:
        """Get the degree (number of connections) of an entity."""
        if entity_name in self.graph:
            return self.graph.degree(entity_name)
        return 0
    
    def is_sparse_entity(self, entity_name: str, threshold: int = 3) -> bool:
        """
        Determine if an entity is sparse (has limited connections).
        
        Args:
            entity_name: Name of the entity
            threshold: Minimum degree for non-sparse entity
            
        Returns:
            True if entity is sparse (degree < threshold)
        """
        return self.get_entity_degree(entity_name) < threshold
    
    def get_entity_utilization_rate(self, entity_name: str, total_samples: int) -> float:
        """
        Calculate the utilization rate of an entity.
        
        Args:
            entity_name: Name of the entity
            total_samples: Total number of samples generated
            
        Returns:
            Utilization rate (frequency / total_samples)
        """
        if total_samples == 0:
            return 0.0
        return self.entity_frequency.get(entity_name, 0) / total_samples
    
    def get_long_tail_entities(self, percentile: float = 0.25) -> Set[str]:
        """
        Identify long-tail entities (those in the lower percentile of frequency).
        
        Args:
            percentile: Percentile threshold for long-tail entities
            
        Returns:
            Set of entity names in the long tail
        """
        frequencies = list(self.entity_frequency.values())
        if not frequencies:
            return set()
        
        threshold = np.percentile(frequencies, percentile * 100)
        return {entity for entity, freq in self.entity_frequency.items() if freq <= threshold}
    
    def get_graph_statistics(self) -> Dict:
        """
        Get statistics about the context graph.
        
        Returns:
            Dictionary containing graph statistics
        """
        return {
            'num_entities': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'num_paragraphs': len(self.paragraphs),
            'avg_degree': sum(dict(self.graph.degree()).values()) / max(self.graph.number_of_nodes(), 1),
            'density': nx.density(self.graph),
            'num_connected_components': nx.number_connected_components(self.graph),
            'entity_frequency_stats': {
                'mean': np.mean(list(self.entity_frequency.values())) if self.entity_frequency else 0,
                'std': np.std(list(self.entity_frequency.values())) if self.entity_frequency else 0,
                'min': min(self.entity_frequency.values()) if self.entity_frequency else 0,
                'max': max(self.entity_frequency.values()) if self.entity_frequency else 0,
            }
        }
