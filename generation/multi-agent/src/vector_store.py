"""
Vector Store with ChromaDB
==========================
Handles vector embeddings storage with metadata for hybrid search.

Features:
- Semantic search with embeddings
- Metadata filtering (chapter, page, section)
- Hybrid search (vector + keyword)
"""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np


class VectorStore:
    """ChromaDB-based vector store with metadata indexing"""
    
    def __init__(
        self,
        collection_name: str = "textbook_chunks",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize vector store.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Where to store the database
            embedding_model: OpenAI embedding model name
        """
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Set up embedding function
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            model_name=embedding_model
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ):
        """
        Add documents to the vector store.
        
        Args:
            texts: List of text chunks
            metadatas: List of metadata dicts with keys:
                - page_number: int
                - section_id: str (e.g., "1.2")
                - chapter: str (e.g., "ch1")
                - chunk_index: int
            ids: Optional custom IDs (auto-generated if None)
        
        Example:
            store.add_documents(
                texts=["Newton's laws describe motion..."],
                metadatas=[{
                    "page_number": 5,
                    "section_id": "1.1",
                    "chapter": "ch1",
                    "chunk_index": 0
                }]
            )
        """
        if ids is None:
            # Generate IDs: chunk_ch1_sec1.1_page5_idx0
            ids = [
                f"chunk_{m['chapter']}_sec{m['section_id']}_page{m['page_number']}_idx{m['chunk_index']}"
                for m in metadatas
            ]
        
        # Validate metadata structure
        required_keys = {"page_number", "section_id", "chapter", "chunk_index"}
        for meta in metadatas:
            if not required_keys.issubset(meta.keys()):
                raise ValueError(f"Metadata missing required keys: {required_keys}")
        
        # Add to collection
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Added {len(texts)} documents to vector store")
    
    def search(
        self,
        query: str,
        k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search with optional metadata filtering.
        
        Args:
            query: Search query
            k: Number of results
            where: Metadata filter (e.g., {"chapter": "ch1"})
            where_document: Document content filter
        
        Returns:
            List of results with text, metadata, and distance
        
        Example:
            # Search in Chapter 1 only
            results = store.search(
                query="What is entropy?",
                k=5,
                where={"chapter": "ch1"}
            )
            
            # Search in specific page range
            results = store.search(
                query="Newton's laws",
                k=3,
                where={"$and": [
                    {"page_number": {"$gte": 1}},
                    {"page_number": {"$lte": 20}}
                ]}
            )
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where,
            where_document=where_document
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "id": results['ids'][0][i],
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            })
        
        return formatted_results
    
    def hybrid_search(
        self,
        query: str,
        keywords: List[str],
        k: int = 5,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector and keyword matching.
        
        Args:
            query: Semantic search query
            keywords: Keywords for exact matching
            k: Number of results
            vector_weight: Weight for vector similarity (0-1)
            keyword_weight: Weight for keyword matching (0-1)
            where: Metadata filter
        
        Returns:
            Ranked list of results
        
        Algorithm:
            1. Perform vector search
            2. Perform keyword search
            3. Combine scores with weights
            4. Re-rank and return top-k
        """
        # Vector search
        vector_results = self.search(query=query, k=k*2, where=where)
        
        # Keyword search
        keyword_results = []
        for keyword in keywords:
            kw_res = self.search(
                query=query,  # Still use query for embedding
                k=k*2,
                where=where,
                where_document={"$contains": keyword}
            )
            keyword_results.extend(kw_res)
        
        # Combine and re-rank
        combined_scores = {}
        
        # Add vector scores
        for result in vector_results:
            doc_id = result['id']
            # Convert distance to similarity (1 - distance for cosine)
            similarity = 1 - result['distance']
            combined_scores[doc_id] = {
                "score": similarity * vector_weight,
                "result": result
            }
        
        # Add keyword scores
        for result in keyword_results:
            doc_id = result['id']
            similarity = 1 - result['distance']
            
            if doc_id in combined_scores:
                combined_scores[doc_id]["score"] += similarity * keyword_weight
            else:
                combined_scores[doc_id] = {
                    "score": similarity * keyword_weight,
                    "result": result
                }
        
        # Sort by combined score
        ranked_results = sorted(
            combined_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        # Return top-k with combined scores
        return [
            {**r["result"], "combined_score": r["score"]}
            for r in ranked_results[:k]
        ]
    
    def get_by_metadata(
        self,
        **metadata_filters
    ) -> List[Dict[str, Any]]:
        """
        Get all documents matching metadata criteria.
        
        Args:
            **metadata_filters: Key-value pairs to filter by
        
        Example:
            # Get all chunks from Chapter 1, Section 1.2
            chunks = store.get_by_metadata(
                chapter="ch1",
                section_id="1.2"
            )
        """
        where_clause = {
            "$and": [
                {key: value} for key, value in metadata_filters.items()
            ]
        } if len(metadata_filters) > 1 else metadata_filters
        
        results = self.collection.get(
            where=where_clause,
            include=["documents", "metadatas"]
        )
        
        formatted_results = []
        for i in range(len(results['ids'])):
            formatted_results.append({
                "id": results['ids'][i],
                "text": results['documents'][i],
                "metadata": results['metadatas'][i]
            })
        
        return formatted_results
    
    def delete_collection(self):
        """Delete the entire collection"""
        self.client.delete_collection(name=self.collection_name)
        print(f"🗑️  Deleted collection: {self.collection_name}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        count = self.collection.count()
        
        # Get sample of metadata to analyze
        sample = self.collection.get(limit=100, include=["metadatas"])
        
        chapters = set()
        sections = set()
        page_range = [float('inf'), 0]
        
        for meta in sample['metadatas']:
            chapters.add(meta.get('chapter', 'unknown'))
            sections.add(meta.get('section_id', 'unknown'))
            page_num = meta.get('page_number', 0)
            page_range[0] = min(page_range[0], page_num)
            page_range[1] = max(page_range[1], page_num)
        
        return {
            "total_chunks": count,
            "unique_chapters": len(chapters),
            "unique_sections": len(sections),
            "page_range": page_range if page_range[0] != float('inf') else [0, 0],
            "collection_name": self.collection_name
        }


# Utility functions for chunking

def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    separator: str = "\n\n"
) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Target size in characters
        overlap: Overlap between chunks
        separator: Prefer splitting on this separator
    
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        
        # If not at the end, try to find separator
        if end < text_length:
            # Look for separator in the last 20% of chunk
            search_start = end - int(chunk_size * 0.2)
            sep_pos = text.rfind(separator, search_start, end)
            
            if sep_pos != -1:
                end = sep_pos + len(separator)
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks


def create_chunk_metadata(
    chunk_index: int,
    page_number: int,
    section_id: str,
    chapter: str,
    additional_meta: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create metadata dict for a chunk.
    
    Args:
        chunk_index: Index of chunk within section
        page_number: Page number
        section_id: Section ID (e.g., "1.2")
        chapter: Chapter ID (e.g., "ch1")
        additional_meta: Optional extra metadata
    
    Returns:
        Metadata dictionary
    """
    metadata = {
        "chunk_index": chunk_index,
        "page_number": page_number,
        "section_id": section_id,
        "chapter": chapter
    }
    
    if additional_meta:
        metadata.update(additional_meta)
    
    return metadata


# Example usage
if __name__ == "__main__":
    # Initialize store
    store = VectorStore(collection_name="test_textbook")
    
    # Add sample documents
    texts = [
        "Newton's First Law states that an object at rest stays at rest.",
        "Newton's Second Law: F = ma relates force, mass, and acceleration.",
        "Newton's Third Law: For every action, there is an equal and opposite reaction."
    ]
    
    metadatas = [
        create_chunk_metadata(0, 5, "1.1", "ch1"),
        create_chunk_metadata(1, 5, "1.1", "ch1"),
        create_chunk_metadata(2, 6, "1.1", "ch1")
    ]
    
    store.add_documents(texts, metadatas)
    
    # Search
    results = store.search("What is Newton's second law?", k=2)
    print("\n🔍 Search Results:")
    for r in results:
        print(f"  - {r['text'][:50]}... (page {r['metadata']['page_number']})")
    
    # Stats
    stats = store.get_stats()
    print(f"\n📊 Stats: {stats}")
    
    # Cleanup
    store.delete_collection()
