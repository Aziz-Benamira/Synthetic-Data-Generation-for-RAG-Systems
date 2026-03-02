"""
RAG Retriever — Semantic Chunks + ChromaDB
==========================================

Charge nos chunks sémantiques (issus du SemanticChunker) dans ChromaDB
et expose une interface de retrieval simple pour l'évaluation du RAG.

Compatible avec notre format chunks_mi201.json :
  {"metadata": {...}, "chunks": [{"chunk_id", "content", "semantic_type", ...}]}
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """
    Retriever utilisant nos chunks sémantiques + ChromaDB + SentenceTransformer.
    
    Contrairement au RAG classique (RecursiveCharacterTextSplitter 512 chars),
    nos chunks respectent les frontières sémantiques (définitions, théorèmes,
    exemples) avec des métadonnées riches (chapitre, section, type sémantique).
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "semantic_chunks",
        persist_directory: Optional[str] = None,
        device: str = "cpu"
    ):
        """
        Args:
            embedding_model: Nom du modèle SentenceTransformer
            collection_name: Nom de la collection ChromaDB
            persist_directory: Répertoire de persistance (None = in-memory)
            device: Device pour l'embedding ('cpu' ou 'cuda')
        """
        logger.info(f"Initializing SemanticRetriever...")
        logger.info(f"  Embedding model: {embedding_model}")
        logger.info(f"  Device: {device}")
        
        # Embedding model
        self.embedder = SentenceTransformer(embedding_model, device=device)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        
        # ChromaDB
        if persist_directory:
            self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.chroma_client = chromadb.Client()
        
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Cache des chunks pour lookup rapide
        self._chunks_by_id: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"  ChromaDB collection: {collection_name}")
        logger.info(f"  Embedding dim: {self.embedding_dim}")
    
    def load_chunks(self, chunks_path: str) -> int:
        """
        Charge les chunks sémantiques depuis notre fichier JSON dans ChromaDB.
        
        Args:
            chunks_path: Chemin vers chunks_mi201.json
        
        Returns:
            Nombre de chunks indexés
        """
        logger.info(f"Loading chunks from {chunks_path}...")
        
        with open(chunks_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks = data["chunks"]
        metadata_doc = data.get("metadata", {})
        
        logger.info(f"  Source: {metadata_doc.get('source_pdf', 'unknown')}")
        logger.info(f"  Total chunks: {len(chunks)}")
        
        # Préparer les données pour ChromaDB
        ids = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            content = chunk["content"]
            
            # Stocker dans le cache
            self._chunks_by_id[chunk_id] = chunk
            
            # Construire le contexte enrichi pour l'embedding
            # On préfixe avec les métadonnées pour améliorer la recherche
            enriched_content = self._build_enriched_content(chunk)
            
            ids.append(chunk_id)
            documents.append(enriched_content)
            metadatas.append({
                "chunk_id": chunk_id,
                "chapter": chunk.get("chapter", ""),
                "section": chunk.get("section", ""),
                "subsection": chunk.get("subsection", ""),
                "semantic_type": chunk.get("semantic_type", "text"),
                "page_start": chunk.get("page_range", [0, 0])[0],
                "page_end": chunk.get("page_range", [0, 0])[1],
                "length": chunk.get("length", len(content)),
                "source_pdf": metadata_doc.get("source_pdf", "")
            })
        
        # Calculer les embeddings
        logger.info("  Computing embeddings...")
        embeddings = self.embedder.encode(documents, show_progress_bar=True).tolist()
        
        # Indexer dans ChromaDB
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        logger.info(f"  ✅ {len(ids)} chunks indexed in ChromaDB")
        return len(ids)
    
    def _build_enriched_content(self, chunk: Dict[str, Any]) -> str:
        """
        Construit un contenu enrichi pour l'embedding en ajoutant le contexte
        structurel. Cela améliore la pertinence du retrieval.
        """
        parts = []
        
        # Ajouter le contexte hiérarchique
        if chunk.get("chapter"):
            parts.append(f"[Chapitre: {chunk['chapter']}]")
        if chunk.get("section"):
            parts.append(f"[Section: {chunk['section']}]")
        if chunk.get("subsection"):
            parts.append(f"[Sous-section: {chunk['subsection']}]")
        if chunk.get("semantic_type") and chunk["semantic_type"] != "text":
            parts.append(f"[Type: {chunk['semantic_type']}]")
        
        # Contenu principal
        parts.append(chunk["content"])
        
        return "\n".join(parts)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_chapter: Optional[str] = None,
        filter_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Recherche les chunks les plus pertinents pour une question.
        
        Args:
            query: Question de l'utilisateur
            top_k: Nombre de chunks à retourner
            filter_chapter: Filtrer par chapitre (optionnel)
            filter_type: Filtrer par type sémantique (optionnel)
        
        Returns:
            Liste de dicts avec chunk, score, metadata
        """
        # Construire le filtre
        where_filter = None
        if filter_chapter or filter_type:
            conditions = {}
            if filter_chapter:
                conditions["chapter"] = filter_chapter
            if filter_type:
                conditions["semantic_type"] = filter_type
            where_filter = conditions
        
        # Embedding de la query
        query_embedding = self.embedder.encode([query]).tolist()
        
        # Recherche
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # Formatter les résultats
        retrieved = []
        for i in range(len(results["ids"][0])):
            chunk_id = results["ids"][0][i]
            distance = results["distances"][0][i]
            similarity = 1 - distance  # cosine distance → similarity
            metadata = results["metadatas"][0][i]
            
            # Récupérer le contenu original (sans enrichissement)
            original_content = self._chunks_by_id.get(chunk_id, {}).get("content", "")
            
            retrieved.append({
                "chunk_id": chunk_id,
                "content": original_content,
                "similarity": round(similarity, 4),
                "metadata": metadata
            })
        
        return retrieved
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Récupère un chunk par son ID depuis le cache."""
        return self._chunks_by_id.get(chunk_id)
    
    @property
    def num_chunks(self) -> int:
        """Nombre de chunks indexés."""
        return self.collection.count()
