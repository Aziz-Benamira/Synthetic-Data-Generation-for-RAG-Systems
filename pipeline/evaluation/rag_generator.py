"""
RAG Generator — LLM-based Answer Generation
=============================================

Génère des réponses à partir des chunks récupérés en utilisant un LLM local
(Qwen2.5-32B-Instruct ou DeepSeek-R1-32B) via llama-cpp-python.

Le prompt RAG est structuré pour exploiter les métadonnées riches de nos
chunks sémantiques (chapitre, section, type de contenu).
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Prompt template pour le RAG
# ──────────────────────────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """Tu es un assistant pédagogique spécialisé en Machine Learning.
Tu réponds UNIQUEMENT à partir des extraits de cours fournis, sans utiliser de connaissances externes.

Règles strictes :
- Réponds en 2 à 5 phrases maximum (sauf si une formule mathématique nécessite plus d'espace).
- Base ta réponse EXCLUSIVEMENT sur les extraits fournis. N'ajoute rien qui n'y figure pas.
- Si les extraits ne contiennent pas l'information, réponds : "L'information n'est pas dans les extraits fournis."
- Inclus les formules mathématiques clés si elles apparaissent dans les extraits.
- Ne reformule pas excessivement : reste proche du vocabulaire des extraits.
- Réponds en français."""

RAG_USER_TEMPLATE = """Extraits de cours :

{context}

---

Question : {question}

Réponds de manière concise (2–5 phrases) en te basant strictement sur les extraits ci-dessus."""


def format_context_from_chunks(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Formate les chunks récupérés en un contexte structuré pour le prompt LLM.
    
    Exploite les métadonnées riches : chapitre, section, type sémantique.
    """
    context_parts = []
    
    for i, chunk in enumerate(retrieved_chunks, 1):
        metadata = chunk.get("metadata", {})
        header_parts = []
        
        if metadata.get("chapter"):
            header_parts.append(f"Chapitre: {metadata['chapter']}")
        if metadata.get("section"):
            header_parts.append(f"Section: {metadata['section']}")
        if metadata.get("semantic_type") and metadata["semantic_type"] != "text":
            header_parts.append(f"Type: {metadata['semantic_type']}")
        
        header = " | ".join(header_parts) if header_parts else f"Extrait {i}"
        similarity = chunk.get("similarity", 0)
        
        context_parts.append(
            f"=== Extrait {i} [{header}] (pertinence: {similarity:.2f}) ===\n"
            f"{chunk['content']}"
        )
    
    return "\n\n".join(context_parts)


class RAGGenerator:
    """
    Générateur de réponses RAG utilisant un LLM local via llama-cpp-python.
    
    Supporte :
    - Qwen2.5-32B-Instruct (Q4_K_M) — réponses directes, pas de <think>
    - DeepSeek-R1-Distill-Qwen-32B (IQ3_M) — raisonnement avec <think>
    """
    
    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = -1,
        n_ctx: int = 4096,
        verbose: bool = False
    ):
        """
        Args:
            model_path: Chemin vers le fichier GGUF
            n_gpu_layers: Layers GPU (-1 = tous)
            n_ctx: Taille du contexte
            verbose: Mode verbose llama.cpp
        """
        from llama_cpp import Llama
        
        logger.info(f"Loading RAG LLM: {model_path}")
        logger.info(f"  GPU layers: {n_gpu_layers}, Context: {n_ctx}")
        
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=verbose,
            chat_format="chatml"
        )
        
        self.model_path = model_path
        self._is_deepseek = "deepseek" in model_path.lower()
        
        logger.info(f"  ✅ RAG LLM loaded ({('DeepSeek' if self._is_deepseek else 'Qwen')} mode)")
    
    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        temperature: float = 0.3,
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """
        Génère une réponse RAG à partir de la question et des chunks récupérés.
        
        Args:
            question: Question de l'utilisateur
            retrieved_chunks: Chunks retournés par le retriever
            temperature: Température LLM
            max_tokens: Tokens max pour la réponse
        
        Returns:
            {
                "answer": str,           # Réponse générée
                "context_used": str,      # Contexte formaté envoyé au LLM
                "chunks_used": int,       # Nombre de chunks utilisés
                "tokens_used": int,       # Tokens consommés
                "model": str              # Modèle utilisé
            }
        """
        # Formater le contexte
        context = format_context_from_chunks(retrieved_chunks)
        
        # Construire les messages
        user_message = RAG_USER_TEMPLATE.format(
            context=context,
            question=question
        )
        
        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        
        # Générer
        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9
        )
        
        raw_answer = response["choices"][0]["message"]["content"]
        tokens_used = response.get("usage", {}).get("total_tokens", 0)
        
        # Si DeepSeek, extraire la réponse après </think>
        answer = self._clean_answer(raw_answer)
        
        return {
            "answer": answer,
            "raw_answer": raw_answer,
            "context_used": context,
            "chunks_used": len(retrieved_chunks),
            "tokens_used": tokens_used,
            "model": self.model_path.split("/")[-1]
        }
    
    def _clean_answer(self, raw: str) -> str:
        """Nettoie la réponse (supprime <think>...</think> pour DeepSeek)."""
        if self._is_deepseek and "<think>" in raw:
            # Extraire après </think>
            parts = raw.split("</think>")
            if len(parts) > 1:
                return parts[-1].strip()
        return raw.strip()
