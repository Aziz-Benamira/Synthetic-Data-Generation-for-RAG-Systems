"""
Direct LlamaCpp Provider - Chargement direct des modèles GGUF
=============================================================

Provider qui charge les modèles GGUF directement en mémoire via llama-cpp-python
sans passer par un serveur HTTP.
"""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from .base import (
    BaseLLMProvider,
    LLMMessage,
    LLMResponse,
    LLMConfig,
    messages_to_dict
)

logger = logging.getLogger(__name__)


class DirectLlamaCppProvider(BaseLLMProvider):
    """
    Provider pour charger directement des modèles GGUF avec llama-cpp-python.
    
    Usage:
        provider = DirectLlamaCppProvider(
            model_path="~/models/qwen2.5-32b/model.gguf",
            n_gpu_layers=-1,  # All layers on GPU
            n_ctx=4096
        )
        response = provider.generate([
            LLMMessage(role="user", content="Hello!")
        ])
    """
    
    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = -1,  # -1 = all layers on GPU
        n_ctx: int = 4096,
        config: Optional[LLMConfig] = None,
        verbose: bool = False
    ):
        """
        Initialize direct llama.cpp provider.
        
        Args:
            model_path: Path to GGUF model file
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
            n_ctx: Context window size
            config: LLM configuration
            verbose: Enable verbose logging
        """
        super().__init__(model_path, config)
        
        # Validate model path
        model_path = Path(model_path).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Import llama-cpp-python
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. "
                "Install with: pip install llama-cpp-python"
            )
        
        # Load model
        logger.info(f"Loading model: {model_path}")
        logger.info(f"  - GPU layers: {n_gpu_layers}")
        logger.info(f"  - Context size: {n_ctx}")
        
        self.llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=verbose,
            chat_format="chatml"  # Default, can be overridden
        )
        
        self.model_path_str = str(model_path)
        logger.info(f"✅ Model loaded: {model_path.name}")
    
    def generate(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """Generate response from messages"""
        cfg = config or self.config
        
        try:
            # Convert messages to dict format
            messages_dict = messages_to_dict(messages)
            
            # Generate with llama.cpp
            response = self.llm.create_chat_completion(
                messages=messages_dict,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                top_p=cfg.top_p,
                frequency_penalty=cfg.frequency_penalty,
                presence_penalty=cfg.presence_penalty,
                stop=cfg.stop_sequences
            )
            
            # Extract content
            content = response['choices'][0]['message']['content']
            finish_reason = response['choices'][0]['finish_reason']
            tokens_used = response.get('usage', {}).get('total_tokens')
            
            return LLMResponse(
                content=content,
                model=self.model_path_str,
                tokens_used=tokens_used,
                finish_reason=finish_reason,
                raw_response=response
            )
            
        except Exception as e:
            logger.error(f"Direct llama.cpp generation error: {e}")
            raise
    
    async def agenerate(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """
        Async generation (currently just wraps sync call).
        llama-cpp-python doesn't have native async support.
        """
        # For now, just call sync version
        # Could be improved with asyncio.to_thread() if needed
        return self.generate(messages, config)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using the model's tokenizer"""
        try:
            tokens = self.llm.tokenize(text.encode('utf-8'))
            return len(tokens)
        except Exception as e:
            logger.warning(f"Token counting failed: {e}, using estimation")
            return len(text) // 4  # Fallback estimation
    
    def __del__(self):
        """Cleanup: free model from memory"""
        if hasattr(self, 'llm'):
            try:
                del self.llm
                logger.info("Model freed from memory")
            except Exception as e:
                logger.warning(f"Error freeing model: {e}")
