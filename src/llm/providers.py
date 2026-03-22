"""
LLM Providers - Implémentations Concrètes
==========================================

Implémentations des providers LLM : Ollama, OpenRouter, llama.cpp
"""

import os
import json
import logging
from typing import List, Optional, Dict, Any

# Make OpenAI optional
try:
    from openai import OpenAI, AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None
    AsyncOpenAI = None

from .base import (
    BaseLLMProvider,
    LLMMessage,
    LLMResponse,
    LLMConfig,
    LLMProvider,
    messages_to_dict
)

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """
    Provider pour Ollama local.
    
    Usage:
        provider = OllamaProvider(model="mistral:latest")
        response = provider.generate([
            LLMMessage(role="user", content="Hello!")
        ])
    """
    
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        config: Optional[LLMConfig] = None
    ):
        super().__init__(model, config)
        self.client = OpenAI(
            base_url=base_url,
            api_key="ollama"  # Ollama n'a pas besoin de vraie clé
        )
        self.async_client = AsyncOpenAI(
            base_url=base_url,
            api_key="ollama"
        )
    
    def generate(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """Générer réponse en mode sync"""
        cfg = config or self.config
        print(config.reasoning)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                reasoning_effort = config.reasoning,
                messages=messages_to_dict(messages),
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                top_p=cfg.top_p,
                frequency_penalty=cfg.frequency_penalty,
                presence_penalty=cfg.presence_penalty,
                stop=cfg.stop_sequences
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                tokens_used=response.usage.total_tokens if response.usage else None,
                finish_reason=response.choices[0].finish_reason,
                raw_response=response
            )
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise
    
    async def agenerate(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """Générer réponse en mode async"""
        cfg = config or self.config
        
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                reasoning_effort= config.reasoning,
                messages=messages_to_dict(messages),
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                top_p=cfg.top_p,
                frequency_penalty=cfg.frequency_penalty,
                presence_penalty=cfg.presence_penalty,
                stop=cfg.stop_sequences
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                tokens_used=response.usage.total_tokens if response.usage else None,
                finish_reason=response.choices[0].finish_reason,
                raw_response=response
            )
        except Exception as e:
            logger.error(f"Ollama async generation error: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """Estimation du nombre de tokens (1 token ≈ 4 chars)"""
        return len(text) // 4


class OpenRouterProvider(BaseLLMProvider):
    """
    Provider pour OpenRouter.
    
    Usage:
        provider = OpenRouterProvider(
            model="mistralai/mistral-small-3.1-24b-instruct:free",
            api_key="sk-or-..."
        )
        response = provider.generate([
            LLMMessage(role="user", content="Hello!")
        ])
    """
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        config: Optional[LLMConfig] = None
    ):
        super().__init__(model, config)
        
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. "
                "Set OPENROUTER_API_KEY env var or pass api_key parameter."
            )
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/agentic-ai",
                "X-Title": "Agentic RAG System"
            }
        )
        
        self.async_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/agentic-ai",
                "X-Title": "Agentic RAG System"
            }
        )
    
    def generate(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """Générer réponse en mode sync"""
        cfg = config or self.config
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages_to_dict(messages),
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                top_p=cfg.top_p,
                frequency_penalty=cfg.frequency_penalty,
                presence_penalty=cfg.presence_penalty,
                stop=cfg.stop_sequences
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                tokens_used=response.usage.total_tokens if response.usage else None,
                finish_reason=response.choices[0].finish_reason,
                raw_response=response
            )
        except Exception as e:
            logger.error(f"OpenRouter generation error: {e}")
            raise
    
    async def agenerate(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """Générer réponse en mode async"""
        cfg = config or self.config
        
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages_to_dict(messages),
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                top_p=cfg.top_p,
                frequency_penalty=cfg.frequency_penalty,
                presence_penalty=cfg.presence_penalty,
                stop=cfg.stop_sequences
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                tokens_used=response.usage.total_tokens if response.usage else None,
                finish_reason=response.choices[0].finish_reason,
                raw_response=response
            )
        except Exception as e:
            logger.error(f"OpenRouter async generation error: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """Estimation du nombre de tokens (1 token ≈ 4 chars)"""
        return len(text) // 4


class LlamaCppProvider(BaseLLMProvider):
    """
    Provider pour llama.cpp local (GGUF).
    
    Compatible avec le serveur llama.cpp lancé avec --api-key.
    
    Usage:
        provider = LlamaCppProvider(
            model="deepseek-r1-distill-qwen-32b",
            base_url="http://localhost:8080/v1"
        )
    """
    
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8080/v1",
        config: Optional[LLMConfig] = None
    ):
        super().__init__(model, config)
        self.client = OpenAI(
            base_url=base_url,
            api_key="not-needed"  # llama.cpp peut ne pas nécessiter de clé
        )
        self.async_client = AsyncOpenAI(
            base_url=base_url,
            api_key="not-needed"
        )
    
    def generate(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """Générer réponse en mode sync"""
        cfg = config or self.config
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages_to_dict(messages),
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                top_p=cfg.top_p,
                frequency_penalty=cfg.frequency_penalty,
                presence_penalty=cfg.presence_penalty,
                stop=cfg.stop_sequences
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                tokens_used=response.usage.total_tokens if response.usage else None,
                finish_reason=response.choices[0].finish_reason,
                raw_response=response
            )
        except Exception as e:
            logger.error(f"llama.cpp generation error: {e}")
            raise
    
    async def agenerate(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """Générer réponse en mode async"""
        cfg = config or self.config
        
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages_to_dict(messages),
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                top_p=cfg.top_p,
                frequency_penalty=cfg.frequency_penalty,
                presence_penalty=cfg.presence_penalty,
                stop=cfg.stop_sequences
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                tokens_used=response.usage.total_tokens if response.usage else None,
                finish_reason=response.choices[0].finish_reason,
                raw_response=response
            )
        except Exception as e:
            logger.error(f"llama.cpp async generation error: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """Estimation du nombre de tokens (1 token ≈ 4 chars)"""
        return len(text) // 4
