"""
LLM Manager - Gestion Centralisée des Appels LLM
================================================

Manager unifié pour simplifier l'utilisation des LLMs.
"""

import logging
from typing import List, Optional, Dict, Any, Union

from .base import (
    BaseLLMProvider,
    LLMMessage,
    LLMResponse,
    LLMConfig,
    LLMProvider
)
from .providers import (
    OllamaProvider,
    OpenRouterProvider,
    LlamaCppProvider
)
from .direct_provider import (
    DirectLlamaCppProvider
)

logger = logging.getLogger(__name__)


class LLMManager:
    """
    Manager centralisé pour gérer les appels LLM.
    
    Simplifie l'utilisation en fournissant des méthodes helper et
    la gestion de plusieurs providers.
    
    Usage:
        # Instancier avec un provider
        manager = LLMManager.from_ollama("mistral:latest")
        
        # Appel simple
        response = manager.generate("Quelle est la capitale de la France?")
        print(response.content)
        
        # Appel avec conversation
        messages = [
            {"role": "system", "content": "Tu es un assistant."},
            {"role": "user", "content": "Bonjour!"}
        ]
        response = manager.generate_from_messages(messages)
    """
    
    def __init__(self, provider: BaseLLMProvider):
        """
        Initialiser le manager avec un provider.
        
        Args:
            provider: Provider LLM (Ollama, OpenRouter, etc.)
        """
        self.provider = provider
        logger.info(f"LLMManager initialized with {provider.__class__.__name__}")
    
    @classmethod
    def from_ollama(
        cls,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        config: Optional[LLMConfig] = None
    ) -> "LLMManager":
        """
        Créer un manager avec Ollama local.
        
        Args:
            model: Nom du modèle (ex: "mistral:latest")
            base_url: URL du serveur Ollama
            config: Configuration optionnelle
            
        Returns:
            LLMManager configuré avec Ollama
        """
        provider = OllamaProvider(model, base_url, config)
        return cls(provider)
    
    @classmethod
    def from_openrouter(
        cls,
        model: str,
        api_key: Optional[str] = None,
        config: Optional[LLMConfig] = None
    ) -> "LLMManager":
        """
        Créer un manager avec OpenRouter.
        
        Args:
            model: Nom du modèle OpenRouter
            api_key: Clé API (ou None pour utiliser env var)
            config: Configuration optionnelle
            
        Returns:
            LLMManager configuré avec OpenRouter
        """
        provider = OpenRouterProvider(model, api_key, config)
        return cls(provider)
    
    @classmethod
    def from_llamacpp(
        cls,
        model: str,
        base_url: str = "http://localhost:8080/v1",
        config: Optional[LLMConfig] = None
    ) -> "LLMManager":
        """
        Créer un manager avec llama.cpp.
        
        Args:
            model: Nom du modèle
            base_url: URL du serveur llama.cpp
            config: Configuration optionnelle
            
        Returns:
            LLMManager configuré avec llama.cpp
        """
        provider = LlamaCppProvider(model, base_url, config)
        return cls(provider)
    
    @classmethod
    def from_direct_llamacpp(
        cls,
        model_path: str,
        n_gpu_layers: int = -1,
        n_ctx: int = 4096,
        config: Optional[LLMConfig] = None,
        verbose: bool = False
    ) -> "LLMManager":
        """
        Créer un manager avec chargement direct du modèle GGUF (sans serveur HTTP).
        
        Args:
            model_path: Chemin vers le fichier GGUF
            n_gpu_layers: Nombre de layers sur GPU (-1 = tous)
            n_ctx: Taille du contexte
            config: Configuration optionnelle
            verbose: Mode verbose
            
        Returns:
            LLMManager configuré avec DirectLlamaCppProvider
        """
        provider = DirectLlamaCppProvider(model_path, n_gpu_layers, n_ctx, config, verbose)
        return cls(provider)
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """
        Générer une réponse à partir d'un prompt simple.
        
        Args:
            prompt: Prompt utilisateur
            system_prompt: Instruction système optionnelle
            config: Configuration optionnelle
            
        Returns:
            LLMResponse avec la réponse
        """
        messages = []
        if system_prompt:
            messages.append(LLMMessage(role="system", content=system_prompt))
        messages.append(LLMMessage(role="user", content=prompt))
        
        return self.provider.generate(messages, config)
    
    async def agenerate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """
        Générer une réponse en mode asynchrone.
        
        Args:
            prompt: Prompt utilisateur
            system_prompt: Instruction système optionnelle
            config: Configuration optionnelle
            
        Returns:
            LLMResponse avec la réponse
        """
        messages = []
        if system_prompt:
            messages.append(LLMMessage(role="system", content=system_prompt))
        messages.append(LLMMessage(role="user", content=prompt))
        
        return await self.provider.agenerate(messages, config)
    
    def generate_from_messages(
        self,
        messages: Union[List[Dict[str, str]], List[LLMMessage]],
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """
        Générer à partir d'une liste de messages.
        
        Args:
            messages: Liste de messages (dict ou LLMMessage)
            config: Configuration optionnelle
            reasoning: Reasoning parameters (e.g., {"effort": "low"}). Defaults to {"effort": "low"}
            
        Returns:
            LLMResponse avec la réponse
        """
        # Set default reasoning
        if not config.reasoning:
            config.reasoning = "low"

        # Convertir dict en LLMMessage si nécessaire
        if messages and isinstance(messages[0], dict):
            messages = [
                LLMMessage(role=m["role"], content=m["content"])
                for m in messages
            ]
        
        return self.provider.generate(messages, config)
    
    async def agenerate_from_messages(
        self,
        messages: Union[List[Dict[str, str]], List[LLMMessage]],
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """
        Générer en async à partir d'une liste de messages.
        
        Args:
            messages: Liste de messages (dict ou LLMMessage)
            config: Configuration optionnelle
            reasoning: Reasoning parameters (e.g., "none, "low", "high")
            
        Returns:
            LLMResponse avec la réponse
        """
        # Set default reasoning
        if not config.reasoning:
            config.reasoning = "low"
            
        # Convertir dict en LLMMessage si nécessaire
        if messages and isinstance(messages[0], dict):
            messages = [
                LLMMessage(role=m["role"], content=m["content"])
                for m in messages
            ]
                
        return await self.provider.agenerate(messages, config)
    
    def count_tokens(self, text: str) -> int:
        """
        Compter les tokens dans un texte.
        
        Args:
            text: Texte à compter
            
        Returns:
            Nombre de tokens
        """
        return self.provider.count_tokens(text)
    
    def get_info(self) -> Dict[str, Any]:
        """
        Obtenir les infos du provider actuel.
        
        Returns:
            Dict avec les infos
        """
        return self.provider.get_model_info()


# Helpers pour créer rapidement un manager
def create_ollama_manager(
    model: str = "mistral:latest",
    **kwargs
) -> LLMManager:
    """Helper rapide pour Ollama"""
    return LLMManager.from_ollama(model, **kwargs)


def create_openrouter_manager(
    model: str = "mistralai/mistral-small-3.1-24b-instruct:free",
    **kwargs
) -> LLMManager:
    """Helper rapide pour OpenRouter"""
    return LLMManager.from_openrouter(model, **kwargs)


def create_llamacpp_manager(
    model: str = "deepseek-r1-distill-qwen-32b",
    **kwargs
) -> LLMManager:
    """Helper rapide pour llama.cpp"""
    return LLMManager.from_llamacpp(model, **kwargs)
