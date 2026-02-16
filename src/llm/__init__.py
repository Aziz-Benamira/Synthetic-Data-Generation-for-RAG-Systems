"""
LLM Package - Gestion Centralisée des Appels LLM
=================================================

Module centralisé pour gérer tous les appels aux LLMs de manière unifiée.

Usage Simple:
    from src.llm import LLMManager
    
    # Ollama local
    manager = LLMManager.from_ollama("mistral:latest")
    response = manager.generate("Quelle est la capitale de la France?")
    print(response.content)
    
    # OpenRouter
    manager = LLMManager.from_openrouter("mistralai/mistral-small-3.1-24b-instruct:free")
    response = manager.generate("Hello!")

Usage Avancé:
    from src.llm import LLMManager, LLMMessage, LLMConfig
    
    # Configuration personnalisée
    config = LLMConfig(temperature=0.3, max_tokens=500)
    manager = LLMManager.from_ollama("mistral:latest", config=config)
    
    # Conversation multi-turn
    messages = [
        LLMMessage(role="system", content="Tu es un expert en probabilités."),
        LLMMessage(role="user", content="Explique la loi normale.")
    ]
    response = manager.generate_from_messages(messages)
"""

from .base import (
    BaseLLMProvider,
    LLMMessage,
    LLMResponse,
    LLMConfig,
    LLMProvider,
    messages_to_dict,
    dict_to_messages
)

from .providers import (
    OllamaProvider,
    OpenRouterProvider,
    LlamaCppProvider
)

from .direct_provider import (
    DirectLlamaCppProvider
)

from .manager import (
    LLMManager,
    create_ollama_manager,
    create_openrouter_manager,
    create_llamacpp_manager
)

__all__ = [
    # Base classes
    "BaseLLMProvider",
    "LLMMessage",
    "LLMResponse",
    "LLMConfig",
    "LLMProvider",
    "messages_to_dict",
    "dict_to_messages",
    
    # Providers
    "OllamaProvider",
    "OpenRouterProvider",
    "LlamaCppProvider",
    "DirectLlamaCppProvider",
    
    # Manager
    "LLMManager",
    "create_ollama_manager",
    "create_openrouter_manager",
    "create_llamacpp_manager",
]
