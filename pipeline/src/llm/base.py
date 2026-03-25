"""
LLM Base Classes - Interface Abstraite
=======================================

Classes abstraites pour définir l'interface des providers LLM.
Permet d'avoir une API unifiée quel que soit le provider (Ollama, OpenRouter, etc.)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum


class LLMProvider(Enum):
    """Types de providers LLM supportés"""
    OLLAMA = "ollama"           # Local via Ollama
    OPENROUTER = "openrouter"   # OpenRouter API
    OPENAI = "openai"           # OpenAI API directe
    LLAMACPP = "llamacpp"       # llama.cpp local (GGUF)


@dataclass
class LLMMessage:
    """Message pour conversation avec LLM"""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    """
    Réponse standardisée d'un LLM.
    
    Attributes:
        content: Contenu de la réponse
        model: Modèle utilisé
        tokens_used: Nombre de tokens utilisés (prompt + completion)
        finish_reason: Raison de fin ("stop", "length", etc.)
        raw_response: Réponse brute du provider (pour debug)
    """
    content: str
    model: str
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None
    raw_response: Optional[Any] = None


@dataclass
class LLMConfig:
    """
    Configuration pour un appel LLM.
    
    Attributes:
        temperature: Créativité (0.0 = déterministe, 1.0 = créatif)
        max_tokens: Nombre max de tokens de sortie
        top_p: Nucleus sampling
        frequency_penalty: Pénalité de répétition
        presence_penalty: Pénalité de présence
        stop_sequences: Séquences d'arrêt
    """
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None


class BaseLLMProvider(ABC):
    """
    Interface abstraite pour tous les providers LLM.
    
    Chaque provider (Ollama, OpenRouter, etc.) doit implémenter cette interface.
    """
    
    def __init__(self, model: str, config: Optional[LLMConfig] = None):
        """
        Initialiser le provider.
        
        Args:
            model: Nom du modèle à utiliser
            config: Configuration optionnelle
        """
        self.model = model
        self.config = config or LLMConfig()
    
    @abstractmethod
    def generate(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """
        Générer une réponse en mode synchrone.
        
        Args:
            messages: Liste des messages de conversation
            config: Config optionnelle (override celle du provider)
            
        Returns:
            LLMResponse avec la réponse du modèle
        """
        pass
    
    @abstractmethod
    async def agenerate(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """
        Générer une réponse en mode asynchrone.
        
        Args:
            messages: Liste des messages de conversation
            config: Config optionnelle
            
        Returns:
            LLMResponse avec la réponse du modèle
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Compter le nombre de tokens dans un texte.
        
        Args:
            text: Texte à compter
            
        Returns:
            Nombre de tokens
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtenir les informations sur le modèle.
        
        Returns:
            Dict avec infos du modèle
        """
        return {
            "model": self.model,
            "provider": self.__class__.__name__,
            "config": self.config.__dict__
        }


# Helpers pour conversion
def messages_to_dict(messages: List[LLMMessage]) -> List[Dict[str, str]]:
    """Convertir LLMMessage en dict pour APIs"""
    return [{"role": msg.role, "content": msg.content} for msg in messages]


def dict_to_messages(messages: List[Dict[str, str]]) -> List[LLMMessage]:
    """Convertir dict en LLMMessage"""
    return [LLMMessage(role=msg["role"], content=msg["content"]) for msg in messages]
