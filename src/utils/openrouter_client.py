"""
OpenRouter Client Wrapper
==========================

Wrapper pour utiliser OpenRouter avec une API compatible OpenAI.
OpenRouter permet d'accéder à plusieurs modèles via une seule API.

Models utilisés:
- Generator (Q/A): mistralai/mistral-small-3.1-24b-instruct:free
- Critic: meta-llama/llama-3.3-70b-instruct:free

Documentation: https://openrouter.ai/docs
"""

import os
from openai import OpenAI


def create_openrouter_client(api_key: str = None) -> OpenAI:
    """
    Crée un client OpenRouter compatible avec l'API OpenAI.
    
    Args:
        api_key: Clé API OpenRouter (si None, utilise OPENROUTER_API_KEY env var)
        
    Returns:
        Client OpenAI configuré pour OpenRouter
        
    Usage:
        client = create_openrouter_client()
        response = client.chat.completions.create(
            model="mistralai/mistral-small-3.1-24b-instruct:free",
            messages=[{"role": "user", "content": "Hello"}]
        )
    """
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        raise ValueError(
            "OpenRouter API key not found. "
            "Set OPENROUTER_API_KEY environment variable or pass api_key parameter."
        )
    
    # OpenRouter utilise une API compatible OpenAI
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://github.com/yourusername/agentic-ai",  # Optionnel
            "X-Title": "Agentic RAG Dataset Generator",  # Optionnel
        }
    )
    
    return client


# Modèles disponibles (gratuits)
OPENROUTER_MODELS = {
    # Generator: Mistral Small 3.1 24B (bon pour la génération créative)
    "generator": "mistralai/mistral-small-3.1-24b-instruct:free",
    
    # Critic: Llama 3.3 70B (puissant pour l'évaluation)
    "critic": "meta-llama/llama-3.3-70b-instruct:free",
    
    # Alternatives gratuites disponibles:
    "mistral_7b": "mistralai/mistral-7b-instruct:free",
    "llama_8b": "meta-llama/llama-3.1-8b-instruct:free",
    "qwen_32b": "qwen/qwen-2.5-32b-instruct:free",
}


def get_model_info(model_id: str) -> dict:
    """
    Obtenir les informations sur un modèle.
    
    Args:
        model_id: ID du modèle (ex: "mistralai/mistral-small-3.1-24b-instruct:free")
        
    Returns:
        Dict avec les infos du modèle
    """
    info = {
        "mistralai/mistral-small-3.1-24b-instruct:free": {
            "name": "Mistral Small 3.1 24B",
            "context_length": 32000,
            "description": "Mistral AI's 24B parameter model, balanced for quality and speed",
            "use_case": "Question/Answer generation"
        },
        "meta-llama/llama-3.3-70b-instruct:free": {
            "name": "Llama 3.3 70B Instruct",
            "context_length": 131072,
            "description": "Meta's powerful 70B model, excellent for evaluation tasks",
            "use_case": "Quality evaluation and critique"
        }
    }
    
    return info.get(model_id, {"name": model_id, "context_length": 8192})


if __name__ == "__main__":
    print("OpenRouter Client Configuration")
    print("=" * 50)
    print()
    print("Modèles configurés:")
    print(f"  Generator: {OPENROUTER_MODELS['generator']}")
    print(f"  Critic:    {OPENROUTER_MODELS['critic']}")
    print()
    
    # Test de connexion
    try:
        client = create_openrouter_client()
        print("✅ Client OpenRouter créé avec succès!")
        print(f"   Base URL: {client.base_url}")
    except Exception as e:
        print(f"❌ Erreur: {e}")
