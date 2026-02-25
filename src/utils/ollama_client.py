"""
Ollama Local Client
===================

Client pour utiliser des modèles locaux via Ollama.
Compatible avec l'API OpenAI pour faciliter l'intégration.

Modèles configurés:
- Generator (Q&A): mistral:latest (Mistral 7B Instruct, ~4.5GB)
- Critic: phi3:mini (Phi-3 Mini 3.8B, ~2.3GB)

Total VRAM: ~6.8GB / 7GB disponibles ✅
"""

from openai import OpenAI


def create_ollama_client() -> OpenAI:
    """
    Crée un client Ollama compatible avec l'API OpenAI.
    
    Returns:
        Client OpenAI configuré pour Ollama local
        
    Usage:
        client = create_ollama_client()
        response = client.chat.completions.create(
            model="mistral:latest",
            messages=[{"role": "user", "content": "Hello"}]
        )
    """
    # Ollama utilise une API compatible OpenAI
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # Ollama ne nécessite pas de vraie clé API
    )
    
    return client


# Configuration des modèles locaux
OLLAMA_MODELS = {
    # Generator: Mistral 7B Instruct (bon pour génération créative)
    "generator": "mistral:latest",
    
    # Critic: Llama 3 8B (STRICT! Plus puissant que Phi-3 Mini)
    "critic": "llama3:8b",
    
    # Alternatives disponibles:
    "mistral_7b": "mistral:latest",
    "phi3_mini": "phi3:mini",
    "llama3_8b": "llama3:8b",
}


def get_model_info(model_id: str) -> dict:
    """
    Obtenir les informations sur un modèle local.
    
    Args:
        model_id: ID du modèle Ollama
        
    Returns:
        Dict avec les infos du modèle
    """
    info = {
        "mistral:latest": {
            "name": "Mistral 7B Instruct",
            "size_gb": 4.4,
            "context_length": 8192,
            "description": "Mistral AI's 7B parameter model, excellent for generation",
            "use_case": "Question/Answer generation"
        },
        "phi3:mini": {
            "name": "Phi-3 Mini 3.8B",
            "size_gb": 2.3,
            "context_length": 4096,
            "description": "Microsoft's compact model, great for classification/evaluation",
            "use_case": "Quality evaluation and critique"
        },
        "llama3:8b": {
            "name": "Llama 3 8B",
            "size_gb": 4.7,
            "context_length": 8192,
            "description": "Meta's powerful model, excellent for strict evaluation",
            "use_case": "Strict quality evaluation and critique"
        }
    }
    
    return info.get(model_id, {"name": model_id, "size_gb": 0, "context_length": 4096})


def calculate_vram_usage(generator_model: str, critic_model: str) -> dict:
    """
    Calcule l'utilisation VRAM estimée pour les deux modèles.
    
    Args:
        generator_model: Modèle du generator
        critic_model: Modèle du critic
        
    Returns:
        Dict avec estimation VRAM par modèle et total
    """
    generator_info = get_model_info(generator_model)
    critic_info = get_model_info(critic_model)
    
    return {
        "generator_gb": generator_info["size_gb"],
        "critic_gb": critic_info["size_gb"],
        "total_gb": generator_info["size_gb"] + critic_info["size_gb"]
    }


def check_ollama_status() -> bool:
    """
    Vérifie si Ollama est accessible.
    
    Returns:
        True si Ollama répond, False sinon
    """
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


if __name__ == "__main__":
    print("Ollama Local Client Configuration")
    print("=" * 60)
    print()
    
    # Check Ollama status
    print("🔍 Vérification d'Ollama...")
    if check_ollama_status():
        print("✅ Ollama est actif sur http://localhost:11434")
    else:
        print("❌ Ollama n'est pas accessible")
        print("   Démarrez Ollama avec: ollama serve")
        exit(1)
    
    print()
    print("Modèles configurés:")
    for key, model_id in OLLAMA_MODELS.items():
        if key in ["generator", "critic"]:
            info = get_model_info(model_id)
            print(f"  {key:10s}: {model_id}")
            print(f"              → {info['name']} ({info['size_gb']}GB)")
    
    print()
    total_vram = get_model_info(OLLAMA_MODELS["generator"])["size_gb"] + \
                 get_model_info(OLLAMA_MODELS["critic"])["size_gb"]
    print(f"Total VRAM estimé: {total_vram:.1f}GB")
    print(f"VRAM disponible: 7GB")
    
    if total_vram <= 7:
        print("✅ Configuration viable pour ton laptop!")
    else:
        print("⚠️  Risque de dépassement mémoire")
    
    print()
    
    # Test de connexion
    try:
        client = create_ollama_client()
        print("✅ Client Ollama créé avec succès!")
        print(f"   Base URL: {client.base_url}")
    except Exception as e:
        print(f"❌ Erreur: {e}")
