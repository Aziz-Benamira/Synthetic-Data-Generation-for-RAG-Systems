"""
Configuration avec Llama 3 8B comme Critic Strict
==================================================

Remplace Phi-3 Mini par Llama 3 8B pour avoir un Critic plus sévère.

Avantages:
- Llama 3 8B (4.5GB) est BEAUCOUP plus puissant que Phi-3 Mini (2.3GB)
- Plus strict dans l'évaluation des 5 critères
- Ollama décharge automatiquement Mistral quand Llama 3 charge
- Pas de problème VRAM (un seul modèle à la fois)

Inconvénients:
- Légèrement plus lent (swap VRAM entre modèles)
- Téléchargement: 4.7GB
"""

# Nouvelle configuration des modèles
OLLAMA_MODELS_STRICT = {
    "generator": "mistral:latest",  # Mistral 7B (4.5GB)
    "critic": "llama3:8b",           # Llama 3 8B (4.5GB) - STRICT!
}

# Comparaison des Critic models:
CRITIC_COMPARISON = {
    "phi3:mini": {
        "size_gb": 2.3,
        "parameters": "3.8B",
        "strictness": "Laxiste (83% scores parfaits)",
        "vram_total": 6.8,  # Avec Mistral
        "recommendation": "❌ Trop laxiste"
    },
    "llama3:8b": {
        "size_gb": 4.5,
        "parameters": "8B",
        "strictness": "Strict (meilleure discrimination)",
        "vram_total": 4.5,  # Un seul à la fois (swap auto)
        "recommendation": "✅ RECOMMANDÉ"
    },
    "gemma2:9b": {
        "size_gb": 5.5,
        "parameters": "9B",
        "strictness": "Très strict",
        "vram_total": 5.5,  # Un seul à la fois
        "recommendation": "⚠️ Bon mais plus lent"
    }
}

print("=" * 80)
print("CONFIGURATION CRITIC STRICT")
print("=" * 80)
print()

print("Modèle recommandé: Llama 3 8B")
print()

for model, info in CRITIC_COMPARISON.items():
    print(f"{model}:")
    print(f"  Taille: {info['size_gb']}GB ({info['parameters']} paramètres)")
    print(f"  Strictness: {info['strictness']}")
    print(f"  VRAM total: {info['vram_total']}GB")
    print(f"  {info['recommendation']}")
    print()

print("=" * 80)
print("IMPACT SUR LE WORKFLOW")
print("=" * 80)
print()

print("Workflow avec Phi-3 Mini (actuel):")
print("  1. Mistral génère Q+A (4.5GB en VRAM)")
print("  2. Phi-3 évalue (2.3GB, les deux en VRAM = 6.8GB)")
print("  3. Résultat: 83% scores parfaits → PEU de retries")
print()

print("Workflow avec Llama 3 8B (proposé):")
print("  1. Mistral génère Q+A (4.5GB en VRAM)")
print("  2. Ollama décharge Mistral")
print("  3. Llama 3 évalue (4.5GB en VRAM)")
print("  4. Résultat attendu: 30-40% scores parfaits → PLUS de retries")
print()

print("=" * 80)
print("PROCHAINES ÉTAPES")
print("=" * 80)
print()
print("1. Attendre téléchargement llama3:8b (~16 min)")
print("2. Modifier ollama_client.py:")
print('   OLLAMA_MODELS["critic"] = "llama3:8b"')
print("3. Modifier critic_agent.py:")
print('   model_name: str = "llama3:8b"')
print("4. Relancer test_pipeline_local.py")
print("5. Observer: BEAUCOUP plus de retries déclenchés!")
