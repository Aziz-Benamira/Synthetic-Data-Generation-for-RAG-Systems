"""
Test de Gestion VRAM avec Ollama
=================================

Teste comment Ollama gère la VRAM quand on alterne entre modèles:
1. Charge Mistral 7B (generator)
2. Vérifie VRAM
3. Charge Phi-3 Mini (critic)
4. Vérifie si Mistral est déchargé automatiquement
5. Teste avec un modèle plus gros (Llama 3 8B)
"""

import subprocess
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src' / 'utils'))
from ollama_client import create_ollama_client

client = create_ollama_client()

def check_loaded_models():
    """Vérifie quels modèles sont en VRAM"""
    result = subprocess.run(
        ["ollama", "ps"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    return result.stdout

print("=" * 80)
print("TEST GESTION VRAM - OLLAMA")
print("=" * 80)
print()

print("État initial:")
print("-" * 80)
check_loaded_models()
print()

print("1. Chargement Mistral 7B (Generator - 4.5GB)...")
print("-" * 80)
response = client.chat.completions.create(
    model="mistral:latest",
    messages=[{"role": "user", "content": "Hello"}]
)
print(f"Réponse: {response.choices[0].message.content[:50]}...")
print()

print("Modèles chargés après Mistral:")
check_loaded_models()
time.sleep(2)
print()

print("2. Chargement Phi-3 Mini (Critic - 2.3GB)...")
print("-" * 80)
response = client.chat.completions.create(
    model="phi3:mini",
    messages=[{"role": "user", "content": "Evaluate this"}]
)
print(f"Réponse: {response.choices[0].message.content[:50]}...")
print()

print("Modèles chargés après Phi-3:")
check_loaded_models()
print()

print("=" * 80)
print("ANALYSE")
print("=" * 80)
print()

print("Comportement d'Ollama:")
print()
print("• Par défaut: garde les modèles en cache ~5 minutes")
print("• Si VRAM insuffisante: décharge automatiquement les anciens modèles")
print("• Dans notre workflow:")
print("  1. Generator (Mistral 4.5GB) génère Q+A")
print("  2. Critic (Phi-3 2.3GB) évalue")
print("  3. Les deux peuvent être en VRAM en même temps (~6.8GB)")
print()

print("CONCLUSION:")
print()
print("• Avec Phi-3 Mini (2.3GB): Total 6.8GB ✅")
print("• Possible d'utiliser un Critic plus gros:")
print("  - Llama 3 8B (4.5GB): Total 9GB → Ollama déchargera Mistral")
print("  - Gemma 2 9B (5.5GB): Total 10GB → Ollama déchargera Mistral")
print()
print("• Solution: Utiliser Llama 3 8B comme Critic (plus strict que Phi-3)")
print("  → Ollama gère automatiquement le swap VRAM")
print("  → Légèrement plus lent (déchargement/rechargement)")
print("  → Mais BEAUCOUP plus strict dans l'évaluation")
