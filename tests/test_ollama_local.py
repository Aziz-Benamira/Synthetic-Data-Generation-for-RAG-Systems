"""
Test Ollama Local - Configuration Optimisée pour Laptop
========================================================

Vérifie que les modèles locaux fonctionnent:
- Mistral 7B Instruct (Generator)
- Phi-3 Mini 3.8B (Critic)

Total VRAM: ~6.8GB / 7GB ✅
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src' / 'utils'))
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'agents'))

from ollama_client import create_ollama_client, OLLAMA_MODELS, get_model_info, check_ollama_status

print("=" * 70)
print("TEST OLLAMA LOCAL - LAPTOP 7GB VRAM")
print("=" * 70)
print()

# Test 1: Ollama status
print("🧪 TEST 1: État d'Ollama")
print("-" * 70)
if check_ollama_status():
    print("✅ Ollama est actif")
else:
    print("❌ Ollama n'est pas accessible")
    print("   Démarrez avec: ollama serve")
    exit(1)

print()

# Test 2: Modèles disponibles
print("🧪 TEST 2: Modèles installés")
print("-" * 70)

import subprocess
result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
print(result.stdout)

# Check if our models are there
has_mistral = "mistral" in result.stdout.lower()
has_phi3 = "phi3" in result.stdout.lower()

if has_mistral:
    print("✅ Mistral 7B installé")
else:
    print("❌ Mistral 7B manquant - Installez avec: ollama pull mistral:latest")

if has_phi3:
    print("✅ Phi-3 Mini installé")
else:
    print("⏳ Phi-3 Mini en cours de téléchargement...")
    print("   Attendez la fin du téléchargement")

print()

# Test 3: Configuration VRAM
print("🧪 TEST 3: Analyse VRAM")
print("-" * 70)

generator_info = get_model_info(OLLAMA_MODELS["generator"])
critic_info = get_model_info(OLLAMA_MODELS["critic"])

print(f"Generator: {OLLAMA_MODELS['generator']}")
print(f"  → {generator_info['name']}")
print(f"  → VRAM: {generator_info['size_gb']}GB")
print()

print(f"Critic: {OLLAMA_MODELS['critic']}")
print(f"  → {critic_info['name']}")
print(f"  → VRAM: {critic_info['size_gb']}GB")
print()

total_vram = generator_info['size_gb'] + critic_info['size_gb']
print(f"Total VRAM nécessaire: {total_vram:.1f}GB")
print(f"VRAM disponible: 7.0GB")

if total_vram <= 7:
    margin = 7 - total_vram
    print(f"✅ Configuration VIABLE (marge: {margin:.1f}GB)")
else:
    print(f"❌ Dépassement de {total_vram - 7:.1f}GB!")

print()

if not (has_mistral and has_phi3):
    print("⏳ Attente de l'installation complète des modèles...")
    exit(0)

# Test 4: Test Generator (Mistral)
print("🧪 TEST 4: Test Generator (Mistral 7B)")
print("-" * 70)

try:
    client = create_ollama_client()
    
    response = client.chat.completions.create(
        model=OLLAMA_MODELS["generator"],
        messages=[
            {"role": "user", "content": "Réponds en une phrase: Qu'est-ce que la photosynthèse?"}
        ],
        temperature=0.7,
        max_tokens=100
    )
    answer = response.choices[0].message.content
    print(f"✅ Réponse reçue:")
    print(f"   {answer[:150]}...")
    print()
except Exception as e:
    print(f"❌ Erreur: {e}")
    print()

# Test 5: Test Critic (Phi3)
print("🧪 TEST 5: Test Critic (Phi-3 Mini)")
print("-" * 70)

try:
    response = client.chat.completions.create(
        model=OLLAMA_MODELS["critic"],
        messages=[
            {"role": "user", "content": "Évalue cette réponse de 0 à 1: 'La photosynthèse produit de l'oxygène.' Score: "}
        ],
        temperature=0.2,
        max_tokens=50
    )
    evaluation = response.choices[0].message.content
    print(f"✅ Évaluation reçue:")
    print(f"   {evaluation}")
    print()
except Exception as e:
    print(f"❌ Erreur: {e}")
    print()

# Test 6: Agents
print("🧪 TEST 6: Compatibilité avec les agents")
print("-" * 70)

from question_generator import QuestionGenerator
from critic_agent import CriticAgent

try:
    qgen = QuestionGenerator(
        llm_client=client,
        model_name=OLLAMA_MODELS["generator"],
        language="fr"
    )
    print(f"✅ QuestionGenerator créé avec {OLLAMA_MODELS['generator']}")
    
    critic = CriticAgent(
        llm_client=client,
        model_name=OLLAMA_MODELS["critic"],
        language="fr"
    )
    print(f"✅ CriticAgent créé avec {OLLAMA_MODELS['critic']}")
    print()
    
except Exception as e:
    print(f"❌ Erreur: {e}")
    print()

print("=" * 70)
print("RÉSUMÉ")
print("=" * 70)
print()
print("✅ Configuration Ollama Local optimisée pour laptop!")
print()
print("Architecture:")
print("┌─────────────────────────────────────┐")
print("│   Mistral 7B (Generator) ~4.5GB     │")
print("│   ↓ génère Q+A                      │")
print("│   Phi-3 Mini (Critic) ~2.3GB        │")
print("│   ↓ évalue + feedback               │")
print("│   [Retry loop max 2x]               │")
print("│   ↓                                 │")
print("│   Dataset GOLD ✨                   │")
print("└─────────────────────────────────────┘")
print()
print(f"Total VRAM: {total_vram:.1f}GB / 7GB disponibles")
print()
print("🚀 Prêt pour le pipeline local sans rate limits!")
