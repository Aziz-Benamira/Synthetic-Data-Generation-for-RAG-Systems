"""
Test OpenRouter Integration
============================

Teste la connexion à OpenRouter et les deux modèles:
- Mistral Small 3.1 24B (generator)
- Llama 3.3 70B (critic)
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add paths
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'utils'))
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'agents'))

from openrouter_client import create_openrouter_client, OPENROUTER_MODELS, get_model_info

print("=" * 60)
print("TEST OPENROUTER - MIGRATION DES LLM")
print("=" * 60)
print()

# Test 1: Création du client
print("🧪 TEST 1: Création du client OpenRouter")
print("-" * 40)
try:
    client = create_openrouter_client()
    print(f"✅ Client créé!")
    print(f"   Base URL: {client.base_url}")
    print(f"   API Key: {os.getenv('OPENROUTER_API_KEY')[:20]}...")
except Exception as e:
    print(f"❌ Erreur: {e}")
    sys.exit(1)

print()

# Test 2: Informations sur les modèles
print("🧪 TEST 2: Configuration des modèles")
print("-" * 40)
generator_model = OPENROUTER_MODELS["generator"]
critic_model = OPENROUTER_MODELS["critic"]

print(f"Generator: {generator_model}")
info = get_model_info(generator_model)
print(f"  → {info['name']}")
print(f"  → Context: {info['context_length']} tokens")
print()

print(f"Critic: {critic_model}")
info = get_model_info(critic_model)
print(f"  → {info['name']}")
print(f"  → Context: {info['context_length']} tokens")
print()

# Test 3: Test simple avec Generator (Mistral)
print("🧪 TEST 3: Test Generator (Mistral 24B)")
print("-" * 40)
try:
    response = client.chat.completions.create(
        model=generator_model,
        messages=[
            {"role": "user", "content": "Réponds en une phrase: Qu'est-ce que la photosynthèse?"}
        ],
        temperature=0.7,
        max_tokens=100
    )
    answer = response.choices[0].message.content
    print(f"✅ Réponse reçue:")
    print(f"   {answer[:200]}...")
    print()
except Exception as e:
    print(f"❌ Erreur: {e}")
    print()

# Test 4: Test simple avec Critic (Llama 70B)
print("🧪 TEST 4: Test Critic (Llama 70B)")
print("-" * 40)
try:
    response = client.chat.completions.create(
        model=critic_model,
        messages=[
            {"role": "user", "content": "Évalue cette réponse sur une échelle de 0 à 1: 'La photosynthèse est un processus.' Score: "}
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

# Test 5: Compatibilité avec les agents
print("🧪 TEST 5: Compatibilité avec les agents")
print("-" * 40)
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'agents'))

from question_generator import QuestionGenerator
from critic_agent import CriticAgent

try:
    # Test QuestionGenerator
    qgen = QuestionGenerator(
        llm_client=client,
        model_name=generator_model,
        language="fr"
    )
    print(f"✅ QuestionGenerator créé avec {generator_model}")
    
    # Test CriticAgent
    critic = CriticAgent(
        llm_client=client,
        model_name=critic_model,
        language="fr"
    )
    print(f"✅ CriticAgent créé avec {critic_model}")
    print()
    
except Exception as e:
    print(f"❌ Erreur: {e}")
    print()

print("=" * 60)
print("RÉSUMÉ")
print("=" * 60)
print()
print("✅ OpenRouter configuré avec succès!")
print(f"   Generator: Mistral Small 3.1 24B (gratuit)")
print(f"   Critic: Llama 3.3 70B Instruct (gratuit)")
print()
print("Architecture finale:")
print("┌─────────────────────────────────────┐")
print("│   Mistral 24B (Generator)           │")
print("│   ↓ génère Q+A                      │")
print("│   Llama 70B (Critic)                │")
print("│   ↓ évalue + feedback               │")
print("│   [Retry loop max 2x]               │")
print("│   ↓                                 │")
print("│   Dataset GOLD ✨                   │")
print("└─────────────────────────────────────┘")
print()
print("🚀 Prêt pour le pipeline complet!")
