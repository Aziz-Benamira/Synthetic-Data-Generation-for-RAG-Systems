"""
Vérification Setup Local - Ollama Ready Check
==============================================

Vérifie que tous les modèles sont téléchargés et prêts avant de lancer le pipeline complet.
"""

import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src' / 'utils'))

from ollama_client import create_ollama_client, OLLAMA_MODELS, calculate_vram_usage

print("=" * 80)
print("VÉRIFICATION SETUP LOCAL - OLLAMA")
print("=" * 80)
print()

# 1. Vérifier que Ollama tourne
print("1️⃣  Vérification serveur Ollama...")
try:
    result = subprocess.run(
        ["ollama", "list"], 
        capture_output=True, 
        text=True, 
        check=True
    )
    print("   ✅ Ollama fonctionne!")
    print()
except Exception as e:
    print(f"   ❌ Ollama non disponible: {e}")
    print("   Lancez: ollama serve")
    sys.exit(1)

# 2. Lister modèles installés
print("2️⃣  Modèles installés:")
print()
lines = result.stdout.strip().split('\n')[1:]  # Skip header
installed = []
for line in lines:
    if line.strip():
        parts = line.split()
        model_name = parts[0]
        size = parts[2] if len(parts) > 2 else "?"
        installed.append(model_name)
        print(f"   • {model_name:20s} {size:>10s}")

print()

# 3. Vérifier modèles requis
print("3️⃣  Modèles requis pour le pipeline:")
print()

required_generator = OLLAMA_MODELS["generator"]
required_critic = OLLAMA_MODELS["critic"]

generator_ready = required_generator in installed
critic_ready = required_critic in installed

print(f"   Generator: {required_generator:20s} {'✅' if generator_ready else '❌ MANQUANT'}")
print(f"   Critic:    {required_critic:20s} {'✅' if critic_ready else '❌ MANQUANT'}")
print()

if not generator_ready:
    print(f"   ⚠️  Téléchargez: ollama pull {required_generator}")
if not critic_ready:
    print(f"   ⚠️  Téléchargez: ollama pull {required_critic}")

# 4. Calcul VRAM
print("4️⃣  Utilisation VRAM estimée:")
print()
vram = calculate_vram_usage(required_generator, required_critic)
print(f"   Generator ({required_generator}): ~{vram['generator_gb']:.1f} GB")
print(f"   Critic ({required_critic}):    ~{vram['critic_gb']:.1f} GB")
print(f"   {'─' * 40}")
print(f"   TOTAL:                       ~{vram['total_gb']:.1f} GB")
print()

if vram['total_gb'] > 7:
    print("   ⚠️  ATTENTION: VRAM > 7GB (votre limite)")
    print("   Le système risque de swapper sur RAM")
elif vram['total_gb'] > 6.5:
    print("   ⚙️  Proche de la limite (7GB disponible)")
    print("   Système devrait fonctionner, mais fermez les autres apps")
else:
    print(f"   ✅ Marge confortable ({7 - vram['total_gb']:.1f} GB libre)")

print()

# 5. Test de connexion client
print("5️⃣  Test du client Ollama...")
try:
    client = create_ollama_client()
    print("   ✅ Client créé avec succès!")
except Exception as e:
    print(f"   ❌ Erreur client: {e}")
    sys.exit(1)

print()

# 6. Résumé final
print("=" * 80)
print("RÉSUMÉ")
print("=" * 80)
print()

all_ready = generator_ready and critic_ready and vram['total_gb'] <= 7

if all_ready:
    print("🎉 SETUP COMPLET - PRÊT POUR LE PIPELINE LOCAL!")
    print()
    print("   Commandes suivantes:")
    print()
    print("   # Test rapide du workflow:")
    print("   python test_ollama_local.py")
    print()
    print("   # Pipeline complet (5 chunks):")
    print("   python test_pipeline_local.py")
    print()
    print("   🎯 Objectif: Voir des retry loops déclenchés!")
    
else:
    print("⚠️  SETUP INCOMPLET")
    print()
    if not generator_ready:
        print(f"   • Téléchargez: ollama pull {required_generator}")
    if not critic_ready:
        print(f"   • Téléchargez: ollama pull {required_critic}")
    if vram['total_gb'] > 7:
        print("   • VRAM insuffisante, envisagez des modèles plus petits")

print()
