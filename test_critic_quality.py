"""
Test de la Qualité du Critic
=============================

Teste si le Critic REJETTE vraiment les mauvais QA.
On va lui donner des exemples intentionnellement défaillants.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / 'src' / 'utils'))
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'agents'))
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'chunking'))

from openrouter_client import create_openrouter_client, OPENROUTER_MODELS
from critic_agent import CriticAgent, FinalDecision
from answer_generator import QAPair

# Fake chunk pour tester
@dataclass
class FakeChunk:
    chunk_id: str = "test_chunk"
    content: str = """
    La photosynthèse est un processus biologique par lequel les plantes 
    convertissent la lumière en énergie chimique. Ce processus se déroule 
    dans les chloroplastes et produit du glucose et de l'oxygène.
    
    La formule est : 6CO2 + 6H2O + lumière → C6H12O6 + 6O2
    """
    chapter_title: str = "Biologie"
    section_title: str = "Photosynthèse"
    page_range: tuple = (1, 1)

print("=" * 70)
print("TEST QUALITÉ DU CRITIC - DÉTECTION DE MAUVAIS QA")
print("=" * 70)
print()

# Create client and critic
client = create_openrouter_client()
critic = CriticAgent(
    llm_client=client,
    model_name=OPENROUTER_MODELS["critic"],
    language="fr",
    strict_mode=True
)

chunk = FakeChunk()

# Test cases : MAUVAIS QA qui DOIVENT être rejetés
bad_qa_pairs = [
    {
        "name": "❌ ANCRAGE FAIBLE - Exemple inventé",
        "qa": QAPair(
            question="Comment fonctionne la photosynthèse?",
            answer="La photosynthèse fonctionne grâce aux chloroplastes. Par exemple, un tournesol peut produire 50g de glucose par jour.",  # 50g/jour N'EST PAS dans le chunk!
            question_type="conceptual",
            difficulty="medium",
            supporting_quotes=[],
            chunk_id="test",
            source_file="test.pdf",
            page_range=(1,1),
            chapter="Bio",
            section="Photo",
            confidence=0.9
        ),
        "should_fail": "anchoring",
        "reason": "L'exemple '50g par jour' n'est PAS dans le chunk source"
    },
    {
        "name": "❌ LOCAL_ANSWERABILITY - Référence externe",
        "qa": QAPair(
            question="Comment la photosynthèse se compare-t-elle à la respiration cellulaire décrite au chapitre 3?",
            answer="La photosynthèse produit du glucose tandis que la respiration le consomme.",
            question_type="comparative",
            difficulty="hard",
            supporting_quotes=["La photosynthèse produit du glucose"],
            chunk_id="test",
            source_file="test.pdf",
            page_range=(1,1),
            chapter="Bio",
            section="Photo",
            confidence=0.8
        ),
        "should_fail": "local_answerability",
        "reason": "La question fait référence au 'chapitre 3' qui n'est pas dans ce chunk"
    },
    {
        "name": "❌ COMPLÉTUDE - Réponse trop courte/triviale",
        "qa": QAPair(
            question="Expliquez en détail le processus de la photosynthèse et son importance écologique.",
            answer="C'est un processus important.",  # Réponse triviale!
            question_type="conceptual",
            difficulty="medium",
            supporting_quotes=[],
            chunk_id="test",
            source_file="test.pdf",
            page_range=(1,1),
            chapter="Bio",
            section="Photo",
            confidence=0.5
        ),
        "should_fail": "completeness",
        "reason": "La réponse est triviale et n'adresse pas tous les aspects de la question"
    },
    {
        "name": "❌ CLARTÉ - Question vague/orale",
        "qa": QAPair(
            question="C'est quoi le truc avec la photosynthèse?",  # Style oral/vague
            answer="La photosynthèse est un processus biologique par lequel les plantes convertissent la lumière en énergie.",
            question_type="factual",
            difficulty="easy",
            supporting_quotes=["processus biologique"],
            chunk_id="test",
            source_file="test.pdf",
            page_range=(1,1),
            chapter="Bio",
            section="Photo",
            confidence=0.9
        ),
        "should_fail": "clarity",
        "reason": "La question utilise un langage oral/vague ('le truc')"
    },
    {
        "name": "❌ EXACTITUDE - Information incorrecte",
        "qa": QAPair(
            question="Quelle est la formule de la photosynthèse?",
            answer="La formule est : 6CO2 + 6H2O + lumière → C6H12O6 + 12O2",  # 12O2 au lieu de 6O2!
            question_type="factual",
            difficulty="easy",
            supporting_quotes=["6CO2 + 6H2O"],
            chunk_id="test",
            source_file="test.pdf",
            page_range=(1,1),
            chapter="Bio",
            section="Photo",
            confidence=0.9
        ),
        "should_fail": "factual_accuracy",
        "reason": "La formule est incorrecte (12O2 au lieu de 6O2)"
    }
]

# Test chaque cas
results = []
print(f"Chunk content pour référence:")
print("-" * 70)
print(chunk.content.strip())
print("-" * 70)
print()

for i, test_case in enumerate(bad_qa_pairs, 1):
    print(f"🧪 TEST {i}/5: {test_case['name']}")
    print(f"   Question: {test_case['qa'].question}")
    print(f"   Réponse: {test_case['qa'].answer[:100]}...")
    print(f"   Devrait échouer sur: {test_case['should_fail']}")
    print(f"   Raison: {test_case['reason']}")
    
    try:
        evaluation = critic.evaluate(test_case['qa'], chunk)
        
        decision = evaluation.decision
        failed_criteria = evaluation.failed_criteria
        overall_score = evaluation.overall_score
        
        # Vérifier si le test a BIEN échoué
        if decision == FinalDecision.REJECT:
            if test_case['should_fail'] in failed_criteria:
                print(f"   ✅ CORRECT - Rejeté sur le bon critère: {test_case['should_fail']}")
                print(f"      Score global: {overall_score:.2f}")
                print(f"      Critères échoués: {failed_criteria}")
                results.append("✅ PASS")
            else:
                print(f"   ⚠️  REJETÉ mais pas le bon critère")
                print(f"      Attendu: {test_case['should_fail']}")
                print(f"      Obtenu: {failed_criteria}")
                results.append("⚠️  PARTIAL")
        else:
            print(f"   ❌ ÉCHEC - Le Critic a ACCEPTÉ un mauvais QA!")
            print(f"      Score: {overall_score:.2f}")
            print(f"      Le Critic est TROP LAXISTE")
            results.append("❌ FAIL")
            
    except Exception as e:
        print(f"   ❌ ERREUR: {e}")
        results.append("❌ ERROR")
    
    print()

# Résumé
print("=" * 70)
print("RÉSUMÉ DU TEST DE QUALITÉ")
print("=" * 70)
print()

correct = results.count("✅ PASS")
partial = results.count("⚠️  PARTIAL")
failed = results.count("❌ FAIL")
errors = results.count("❌ ERROR")

print(f"✅ Détections correctes: {correct}/5")
print(f"⚠️  Détections partielles: {partial}/5")
print(f"❌ Faux positifs (accepté alors que mauvais): {failed}/5")
print(f"❌ Erreurs: {errors}/5")
print()

if failed > 0:
    print("⚠️  PROBLÈME DÉTECTÉ:")
    print(f"   Le Critic accepte {failed} QA pair(s) qui devraient être rejetés!")
    print(f"   Le Critic est SOUS-CALIBRÉ (trop permissif)")
    print()
    print("   SOLUTIONS POSSIBLES:")
    print("   1. Réduire le threshold de PASS (actuellement 0.7 → essayer 0.8)")
    print("   2. Renforcer les prompts du Critic avec plus d'exemples négatifs")
    print("   3. Utiliser un modèle plus puissant pour le Critic")
    print("   4. Ajouter des pénalités pour patterns suspects")
elif correct == 5:
    print("✅ Le Critic fonctionne CORRECTEMENT!")
    print("   Il détecte et rejette les mauvais QA pairs")
else:
    print("⚠️  Le Critic fonctionne partiellement")
    print("   Quelques améliorations nécessaires")
