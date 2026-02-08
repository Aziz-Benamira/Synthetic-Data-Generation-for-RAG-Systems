"""
Exemples d'Utilisation du Module LLM
=====================================

Démonstration des différentes façons d'utiliser le module LLM.
"""

import asyncio
from src.llm import (
    LLMManager,
    LLMMessage,
    LLMConfig,
    create_ollama_manager,
    create_openrouter_manager
)


# =============================================================================
# EXEMPLE 1 : Utilisation Basique avec Ollama
# =============================================================================

def example_basic_ollama():
    """Exemple simple avec Ollama"""
    print("\n=== EXEMPLE 1: Ollama Basique ===")
    
    # Créer un manager
    manager = LLMManager.from_ollama("mistral:latest")
    
    # Appel simple
    response = manager.generate("Quelle est la capitale de la France?")
    
    print(f"Question: Quelle est la capitale de la France?")
    print(f"Réponse: {response.content}")
    print(f"Tokens utilisés: {response.tokens_used}")
    print(f"Modèle: {response.model}")


# =============================================================================
# EXEMPLE 2 : Configuration Personnalisée
# =============================================================================

def example_custom_config():
    """Exemple avec configuration personnalisée"""
    print("\n=== EXEMPLE 2: Configuration Personnalisée ===")
    
    # Configuration pour évaluation (déterministe)
    config = LLMConfig(
        temperature=0.1,  # Très déterministe
        max_tokens=100,   # Réponses courtes
        top_p=0.9
    )
    
    manager = LLMManager.from_ollama("mistral:latest", config=config)
    
    response = manager.generate(
        prompt="Is 2+2=4? Answer with yes or no only.",
        system_prompt="You are a precise calculator."
    )
    
    print(f"Réponse: {response.content}")


# =============================================================================
# EXEMPLE 3 : Conversation Multi-turn
# =============================================================================

def example_conversation():
    """Exemple de conversation avec contexte"""
    print("\n=== EXEMPLE 3: Conversation Multi-turn ===")
    
    manager = LLMManager.from_ollama("mistral:latest")
    
    # Conversation
    messages = [
        LLMMessage(role="system", content="Tu es un expert en mathématiques."),
        LLMMessage(role="user", content="Qu'est-ce qu'une intégrale?")
    ]
    
    response1 = manager.generate_from_messages(messages)
    print(f"User: Qu'est-ce qu'une intégrale?")
    print(f"Assistant: {response1.content[:100]}...")
    
    # Continuer la conversation
    messages.append(LLMMessage(role="assistant", content=response1.content))
    messages.append(LLMMessage(role="user", content="Donne un exemple simple."))
    
    response2 = manager.generate_from_messages(messages)
    print(f"\nUser: Donne un exemple simple.")
    print(f"Assistant: {response2.content[:100]}...")


# =============================================================================
# EXEMPLE 4 : Utilisation pour Métriques (Cas d'usage de ton collègue)
# =============================================================================

class FaithfulnessMetric:
    """Métrique de fidélité (faithfulness)"""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm = llm_manager
        # Config optimale pour évaluation
        self.eval_config = LLMConfig(temperature=0.1, max_tokens=50)
    
    def evaluate(self, answer: str, context: str) -> float:
        """Évaluer si la réponse est fidèle au contexte"""
        prompt = f"""Given the following context and answer, determine if the answer is faithful to the context.

Context: {context}

Answer: {answer}

Is the answer faithful to the context? Reply with ONLY "yes" or "no".
"""
        
        response = self.llm.generate(prompt, config=self.eval_config)
        
        # Parse réponse
        content = response.content.lower().strip()
        return 1.0 if "yes" in content else 0.0


def example_metrics():
    """Exemple d'utilisation pour métriques d'évaluation"""
    print("\n=== EXEMPLE 4: Utilisation pour Métriques ===")
    
    manager = LLMManager.from_ollama("mistral:latest")
    metric = FaithfulnessMetric(manager)
    
    # Test 1 : Réponse fidèle
    score1 = metric.evaluate(
        answer="Paris est la capitale de la France.",
        context="La France est un pays européen dont la capitale est Paris."
    )
    print(f"Test 1 (fidèle): Score = {score1}")
    
    # Test 2 : Réponse infidèle
    score2 = metric.evaluate(
        answer="Londres est la capitale de la France.",
        context="La France est un pays européen dont la capitale est Paris."
    )
    print(f"Test 2 (infidèle): Score = {score2}")


# =============================================================================
# EXEMPLE 5 : Comparaison Ollama vs OpenRouter
# =============================================================================

def example_compare_providers():
    """Comparer différents providers"""
    print("\n=== EXEMPLE 5: Comparaison Providers ===")
    
    prompt = "What is the capital of France? Answer in one word."
    
    # Ollama
    manager_ollama = LLMManager.from_ollama("mistral:latest")
    response_ollama = manager_ollama.generate(prompt)
    print(f"Ollama: {response_ollama.content}")
    
    # OpenRouter (décommenter si tu as une clé)
    # manager_openrouter = LLMManager.from_openrouter(
    #     "mistralai/mistral-small-3.1-24b-instruct:free"
    # )
    # response_openrouter = manager_openrouter.generate(prompt)
    # print(f"OpenRouter: {response_openrouter.content}")


# =============================================================================
# EXEMPLE 6 : Mode Asynchrone
# =============================================================================

async def example_async():
    """Exemple d'utilisation asynchrone"""
    print("\n=== EXEMPLE 6: Mode Asynchrone ===")
    
    manager = LLMManager.from_ollama("mistral:latest")
    
    # Appel async
    response = await manager.agenerate("Bonjour!")
    print(f"Réponse async: {response.content}")
    
    # Plusieurs appels en parallèle
    tasks = [
        manager.agenerate("Capitale de France?"),
        manager.agenerate("Capitale d'Italie?"),
        manager.agenerate("Capitale d'Espagne?")
    ]
    
    responses = await asyncio.gather(*tasks)
    
    for i, resp in enumerate(responses, 1):
        print(f"Réponse {i}: {resp.content}")


# =============================================================================
# EXEMPLE 7 : Usage dans un Agent
# =============================================================================

class SimpleCriticAgent:
    """Agent critique simplifié"""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm = llm_manager
        self.config = LLMConfig(temperature=0.2, max_tokens=200)
    
    def evaluate_qa_pair(self, question: str, answer: str, context: str) -> dict:
        """Évaluer une paire QA"""
        prompt = f"""Evaluate this Question-Answer pair given the context.

Context: {context}

Question: {question}
Answer: {answer}

Rate the quality from 1-5 and explain why.
Format: Score: X/5
Reason: ...
"""
        
        response = self.llm.generate(
            prompt=prompt,
            system_prompt="You are a quality evaluator for educational content.",
            config=self.config
        )
        
        return {
            "evaluation": response.content,
            "tokens": response.tokens_used
        }


def example_agent():
    """Exemple d'utilisation dans un agent"""
    print("\n=== EXEMPLE 7: Usage dans un Agent ===")
    
    manager = LLMManager.from_ollama("mistral:latest")
    critic = SimpleCriticAgent(manager)
    
    result = critic.evaluate_qa_pair(
        question="Qu'est-ce que la loi normale?",
        answer="La loi normale est une distribution de probabilité continue.",
        context="En théorie des probabilités, la loi normale ou distribution normale..."
    )
    
    print(f"Évaluation: {result['evaluation'][:150]}...")
    print(f"Tokens utilisés: {result['tokens']}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Exécuter tous les exemples"""
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║     EXEMPLES D'UTILISATION DU MODULE LLM                  ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    
    # Exemples synchrones
    example_basic_ollama()
    example_custom_config()
    example_conversation()
    example_metrics()
    example_compare_providers()
    example_agent()
    
    # Exemple asynchrone
    print("\n" + "="*60)
    print("Exemples asynchrones...")
    asyncio.run(example_async())
    
    print("\n" + "="*60)
    print("✅ Tous les exemples terminés!")


if __name__ == "__main__":
    main()
