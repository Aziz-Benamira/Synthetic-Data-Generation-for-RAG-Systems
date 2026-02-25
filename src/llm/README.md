#  Module LLM - Gestion Centralisée des Appels LLM

Module unifié pour gérer tous les appels aux LLMs de manière cohérente et réutilisable.

##  Objectif

Fournir une **interface unique** pour appeler différents LLMs (Ollama, OpenRouter, llama.cpp) sans dupliquer le code dans chaque agent ou métrique d'évaluation.

##  Structure

```
src/llm/
├── base.py         # Classes abstraites et interfaces
├── providers.py    # Implémentations (Ollama, OpenRouter, llama.cpp)
├── manager.py      # Manager unifié simplifié
├── __init__.py     # Exports
└── README.md       # Cette documentation
```

##  Usage Simple

### 1. Appel Basique avec Ollama

```python
from src.llm import LLMManager

# Créer un manager avec Ollama
manager = LLMManager.from_ollama("mistral:latest")

# Appel simple
response = manager.generate("Quelle est la capitale de la France?")
print(response.content)  # "Paris"
print(response.tokens_used)  # 150
```

### 2. Avec OpenRouter

```python
from src.llm import LLMManager

# Créer un manager avec OpenRouter
manager = LLMManager.from_openrouter(
    model="mistralai/mistral-small-3.1-24b-instruct:free",
    api_key="sk-or-..."  # Ou utiliser OPENROUTER_API_KEY env var
)

response = manager.generate("Hello, how are you?")
print(response.content)
```

### 3. Avec llama.cpp (DeepSeek R1)

```python
from src.llm import LLMManager

# Utiliser le serveur llama.cpp local
manager = LLMManager.from_llamacpp(
    model="deepseek-r1-distill-qwen-32b",
    base_url="http://localhost:8080/v1"
)

response = manager.generate("Explain quantum computing.")
print(response.content)
```

##  Usage Avancé

### Configuration Personnalisée

```python
from src.llm import LLMManager, LLMConfig

# Configuration fine
config = LLMConfig(
    temperature=0.3,        # Plus déterministe
    max_tokens=500,         # Limite de tokens
    top_p=0.9,
    frequency_penalty=0.2
)

manager = LLMManager.from_ollama("mistral:latest", config=config)
response = manager.generate("Explain...")
```

### Conversations Multi-turn

```python
from src.llm import LLMManager, LLMMessage

manager = LLMManager.from_ollama("mistral:latest")

# Conversation avec contexte
messages = [
    LLMMessage(role="system", content="Tu es un expert en probabilités."),
    LLMMessage(role="user", content="Qu'est-ce que la loi normale?"),
]

response = manager.generate_from_messages(messages)
print(response.content)

# Continuer la conversation
messages.append(LLMMessage(role="assistant", content=response.content))
messages.append(LLMMessage(role="user", content="Donne un exemple."))

response = manager.generate_from_messages(messages)
```

### Mode Asynchrone

```python
import asyncio
from src.llm import LLMManager

async def main():
    manager = LLMManager.from_ollama("mistral:latest")
    
    # Appel async
    response = await manager.agenerate("Hello!")
    print(response.content)

asyncio.run(main())
```

##  Utilisation dans les Agents

### Avant (code dupliqué)

```python
class CriticAgent:
    def __init__(self):
        self.llm_client = OpenAI(base_url="...", api_key="...")
    
    def evaluate(self, qa_pair):
        response = self.llm_client.chat.completions.create(
            model="mistral:latest",
            messages=[...],
            temperature=0.7,
            # Code dupliqué partout !
        )
        return response.choices[0].message.content
```

### Après (module centralisé)

```python
from src.llm import LLMManager

class CriticAgent:
    def __init__(self, llm_manager: LLMManager):
        self.llm = llm_manager
    
    def evaluate(self, qa_pair):
        response = self.llm.generate(
            prompt=f"Evaluate this QA: {qa_pair}",
            system_prompt="You are a quality evaluator."
        )
        return response.content
```

##  Utilisation pour les Métriques d'Évaluation


```python
from src.llm import LLMManager, LLMConfig

class FaithfulnessMetric:
    def __init__(self, llm_manager: LLMManager):
        self.llm = llm_manager
        # Configuration optimale pour évaluation
        self.eval_config = LLMConfig(temperature=0.1, max_tokens=100)
    
    def evaluate(self, answer: str, context: str) -> float:
        prompt = f"""
        Answer: {answer}
        Context: {context}
        
        Is the answer faithful to the context? (yes/no)
        """
        
        response = self.llm.generate(prompt, config=self.eval_config)
        return 1.0 if "yes" in response.content.lower() else 0.0

# Utilisation
manager = LLMManager.from_ollama("mistral:latest")
metric = FaithfulnessMetric(manager)

score = metric.evaluate(
    answer="Paris est la capitale.",
    context="La capitale de la France est Paris."
)
print(f"Faithfulness score: {score}")
```

##  Avantages

| Avantage | Description |
|----------|-------------|
| **Réutilisabilité** | Une seule implémentation pour tous les agents ET métriques |
| **Maintenabilité** | Changer de provider = changer 1 ligne |
| **Testabilité** | Facile de mocker le LLMManager pour les tests |
| **Flexibilité** | Supporte plusieurs providers avec la même API |
| **Type Safety** | Classes typées avec dataclasses |

##  Migration du Code Existant

### Étape 1 : Remplacer les imports

**Avant :**
```python
from openai import OpenAI
client = OpenAI(base_url="...", api_key="...")
```

**Après :**
```python
from src.llm import LLMManager
manager = LLMManager.from_ollama("mistral:latest")
```

### Étape 2 : Remplacer les appels

**Avant :**
```python
response = client.chat.completions.create(
    model="mistral:latest",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7
)
content = response.choices[0].message.content
```

**Après :**
```python
response = manager.generate("Hello")
content = response.content
```

##  API Reference

### LLMManager

#### Méthodes de Création

- `LLMManager.from_ollama(model, base_url, config)` → Manager Ollama
- `LLMManager.from_openrouter(model, api_key, config)` → Manager OpenRouter
- `LLMManager.from_llamacpp(model, base_url, config)` → Manager llama.cpp

#### Méthodes de Génération

- `generate(prompt, system_prompt, config)` → LLMResponse
- `agenerate(prompt, system_prompt, config)` → LLMResponse (async)
- `generate_from_messages(messages, config)` → LLMResponse
- `agenerate_from_messages(messages, config)` → LLMResponse (async)

#### Utilitaires

- `count_tokens(text)` → int
- `get_info()` → Dict[str, Any]

### LLMConfig

```python
LLMConfig(
    temperature=0.7,          # Créativité (0.0-1.0)
    max_tokens=2000,          # Limite de sortie
    top_p=1.0,                # Nucleus sampling
    frequency_penalty=0.0,    # Pénalité répétition
    presence_penalty=0.0,     # Pénalité présence
    stop_sequences=None       # Séquences d'arrêt
)
```

### LLMResponse

```python
response = manager.generate("Hello")
response.content           # str: Contenu de la réponse
response.model            # str: Modèle utilisé
response.tokens_used      # int: Tokens consommés
response.finish_reason    # str: Raison de fin
response.raw_response     # Any: Réponse brute du provider
```



