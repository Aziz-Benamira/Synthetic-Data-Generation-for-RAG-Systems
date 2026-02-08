# 🧪 Guide de Test - Commandes Rapides

## 📋 Fichiers de Test Disponibles

### 1️⃣ **Tester le Semantic Chunker Seul**

**Fichier:** Créer `test_chunker.py`

```python
# test_chunker.py
from src.chunking.semantic_chunker import SemanticChunker

# Initialiser le chunker
chunker = SemanticChunker(
    max_chunk_size=1000,
    min_chunk_size=200,
    overlap=50
)

# Parser un PDF
chunks = chunker.chunk_pdf("data/pdfs/M2_cours.pdf")

# Afficher les résultats
print(f"\n✅ {len(chunks)} chunks extraits\n")
for i, chunk in enumerate(chunks[:3], 1):
    print(f"Chunk {i}: {chunk.chunk_id}")
    print(f"  Type: {chunk.semantic_type}")
    print(f"  Pages: {chunk.page_range}")
    print(f"  Chapter: {chunk.chapter_title}")
    print(f"  Section: {chunk.section_title}")
    print(f"  Longueur: {len(chunk.content)} caractères")
    print(f"  Preview: {chunk.content[:100]}...")
    print()
```

**Commande:**
```powershell
C:\Users\benam\miniconda3\python.exe test_chunker.py
```

---

### 2️⃣ **Tester le Pipeline Complet (Recommandé)**

**Fichier:** `src/orchestrator/pipeline.py` (a déjà un main)

**Option A - Pipeline Simple (5 chunks, rapide):**

```python
# test_full_pipeline.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.ollama_client import create_ollama_client, OLLAMA_MODELS
from src.orchestrator.pipeline import DatasetPipeline, PipelineConfig

# Configuration
config = PipelineConfig(
    pdf_path="data/pdfs/M2_cours.pdf",
    output_dir="output_test",
    max_chunks=5,                    # PETIT TEST
    questions_per_chunk=2,
    generator_model=OLLAMA_MODELS["generator"],
    critic_model=OLLAMA_MODELS["critic"],
    max_retries=2,
    temperature=0.7,
    language="fr"
)

# Créer client Ollama
print("🔌 Connexion à Ollama...")
client = create_ollama_client()
print("✅ Connecté!\n")

# Lancer le pipeline
print("🚀 Lancement du pipeline...\n")
pipeline = DatasetPipeline(config=config, llm_client=client)
dataset = pipeline.run()

# Afficher les statistiques
print("\n" + "="*80)
print("📊 RÉSULTATS")
print("="*80)
print(f"Chunks traités: {pipeline.stats.processed_chunks}")
print(f"Questions générées: {pipeline.stats.total_questions_generated}")
print(f"QA pairs acceptées: {pipeline.stats.passed_qa_pairs}")
print(f"QA pairs rejetées: {pipeline.stats.rejected_qa_pairs}")
print(f"Taux de rejet: {(1-pipeline.stats.pass_rate)*100:.1f}%")
print(f"Retries totaux: {pipeline.stats.total_retries}")
print(f"\n✅ Dataset exporté: {config.output_dir}/")
```

**Commande:**
```powershell
C:\Users\benam\miniconda3\python.exe test_full_pipeline.py
```

**Option B - Pipeline Complet (tous les chunks, ~30-40 min):**

```python
# Même code mais:
config = PipelineConfig(
    pdf_path="data/pdfs/M2_cours.pdf",
    output_dir="output_full",
    max_chunks=None,                 # TOUS les chunks (110)
    questions_per_chunk=2,           # 220 QA pairs au total
    ...
)
```

**Commande:**
```powershell
C:\Users\benam\miniconda3\python.exe test_full_pipeline.py
```

---

### 3️⃣ **Tests Existants (Déjà dans le projet)**

#### **A. Test du Pipeline Détaillé**
```powershell
C:\Users\benam\miniconda3\python.exe test_pipeline_detailed_logging.py
```
- Montre tous les logs détaillés
- Bon pour debug

#### **B. Test avec Chunks Ambigus (Celui qu'on vient de faire)**
```powershell
C:\Users\benam\miniconda3\python.exe quick_challenge_test.py
```
- 3 chunks ambigus
- Montre les scores détaillés du critic
- Rapide (~2 min)

#### **C. Comparaison Avant/Après Seif**
```powershell
C:\Users\benam\miniconda3\python.exe run_demo.py
```
- Compare current vs Seif's critic
- Run complet (2x 5 chunks)
- Prend ~10 minutes

#### **D. Test Question Generator Seul**
```powershell
C:\Users\benam\miniconda3\python.exe test_question_generator.py
```

#### **E. Test Critic Seul**
```powershell
C:\Users\benam\miniconda3\python.exe test_critic_agent.py
```

---

## 🎯 **Commandes Recommandées pour la Présentation**

### **Test Rapide (5 minutes):**
```powershell
# 1. Pré-charger les modèles Ollama
ollama run mistral:latest "test"
ollama run llama3:8b "test"

# 2. Tester avec chunks ambigus (montre les scores détaillés)
C:\Users\benam\miniconda3\python.exe quick_challenge_test.py
```

### **Test Complet (10 minutes):**
```powershell
# Comparer avant/après Seif avec vraies données
C:\Users\benam\miniconda3\python.exe run_demo.py
```

### **Test Production (30-40 minutes):**
```powershell
# Créer test_full_pipeline.py avec max_chunks=None
C:\Users\benam\miniconda3\python.exe test_full_pipeline.py
```

---

## 🔧 **Avant de Lancer un Test**

### **1. Vérifier Ollama:**
```powershell
ollama ps
```
Si vide → charger les modèles:
```powershell
ollama run mistral:latest "test"
ollama run llama3:8b "test"
```

### **2. Configurer GPU (RTX 5060):**
```powershell
. .\setup_gpu.ps1
```

### **3. Vérifier Python:**
```powershell
C:\Users\benam\miniconda3\python.exe --version
```

---

## 📊 **Fichiers de Sortie**

Après le test, tu trouveras:

```
output_test/                         (ou output_full/)
├── dataset.json                     # Dataset complet (HuggingFace format)
├── dataset_metadata.json            # Métadonnées
├── statistics.json                  # Stats détaillées
└── chunks/
    ├── chunk_1.1.c1.json           # Chunks individuels
    └── ...
```

---

## 💡 **Si le Tutor Demande:**

### **"Montre-moi le chunking"**
→ Créer `test_chunker.py` (code ci-dessus) et lancer

### **"Montre-moi le pipeline complet"**
→ Lancer `run_demo.py` (déjà prêt, montre avant/après Seif)

### **"Montre-moi comment le critic évalue"**
→ Lancer `quick_challenge_test.py` (montre scores détaillés)

### **"Génère un vrai dataset"**
→ Créer `test_full_pipeline.py` avec `max_chunks=None`

---

## ⚡ **Commande Ultra-Rapide (30 secondes)**

Si besoin de montrer que ça marche vite:

```python
# quick_demo.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.chunking.semantic_chunker import SemanticChunker

chunker = SemanticChunker()
chunks = chunker.chunk_pdf("data/pdfs/M2_cours.pdf")

print(f"\n✅ {len(chunks)} chunks extraits en quelques secondes!")
print(f"\nPremier chunk:")
print(f"  ID: {chunks[0].chunk_id}")
print(f"  Type: {chunks[0].semantic_type}")
print(f"  Chapter: {chunks[0].chapter_title}")
print(f"  Section: {chunks[0].section_title}")
print(f"  Contenu: {chunks[0].content[:200]}...")
```

```powershell
C:\Users\benam\miniconda3\python.exe quick_demo.py
```

---

## 📝 **Résumé des Commandes Principales**

| Test | Fichier | Durée | Commande |
|------|---------|-------|----------|
| **Chunker seul** | `test_chunker.py` | 10s | `python test_chunker.py` |
| **Scores détaillés** | `quick_challenge_test.py` | 2min | `python quick_challenge_test.py` |
| **Comparaison Seif** | `run_demo.py` | 10min | `python run_demo.py` |
| **Pipeline complet** | `test_full_pipeline.py` | 30min | `python test_full_pipeline.py` |

**Note:** Toujours préfixer avec `C:\Users\benam\miniconda3\python.exe` pour utiliser le bon environnement.
