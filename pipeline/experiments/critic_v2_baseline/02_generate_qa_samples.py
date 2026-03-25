#!/usr/bin/env python3
"""
Étape 2: Génération de paires QA sur les chunks extraits
=========================================================

Génère 3 paires QA par chunk en utilisant le QuestionGenerator et AnswerGenerator.

Input: data/chunks.json
Output: data/qa_samples.json
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/experiment.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Imports
try:
    from src.llm import LLMManager, LLMConfig
    from src.agents.question_generator import QuestionGenerator
    from src.agents.answer_generator import AnswerGenerator, QAPair
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)


@dataclass
class SimpleChunk:
    """Chunk simplifié pour compatibilité avec les agents"""
    chunk_id: str
    content: str
    semantic_type: str
    chapter_title: str
    section_title: str
    page_range: tuple
    metadata: dict


def load_chunks(chunks_path: str) -> List[Dict[str, Any]]:
    """Charger les chunks depuis JSON"""
    logger.info(f"📖 Loading chunks from: {chunks_path}")
    with open(chunks_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    chunks = data.get('chunks', [])
    logger.info(f"  - {len(chunks)} chunks loaded")
    return chunks


def generate_qa_for_chunk(
    chunk_dict: Dict[str, Any],
    question_gen: QuestionGenerator,
    answer_gen: AnswerGenerator,
    num_qa: int = 3
) -> List[Dict[str, Any]]:
    """Générer des paires QA pour un chunk"""
    
    # Convertir dict en SimpleChunk
    chunk = SimpleChunk(
        chunk_id=chunk_dict['chunk_id'],
        content=chunk_dict['content'],
        semantic_type=chunk_dict['semantic_type'],
        chapter_title=chunk_dict['chapter'],
        section_title=chunk_dict['section'],
        page_range=tuple(chunk_dict['page_range']),
        metadata={'source': chunk_dict.get('source_file', '')}
    )
    
    logger.info(f"  🔹 Chunk: {chunk.chunk_id[:30]}... (type: {chunk.semantic_type})")
    
    # Générer questions
    try:
        questions = question_gen.generate_from_chunk(chunk, num_questions=num_qa)
        logger.info(f"     ✅ {len(questions)} questions generated")
    except Exception as e:
        logger.error(f"     ❌ Question generation failed: {e}")
        return []
    
    # Générer réponses
    qa_pairs = []
    for i, question in enumerate(questions):
        try:
            answer = answer_gen.generate_answer(question, chunk)
            qa_pair = QAPair.from_question_and_answer(question, answer)
            
            qa_pairs.append({
                "qa_id": f"{chunk.chunk_id[:8]}_q{i+1}",
                "chunk_id": chunk.chunk_id,
                "question": qa_pair.question,
                "answer": qa_pair.answer,
                "question_type": qa_pair.question_type,
                "difficulty": qa_pair.difficulty,
                "supporting_quotes": qa_pair.supporting_quotes,
                "confidence": qa_pair.confidence,
                "chunk_content": chunk.content,  # Pour le critic
                "metadata": {
                    "chapter": chunk.chapter_title,
                    "section": chunk.section_title,
                    "page_range": list(chunk.page_range),
                    "semantic_type": chunk.semantic_type
                }
            })
            logger.info(f"     ✅ QA {i+1}/{num_qa} generated")
        except Exception as e:
            logger.error(f"     ❌ Answer generation failed for Q{i+1}: {e}")
            continue
    
    return qa_pairs


def main():
    """Main execution"""
    logger.info("\n" + "=" * 60)
    logger.info("ÉTAPE 2: GÉNÉRATION DE PAIRES QA")
    logger.info("=" * 60)
    
    # Vérifier que chunks.json existe
    project_root = Path(__file__).parent.parent.parent
    chunks_path = project_root / "experiments/critic_v2_baseline/data/chunks.json"
    if not chunks_path.exists():
        logger.error(f"❌ chunks.json not found: {chunks_path}")
        logger.info("Please run 01_extract_chunks.py first")
        sys.exit(1)
    
    # Charger chunks
    chunks = load_chunks(str(chunks_path))
    
    # Setup LLM (chargement direct GGUF)
    logger.info("\n🤖 Loading Qwen2.5:32b directly...")
    model_path = "~/models/qwen2.5-32b-instruct/Qwen2.5-32B-Instruct-Q4_K_M.gguf"
    llm = LLMManager.from_direct_llamacpp(
        model_path=model_path,
        n_gpu_layers=-1,  # All layers on GPU
        n_ctx=4096,
        verbose=False
    )
    logger.info("✅ Model loaded")
    
    # Setup generators
    logger.info("🛠️  Initializing generators...")
    question_gen = QuestionGenerator(
        llm_manager=llm,
        language="fr",
        default_num_questions=3,
        temperature=0.7
    )
    
    answer_gen = AnswerGenerator(
        llm_manager=llm,
        language="fr",
        temperature=0.3
    )
    
    # Générer QA pour chaque chunk
    logger.info(f"\n📝 Generating QA pairs for {len(chunks)} chunks...")
    all_qa = []
    
    for i, chunk_dict in enumerate(chunks):
        logger.info(f"\n[{i+1}/{len(chunks)}] Processing chunk...")
        qa_pairs = generate_qa_for_chunk(chunk_dict, question_gen, answer_gen, num_qa=3)
        all_qa.extend(qa_pairs)
    
    # Sauvegarder
    output_path = project_root / "experiments/critic_v2_baseline/data/qa_samples.json"
    logger.info(f"\n💾 Saving {len(all_qa)} QA pairs to: {output_path}")
    
    with open(str(output_path), 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "num_qa_pairs": len(all_qa),
                "num_chunks": len(chunks),
                "generation_date": "2026-02-09",
                "llm_model": "qwen2.5:32b-instruct-q4_K_M",
                "qa_per_chunk": 3
            },
            "qa_pairs": all_qa
        }, f, indent=2, ensure_ascii=False)
    
    # Résumé
    logger.info("\n" + "=" * 60)
    logger.info("📊 RÉSUMÉ")
    logger.info("=" * 60)
    logger.info(f"Total QA pairs: {len(all_qa)}")
    logger.info(f"Chunks processed: {len(chunks)}")
    logger.info(f"Average per chunk: {len(all_qa)/len(chunks):.1f}")
    
    # Distribution par type de question
    q_types = {}
    for qa in all_qa:
        t = qa['question_type']
        q_types[t] = q_types.get(t, 0) + 1
    
    logger.info("\nQuestion types distribution:")
    for t, count in sorted(q_types.items()):
        logger.info(f"  - {t}: {count}")
    
    logger.info(f"\n✅ Output: {output_path}")
    logger.info("\n➡️  Next: Run 03_run_critic_v2.py")


if __name__ == "__main__":
    main()
