#!/usr/bin/env python3
"""
Étape 1: Extraction de chunks variés du document M2
====================================================

Extrait 8 chunks représentatifs de différents types de contenu :
- Définitions théoriques
- Formules mathématiques  
- Théorèmes avec preuves
- Exemples/Applications
- Procédures/Méthodes

Output: data/chunks.json
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Imports
try:
    from src.chunking.semantic_chunker import SemanticChunker
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info("Make sure you're in the project root directory")
    sys.exit(1)


def extract_diverse_chunks(pdf_path: str, num_chunks: int = 8) -> List[Dict[str, Any]]:
    """
    Extrait des chunks variés du PDF en utilisant le SemanticChunker.
    
    Args:
        pdf_path: Chemin vers le PDF
        num_chunks: Nombre de chunks à sélectionner (par défaut 8)
        
    Returns:
        Liste de dictionnaires représentant les chunks
    """
    logger.info(f"📄 Loading PDF: {pdf_path}")
    
    # Initialiser le chunker
    chunker = SemanticChunker(
        pdf_path=pdf_path,
        target_chunk_size=1000,
        max_chunk_size=2000,
        chunk_overlap=200,
        min_chunk_size=300
    )
    
    # Chunker le document
    logger.info("🔪 Chunking document with semantic boundaries...")
    all_chunks = chunker.chunk_document()
    
    logger.info(f"  - {len(all_chunks)} total chunks extracted")
    
    # Sélectionner des chunks diversifiés
    # Stratégie: prendre des chunks de différents types sémantiques
    selected = []
    
    # Grouper par type sémantique
    by_type = {}
    for chunk in all_chunks:
        t = chunk.semantic_type
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(chunk)
    
    logger.info(f"  - Semantic types found: {list(by_type.keys())}")
    
    # Sélectionner de manière équilibrée
    chunks_per_type = max(1, num_chunks // len(by_type))
    
    for sem_type, chunks in by_type.items():
        # Filtrer les chunks trop courts
        valid_chunks = [c for c in chunks if len(c.content) >= 300]
        
        if not valid_chunks:
            continue
        
        # Prendre un échantillon
        import random
        sample_size = min(chunks_per_type, len(valid_chunks))
        sampled = random.sample(valid_chunks, sample_size)
        selected.extend(sampled)
        
        if len(selected) >= num_chunks:
            break
    
    # Limiter au nombre demandé
    selected = selected[:num_chunks]
    
    logger.info(f"  - {len(selected)} diverse chunks selected")
    
    # Convertir en dictionnaires JSON-sérialisables
    chunks_dict = []
    for i, chunk in enumerate(selected):
        chunks_dict.append({
            "chunk_id": chunk.chunk_id,
            "content": chunk.content,
            "chapter": chunk.chapter_title,
            "section": chunk.section_title,
            "subsection": chunk.subsection_title,
            "page_range": list(chunk.page_range),
            "semantic_type": chunk.semantic_type,
            "source_file": pdf_path,
            "metadata": chunk.metadata
        })
    
    return chunks_dict


def save_chunks(chunks: List[Dict[str, Any]], output_path: str):
    """Sauvegarder les chunks au format JSON"""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "num_chunks": len(chunks),
                "extraction_date": "2026-02-09",
                "source": "tipler_mosca_chapitre_m2.pdf"
            },
            "chunks": chunks
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Saved to: {output_path}")


def main():
    """Main execution"""
    logger.info("=" * 60)
    logger.info("ÉTAPE 1: EXTRACTION DE CHUNKS")
    logger.info("=" * 60)
    
    # Vérifier que le PDF existe (chemin depuis project root)
    project_root = Path(__file__).parent.parent.parent
    pdf_path = project_root / "data/pdfs/M2_cours.pdf"
    
    if not pdf_path.exists():
        logger.error(f"❌ PDF not found: {pdf_path}")
        logger.info("Please ensure the M2 PDF is in data/pdfs/")
        sys.exit(1)
    
    # Créer les dossiers (chemins depuis project root)
    data_dir = project_root / "experiments/critic_v2_baseline/data"
    logs_dir = project_root / "experiments/critic_v2_baseline/logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Extraire
    chunks = extract_diverse_chunks(str(pdf_path), num_chunks=8)
    
    # Sauvegarder
    output_path = data_dir / "chunks.json"
    save_chunks(chunks, str(output_path))
    
    # Résumé
    logger.info("\n" + "=" * 60)
    logger.info("📊 RÉSUMÉ")
    logger.info("=" * 60)
    logger.info(f"Chunks extraits: {len(chunks)}")
    
    types_count = {}
    for chunk in chunks:
        t = chunk['semantic_type']
        types_count[t] = types_count.get(t, 0) + 1
    
    logger.info("Distribution par type:")
    for t, count in types_count.items():
        logger.info(f"  - {t}: {count}")
    
    logger.info(f"\n✅ Output: {output_path}")
    logger.info("\n➡️  Next: Run 02_generate_qa_samples.py")


if __name__ == "__main__":
    main()
