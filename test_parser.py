"""
Test du Parser Intelligent (Semantic Chunker)
Pour validation avant intégration dans le pipeline
"""

from src.chunking.semantic_chunker import SemanticChunker
import json

def test_parser(pdf_path: str):
    print("=" * 70)
    print(f"TEST: {pdf_path}")
    print("=" * 70)
    
    # Initialisation
    chunker = SemanticChunker(pdf_path)
    print(f"Pages dans le PDF: {chunker.num_pages}")
    
    # 1. Test TOC
    print("\n" + "-" * 40)
    print("1. EXTRACTION TOC")
    print("-" * 40)
    
    toc = chunker.extract_toc()
    chapters = toc["chapters"]
    print(f"Chapitres trouvés: {len(chapters)}")
    
    for ch in chapters[:5]:  # Max 5 chapitres
        print(f"\n  [{ch['id']}] {ch['title']}")
        print(f"      Pages: {ch['page_start']} - {ch['page_end']}")
        print(f"      Sections: {len(ch['sections'])}")
        
        for sec in ch["sections"][:3]:  # Max 3 sections par chapitre
            print(f"        - {sec['id']}: {sec['title']}")
    
    # 2. Test Chunking
    print("\n" + "-" * 40)
    print("2. CHUNKING SÉMANTIQUE")
    print("-" * 40)
    
    chunks = chunker.chunk_document()
    print(f"Chunks créés: {len(chunks)}")
    
    # Stats par type sémantique
    types = {}
    for c in chunks:
        types[c.semantic_type] = types.get(c.semantic_type, 0) + 1
    
    print(f"\nTypes sémantiques:")
    for t, count in sorted(types.items(), key=lambda x: -x[1]):
        print(f"  - {t}: {count} ({100*count/len(chunks):.1f}%)")
    
    # Stats de taille
    sizes = [len(c.content) for c in chunks]
    print(f"\nTailles des chunks:")
    print(f"  - Moyenne: {sum(sizes)/len(sizes):.0f} caractères")
    print(f"  - Min: {min(sizes)}")
    print(f"  - Max: {max(sizes)}")
    
    # 3. Exemples de chunks
    print("\n" + "-" * 40)
    print("3. EXEMPLES DE CHUNKS")
    print("-" * 40)
    
    # Trouver un chunk de type "definition" si existe
    def_chunks = [c for c in chunks if c.semantic_type == "definition"]
    example_chunks = [c for c in chunks if c.semantic_type == "example"]
    
    print(f"\n[A] Chunk type 'definition' ({len(def_chunks)} trouvés):")
    if def_chunks:
        sample = def_chunks[0]
        print(f"    ID: {sample.chunk_id}")
        print(f"    Chapter: {sample.chapter_title}")
        print(f"    Section: {sample.section_title}")
        print(f"    Pages: {sample.page_range}")
        print(f"    Taille: {len(sample.content)} chars")
        print(f"    Contenu (aperçu):")
        preview = sample.content[:300].replace('\n', ' ')
        print(f"    '{preview}...'")
    
    print(f"\n[B] Chunk type 'example' ({len(example_chunks)} trouvés):")
    if example_chunks:
        sample = example_chunks[0]
        print(f"    ID: {sample.chunk_id}")
        print(f"    Chapter: {sample.chapter_title}")
        print(f"    Section: {sample.section_title}")
        print(f"    Pages: {sample.page_range}")
        print(f"    Taille: {len(sample.content)} chars")
        print(f"    Contenu (aperçu):")
        preview = sample.content[:300].replace('\n', ' ')
        print(f"    '{preview}...'")
    
    print(f"\n[C] Chunk type 'text' (premier):")
    text_chunks = [c for c in chunks if c.semantic_type == "text"]
    if text_chunks:
        sample = text_chunks[0]
        print(f"    ID: {sample.chunk_id}")
        print(f"    Chapter: {sample.chapter_title}")
        print(f"    Section: {sample.section_title}")
        print(f"    Taille: {len(sample.content)} chars")
        preview = sample.content[:300].replace('\n', ' ')
        print(f"    Contenu (aperçu):")
        print(f"    '{preview}...'")
    
    # 4. Format de sortie
    print("\n" + "-" * 40)
    print("4. FORMAT DE SORTIE (JSON)")
    print("-" * 40)
    
    if chunks:
        sample = chunks[5] if len(chunks) > 5 else chunks[0]
        output = {
            "chunk_id": sample.chunk_id,
            "content": sample.content[:200] + "...",
            "chapter": sample.chapter_title,
            "section": sample.section_title,
            "subsection": sample.subsection_title,
            "pages": list(sample.page_range),
            "semantic_type": sample.semantic_type,
            "size": len(sample.content)
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 70)
    print("FIN DU TEST")
    print("=" * 70)
    
    return chunks


if __name__ == "__main__":
    # Test sur le PDF français
    chunks = test_parser("data/pdfs/M2_cours.pdf")
