"""
Analyse du Chunking Intelligent
================================

Vérifie si le chunking sémantique avec TOC awareness fonctionne vraiment.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src' / 'chunking'))

from semantic_chunker import SemanticChunker

print("=" * 70)
print("ANALYSE DU CHUNKING INTELLIGENT (TOC-AWARE)")
print("=" * 70)
print()

pdf_path = "data/pdfs/M2_cours.pdf"
print(f"📄 PDF: {pdf_path}")
print()

# Parse with TOC detection
chunker = SemanticChunker(pdf_path)
chunks = chunker.chunk_document()

print(f"✅ {len(chunks)} chunks extraits")
print()

# Analyze TOC structure
print("=" * 70)
print("1. DÉTECTION DE LA TABLE DES MATIÈRES")
print("=" * 70)
print()

# Extract TOC using the method
toc_data = chunker.extract_toc()

if toc_data and len(toc_data.get('chapters', [])) > 0:
    chapters_list = toc_data['chapters']
    print(f"✅ TOC détectée: {len(chapters_list)} chapitres")
    print()
if toc_data and len(toc_data.get('chapters', [])) > 0:
    chapters_list = toc_data['chapters']
    print(f"✅ TOC détectée: {len(chapters_list)} chapitres")
    print()
    print("Hiérarchie détectée:")
    for i, chapter in enumerate(chapters_list[:10], 1):  # Show first 10
        title = chapter.get('title', 'Untitled')
        page = chapter.get('page', '?')
        sections = chapter.get('sections', [])
        print(f"  {i}. {title} (page {page}) - {len(sections)} sections")
        for j, section in enumerate(sections[:3], 1):
            print(f"     {i}.{j}. {section.get('title', 'Untitled')} (page {section.get('page', '?')})")
        if len(sections) > 3:
            print(f"     ... et {len(sections) - 3} autres sections")
    
    if len(chapters_list) > 10:
        print(f"   ... et {len(chapters_list) - 10} autres chapitres")
else:
    print("❌ TOC NON DÉTECTÉE")
    print("   Le chunking n'utilise PAS la structure hiérarchique du document!")

print()
print("=" * 70)
print("2. ANALYSE DES CHUNKS (Échantillon)")
print("=" * 70)
print()

# Show detailed info for first 5 chunks
for i, chunk in enumerate(chunks[:5], 1):
    print(f"Chunk {i}: {chunk.chunk_id}")
    print(f"  Type: {chunk.semantic_type}")
    print(f"  Pages: {chunk.page_range}")
    print(f"  Chapter: {chunk.chapter_title or 'N/A'}")
    print(f"  Section: {chunk.section_title or 'N/A'}")
    print(f"  Taille: {len(chunk.content)} chars")
    print(f"  Contenu (100 premiers chars):")
    print(f"    {chunk.content[:100].replace(chr(10), ' ')}...")
    print()

# Analyze semantic types
print("=" * 70)
print("3. DISTRIBUTION DES TYPES SÉMANTIQUES")
print("=" * 70)
print()

type_counts = {}
for chunk in chunks:
    t = chunk.semantic_type
    type_counts[t] = type_counts.get(t, 0) + 1

for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
    percentage = (count / len(chunks)) * 100
    bar = "█" * int(percentage / 2)
    print(f"  {t:15s} │ {bar} {count:3d} ({percentage:5.1f}%)")

print()

# Check chapter/section coverage
print("=" * 70)
print("4. COUVERTURE HIÉRARCHIQUE (Chapitres/Sections)")
print("=" * 70)
print()

chapters = set()
sections = set()

for chunk in chunks:
    if chunk.chapter_title:
        chapters.add(chunk.chapter_title)
    if chunk.section_title:
        sections.add((chunk.chapter_title, chunk.section_title))

print(f"Chapitres uniques: {len(chapters)}")
print(f"Sections uniques: {len(sections)}")
print()

if len(chapters) > 0:
    print("✅ Le chunking UTILISE la hiérarchie du document")
    print()
    print("Chapitres détectés:")
    for chapter in sorted(list(chapters)[:10]):
        section_count = len([s for s in sections if s[0] == chapter])
        print(f"  • {chapter} ({section_count} sections)")
else:
    print("❌ Le chunking N'UTILISE PAS la hiérarchie du document")
    print("   Les chunks sont créés sans contexte structurel")

print()

# Quality check
print("=" * 70)
print("5. QUALITÉ DU CHUNKING")
print("=" * 70)
print()

# Check if chunks have context
chunks_with_metadata = sum(1 for c in chunks if c.chapter_title and c.section_title)
metadata_percentage = (chunks_with_metadata / len(chunks)) * 100

print(f"Chunks avec métadonnées complètes: {chunks_with_metadata}/{len(chunks)} ({metadata_percentage:.1f}%)")
print()

if metadata_percentage > 80:
    print("✅ EXCELLENT - La majorité des chunks ont un contexte hiérarchique")
elif metadata_percentage > 50:
    print("⚠️  MOYEN - Certains chunks manquent de contexte")
else:
    print("❌ PROBLÈME - Peu de chunks ont un contexte hiérarchique")
    print("   Le TOC awareness ne fonctionne pas correctement")

print()

# Hierarchical structure check
print("=" * 70)
print("6. STRUCTURE HIÉRARCHIQUE POUR RETRIEVAL")
print("=" * 70)
print()

print("Concept: Hierarchical Retrieval with Compression")
print()
print("Au lieu de: [Flat embedding space avec 50k chunks]")
print("            recherche linéaire O(n)")
print()
print("Proposé:    Encyclopedia")
print("             ├─ Chapter 1 (résumé)")
print("             │  ├─ Section 1.1 (résumé)")
print("             │  │  └─ Paragraphe 1.1.1")
print("             │  └─ Section 1.2")
print("             └─ Chapter 2")
print()
print("Recherche: O(log n) avec ~200 comparaisons au lieu de 50k")
print()

if len(chapters) > 0 and len(sections) > 0:
    print(f"✅ Structure hiérarchique disponible:")
    print(f"   Niveau 1 (Chapitres): {len(chapters)} nœuds")
    print(f"   Niveau 2 (Sections): {len(sections)} nœuds")
    print(f"   Niveau 3 (Chunks): {len(chunks)} nœuds")
    print()
    print("   Cette structure PEUT être utilisée pour Hierarchical Retrieval!")
else:
    print("❌ Structure hiérarchique MANQUANTE")
    print("   Impossible de faire du Hierarchical Retrieval sans hiérarchie")

print()
print("=" * 70)
print("CONCLUSION")
print("=" * 70)
print()

toc_detected = toc_data and len(toc_data.get('chapters', [])) > 0

if toc_detected and len(chapters) > 0 and metadata_percentage > 50:
    print("✅ Le chunking TOC-aware FONCTIONNE")
    print(f"   • TOC détectée: {len(toc_data.get('chapters', []))} chapitres")
    print(f"   • Hiérarchie préservée: {len(chapters)} chapitres, {len(sections)} sections")
    print(f"   • Métadonnées: {metadata_percentage:.1f}% des chunks")
    print()
    print("   Mais peut être AMÉLIORÉ avec:")
    print("   1. Hierarchical summarization (résumés progressifs)")
    print("   2. Tree-based indexing au lieu de flat embeddings")
    print("   3. Compression à chaque niveau de la hiérarchie")
else:
    print("⚠️  Le chunking TOC-aware pourrait être AMÉLIORÉ")
    print()
    print("   Problèmes potentiels:")
    if not toc_detected:
        print("   • TOC non détectée")
    if len(chapters) == 0:
        print("   • Hiérarchie non préservée dans les chunks")
    if metadata_percentage < 50:
        print("   • Métadonnées manquantes")
