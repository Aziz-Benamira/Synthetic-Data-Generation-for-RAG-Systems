"""Test parser on English textbook (Tipler_Llewellyn.pdf)"""
import sys
sys.path.insert(0, 'src/chunking')
from semantic_chunker import SemanticChunker
import json

# Test sur le livre anglais
pdf_path = 'data/pdfs/Tipler_Llewellyn.pdf'
print(f'Testing: {pdf_path}')
print('='*70)

chunker = SemanticChunker(pdf_path)
print(f'Pages: {chunker.num_pages}')

# Extract TOC
toc = chunker.extract_toc()
chapters = toc.get('chapters', [])
print(f'Chapters found: {len(chapters)}')
print()

# Show first 8 chapters
print("=" * 40)
print("STRUCTURE DETECTED")
print("=" * 40)
for i, ch in enumerate(chapters[:8]):
    print(f"\n  [{ch['id']}] {ch['title']}")
    print(f"      Pages: {ch['page_start']} - {ch['page_end']}")
    sections = ch.get('sections', [])
    if sections:
        print(f"      Sections: {len(sections)} (showing first 3)")
        for s in sections[:3]:
            print(f"        - {s['id']}: {s['title']}")

# Now do chunking
print("\n" + "=" * 40)
print("CHUNKING RESULTS")
print("=" * 40)

chunks = chunker.chunk_document()
print(f"\nChunks created: {len(chunks)}")

# Stats by type
type_counts = {}
for chunk in chunks:
    t = chunk.semantic_type
    type_counts[t] = type_counts.get(t, 0) + 1

print("\nSemantic types:")
for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
    pct = 100 * count / len(chunks)
    print(f"  - {t}: {count} ({pct:.1f}%)")

# Size stats
sizes = [len(c.content) for c in chunks]
print(f"\nChunk sizes:")
print(f"  - Average: {sum(sizes)//len(sizes)} chars")
print(f"  - Min: {min(sizes)}")
print(f"  - Max: {max(sizes)}")

# Show sample chunks
print("\n" + "=" * 40)
print("SAMPLE CHUNKS")
print("=" * 40)

# Find one of each type
shown_types = set()
for chunk in chunks:
    if chunk.semantic_type not in shown_types and len(shown_types) < 3:
        shown_types.add(chunk.semantic_type)
        print(f"\n[{chunk.semantic_type.upper()}] {chunk.chunk_id}")
        print(f"  Chapter: {chunk.chapter_title}")
        print(f"  Section: {chunk.section_title}")
        print(f"  Pages: {chunk.page_range}")
        print(f"  Size: {len(chunk.content)} chars")
        preview = chunk.content[:300].replace('\n', ' ')
        print(f"  Preview: {preview}...")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
