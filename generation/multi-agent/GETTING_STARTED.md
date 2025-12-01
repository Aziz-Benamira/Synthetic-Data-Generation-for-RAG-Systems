# Getting Started - Phase 1: PDF Processing

## What We're Building

A multi-agent system that generates high-quality question-answer pairs from textbooks.

Phase 1 focuses on getting the data pipeline working:
- Load and parse PDF textbooks
- Extract structure (chapters, sections)
- Extract text content
- Prepare data for later processing

## Prerequisites

1. **Python Environment**
   ```bash
   # Make sure you're in the multi-agent directory
   cd generation/multi-agent
   
   # Install dependencies (if not already done)
   pip install pymupdf python-dotenv pydantic
   ```

2. **PDF Textbook**
   - Download a free textbook PDF (see data/README.md for sources)
   - Place it at: `data/sample_textbook.pdf`
   - Recommended: OpenStax Physics or Biology textbook

## Step 1: Test PDF Processor

Run the test script to verify PDF processing works:

```bash
cd generation/multi-agent
python scripts/test_pdf_processor.py
```

### Expected Output

If everything works, you should see:
```
TEST 1: Loading PDF
SUCCESS: Loaded PDF
  Pages: 450

TEST 2: Extracting Curriculum Structure
SUCCESS: Extracted curriculum
  Chapters: 15
  Chapter 1: Classical Mechanics
    Pages: 1 - 35
    Sections: 4

... (more tests)

TEST SUMMARY
  ✓ Loading PDF: PASS
  ✓ Curriculum Extraction: PASS
  ✓ Text Extraction: PASS
  ✓ Definition Extraction: PASS
  ✓ Equation Extraction: PASS
  ✓ Keyword Search: PASS

Total: 6/6 tests passed
```

### Troubleshooting

**Error: PDF not found**
- Make sure PDF is at `data/sample_textbook.pdf`
- Or update `PDF_PATH` in the test script

**Error: Module not found**
- Run: `pip install pymupdf`
- Check you're in the right directory

**Text extraction looks wrong**
- Some PDFs have weird formatting
- Try a different PDF (OpenStax textbooks work well)

## Step 2: Understand the Code

Once tests pass, let's understand what the PDFProcessor does:

### Key Methods

1. **`extract_curriculum_structure()`**
   - Extracts table of contents
   - Identifies chapters and sections
   - Returns hierarchical structure
   
   Purpose: We need to know the document structure to generate questions systematically

2. **`extract_text_from_page(page_number)`**
   - Gets text from a single page
   - Caches results for efficiency
   
   Purpose: Basic building block for all text extraction

3. **`extract_text_range(start_page, end_page)`**
   - Gets text from multiple pages
   - Combines into single string
   
   Purpose: Extract a full chapter or section

4. **`extract_definitions(text)`**
   - Finds definition sentences using patterns
   - Returns list of definitions
   
   Purpose: Definitions make good question targets

5. **`extract_equations(text)`**
   - Finds mathematical equations
   - Returns list of equations
   
   Purpose: Equations can be used for calculation questions

6. **`find_exact_matches(query, context_window)`**
   - Searches for keywords across document
   - Returns matches with surrounding context
   
   Purpose: Helps verify citations and find relevant passages

### How It Works

Algorithm for TOC extraction:
```
1. Try to use PDF's built-in table of contents
   - Most properly formatted PDFs have this
   - PyMuPDF can extract it directly

2. If no built-in TOC, fall back to pattern matching
   - Look for "Chapter X: Title" patterns
   - Look for "X.Y Section Title" patterns
   - Scan first 20 pages (TOC usually at start)

3. If still nothing found, create default structure
   - Treat entire document as one chapter
   - This ensures we always have some structure
```

## Step 3: Experiment

Try these experiments to understand better:

### Experiment 1: Extract a Chapter

Create a file `experiments/extract_chapter.py`:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pdf_processor import PDFProcessor

# Load PDF
with PDFProcessor("data/sample_textbook.pdf") as processor:
    # Get curriculum
    curriculum = processor.extract_curriculum_structure()
    
    # Get first chapter
    first_chapter = curriculum['chapters'][0]
    print(f"Chapter: {first_chapter['title']}")
    print(f"Pages: {first_chapter['pages']}")
    
    # Extract text
    start_page, end_page = first_chapter['pages']
    chapter_text = processor.extract_text_range(start_page, end_page)
    
    print(f"\nText length: {len(chapter_text)} characters")
    print(f"\nFirst 500 characters:")
    print(chapter_text[:500])
```

Run it:
```bash
python experiments/extract_chapter.py
```

### Experiment 2: Find Key Concepts

```python
# Same imports as above

with PDFProcessor("data/sample_textbook.pdf") as processor:
    # Extract first 20 pages
    text = processor.extract_text_range(1, 20)
    
    # Find definitions
    definitions = processor.extract_definitions(text)
    print(f"Found {len(definitions)} definitions:\n")
    for i, defn in enumerate(definitions[:5], 1):
        print(f"{i}. {defn}\n")
    
    # Find equations  
    equations = processor.extract_equations(text)
    print(f"\nFound {len(equations)} equations:\n")
    for i, eq in enumerate(equations[:5], 1):
        print(f"{i}. {eq}")
```

### Experiment 3: Search for Terms

```python
# Same imports

with PDFProcessor("data/sample_textbook.pdf") as processor:
    # Search for a term (change to match your PDF content)
    matches = processor.find_exact_matches("energy", context_window=150)
    
    print(f"Found '{energy}' on {len(matches)} pages\n")
    
    # Show first 3 matches
    for i, match in enumerate(matches[:3], 1):
        print(f"Match {i} (page {match['page_number']}):")
        print(match['full_context'])
        print("\n" + "-"*60 + "\n")
```

## Step 4: Questions to Consider

Before moving to Phase 1, Task 2 (Chunking), think about:

1. **Document Structure**
   - How many chapters does your PDF have?
   - Are sections clearly identified?
   - Will this structure help organize questions?

2. **Text Quality**
   - Is extracted text clean and readable?
   - Are equations preserved correctly?
   - Any formatting issues?

3. **Content Understanding**
   - Can you identify good question targets (definitions, key concepts)?
   - Are there obvious chunks (sections, subsections)?
   - What types of questions would this content support?

## Next Steps

Once you understand PDF processing and tests pass:
1. Review the implementation plan: `IMPLEMENTATION_PLAN.md`
2. Move to Phase 1, Task 2: Implementing chunking
3. We'll break documents into smaller pieces for the vector store

## Learning Resources

- **PyMuPDF Docs**: https://pymupdf.readthedocs.io/
- **Regular Expressions**: https://docs.python.org/3/library/re.html
- **Pydantic Models**: https://docs.pydantic.dev/

## Questions?

If anything is unclear:
1. Run the test script and share the output
2. Try the experiments above
3. Ask specific questions about what you don't understand

The goal is to deeply understand each component before moving forward.
