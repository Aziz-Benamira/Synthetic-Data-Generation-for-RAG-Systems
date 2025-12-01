"""
Test Script for PDF Processor
==============================

This script tests the PDF processor on a sample textbook.
We'll verify that we can:
1. Load a PDF file
2. Extract table of contents
3. Extract text from specific pages
4. Find definitions and equations
5. Search for keywords

Before running:
- Place a PDF file in: generation/multi-agent/data/sample_textbook.pdf
- Or update the PDF_PATH variable below
"""

import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pdf_processor import PDFProcessor


# Configuration
PDF_PATH = "data/sample_textbook.pdf"


def test_pdf_loading():
    """Test 1: Can we load the PDF?"""
    print("\n" + "="*60)
    print("TEST 1: Loading PDF")
    print("="*60)
    
    try:
        processor = PDFProcessor(PDF_PATH)
        print(f"SUCCESS: Loaded PDF from {PDF_PATH}")
        print(f"  Pages: {processor.num_pages}")
        print(f"  Size: {processor.doc.metadata.get('format', 'Unknown')}")
        processor.close()
        return True
    except FileNotFoundError:
        print(f"ERROR: PDF not found at {PDF_PATH}")
        print("\nPlease add a PDF file to test with:")
        print(f"  1. Place your PDF at: {Path(PDF_PATH).absolute()}")
        print(f"  2. Or update PDF_PATH in this script")
        return False
    except Exception as e:
        print(f"ERROR: Failed to load PDF: {e}")
        return False


def test_curriculum_extraction():
    """Test 2: Can we extract the table of contents?"""
    print("\n" + "="*60)
    print("TEST 2: Extracting Curriculum Structure")
    print("="*60)
    
    try:
        with PDFProcessor(PDF_PATH) as processor:
            curriculum = processor.extract_curriculum_structure()
            
            num_chapters = len(curriculum.get('chapters', []))
            print(f"SUCCESS: Extracted curriculum")
            print(f"  Chapters: {num_chapters}")
            
            # Show first 3 chapters
            for i, chapter in enumerate(curriculum['chapters'][:3]):
                print(f"\n  Chapter {i+1}:")
                print(f"    ID: {chapter['id']}")
                print(f"    Title: {chapter['title']}")
                print(f"    Pages: {chapter['pages'][0]} - {chapter['pages'][1]}")
                print(f"    Sections: {len(chapter.get('sections', []))}")
            
            if num_chapters > 3:
                print(f"\n  ... and {num_chapters - 3} more chapters")
            
            return True
            
    except Exception as e:
        print(f"ERROR: Failed to extract curriculum: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_text_extraction():
    """Test 3: Can we extract text from pages?"""
    print("\n" + "="*60)
    print("TEST 3: Extracting Text")
    print("="*60)
    
    try:
        with PDFProcessor(PDF_PATH) as processor:
            # Extract first page
            page1_text = processor.extract_text_from_page(1)
            print(f"SUCCESS: Extracted text from page 1")
            print(f"  Length: {len(page1_text)} characters")
            print(f"  Preview (first 200 chars):")
            print(f"  {page1_text[:200]}")
            
            # Extract a range
            print(f"\nExtracting pages 1-3...")
            range_text = processor.extract_text_range(1, 3)
            print(f"SUCCESS: Extracted text from pages 1-3")
            print(f"  Total length: {len(range_text)} characters")
            
            return True
            
    except Exception as e:
        print(f"ERROR: Failed to extract text: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_definition_extraction():
    """Test 4: Can we find definitions?"""
    print("\n" + "="*60)
    print("TEST 4: Extracting Definitions")
    print("="*60)
    
    try:
        with PDFProcessor(PDF_PATH) as processor:
            # Get text from first chapter (pages 1-10)
            text = processor.extract_text_range(1, min(10, processor.num_pages))
            
            definitions = processor.extract_definitions(text)
            print(f"SUCCESS: Found {len(definitions)} definitions")
            
            # Show first 3
            for i, definition in enumerate(definitions[:3]):
                print(f"\n  Definition {i+1}:")
                print(f"    {definition[:150]}...")
            
            return True
            
    except Exception as e:
        print(f"ERROR: Failed to extract definitions: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_equation_extraction():
    """Test 5: Can we find equations?"""
    print("\n" + "="*60)
    print("TEST 5: Extracting Equations")
    print("="*60)
    
    try:
        with PDFProcessor(PDF_PATH) as processor:
            # Get text from first chapter
            text = processor.extract_text_range(1, min(10, processor.num_pages))
            
            equations = processor.extract_equations(text)
            print(f"SUCCESS: Found {len(equations)} equations")
            
            # Show first 5
            for i, equation in enumerate(equations[:5]):
                print(f"  {i+1}. {equation}")
            
            return True
            
    except Exception as e:
        print(f"ERROR: Failed to extract equations: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_keyword_search():
    """Test 6: Can we search for keywords?"""
    print("\n" + "="*60)
    print("TEST 6: Keyword Search")
    print("="*60)
    
    # You can change this keyword to something relevant to your PDF
    SEARCH_KEYWORD = "algorithm"
    
    try:
        with PDFProcessor(PDF_PATH) as processor:
            matches = processor.find_exact_matches(
                SEARCH_KEYWORD, 
                context_window=100
            )
            
            print(f"SUCCESS: Found {len(matches)} matches for '{SEARCH_KEYWORD}'")
            
            # Show first 2 matches
            for i, match in enumerate(matches[:2]):
                print(f"\n  Match {i+1}:")
                print(f"    Page: {match['page_number']}")
                print(f"    Context: ...{match['full_context'][:150]}...")
            
            return True
            
    except Exception as e:
        print(f"ERROR: Failed to search: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests in sequence"""
    print("\n")
    print("#" * 60)
    print("#  PDF PROCESSOR TEST SUITE")
    print("#" * 60)
    
    results = []
    
    # Test 1: Loading
    results.append(("Loading PDF", test_pdf_loading()))
    
    # Only continue if loading succeeded
    if not results[0][1]:
        print("\n" + "="*60)
        print("STOPPING: Cannot proceed without a valid PDF")
        print("="*60)
        return
    
    # Run remaining tests
    results.append(("Curriculum Extraction", test_curriculum_extraction()))
    results.append(("Text Extraction", test_text_extraction()))
    results.append(("Definition Extraction", test_definition_extraction()))
    results.append(("Equation Extraction", test_equation_extraction()))
    results.append(("Keyword Search", test_keyword_search()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {test_name}: {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nAll tests passed! PDF processor is working correctly.")
        print("Ready to move to Phase 1, Task 2: Chunking")
    else:
        print("\nSome tests failed. Review errors above.")


if __name__ == "__main__":
    run_all_tests()
