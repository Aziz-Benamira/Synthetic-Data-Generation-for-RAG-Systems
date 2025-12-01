"""
PDF Processor
=============
Extract text, structure, and metadata from PDF textbooks.

Features:
- Extract table of contents
- Extract text by page range
- Find definitions and equations
- Exact keyword matching
"""

from typing import List, Dict, Any, Optional, Tuple
import re
from pathlib import Path
import fitz  # PyMuPDF


class PDFProcessor:
    """Process PDF textbooks and extract structured information"""
    
    def __init__(self, pdf_path: str):
        """
        Initialize PDF processor.
        
        Args:
            pdf_path: Path to PDF file
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        self.doc = fitz.open(str(self.pdf_path))
        self.num_pages = len(self.doc)
        
        # Cache for extracted text
        self._page_cache = {}
    
    def extract_curriculum_structure(self) -> Dict[str, Any]:
        """
        Extract table of contents from PDF.
        
        Returns:
            {
                "chapters": [
                    {
                        "id": "ch1",
                        "title": "Classical Mechanics",
                        "pages": [1, 45],
                        "sections": [...]
                    }
                ]
            }
        
        Algorithm:
            1. Use PDF's built-in TOC if available
            2. Fall back to pattern matching on headings
            3. Identify chapter/section hierarchy
        """
        # Try built-in TOC first
        toc = self.doc.get_toc()
        
        if toc:
            return self._parse_toc(toc)
        else:
            # Fallback: pattern-based extraction
            return self._extract_toc_from_text()
    
    def _parse_toc(self, toc: List) -> Dict[str, Any]:
        """Parse PyMuPDF TOC format"""
        chapters = []
        current_chapter = None
        
        for level, title, page in toc:
            if level == 1:  # Chapter
                if current_chapter:
                    chapters.append(current_chapter)
                
                chapter_num = len(chapters) + 1
                current_chapter = {
                    "id": f"ch{chapter_num}",
                    "title": title,
                    "pages": [page, None],  # Will update end page
                    "sections": []
                }
            
            elif level == 2 and current_chapter:  # Section
                section_num = len(current_chapter["sections"]) + 1
                section_id = f"{len(chapters) + 1}.{section_num}"
                
                current_chapter["sections"].append({
                    "id": section_id,
                    "title": title,
                    "pages": [page, None]  # Will update end page
                })
        
        if current_chapter:
            chapters.append(current_chapter)
        
        # Update end pages
        for i, chapter in enumerate(chapters):
            if i + 1 < len(chapters):
                chapter["pages"][1] = chapters[i + 1]["pages"][0] - 1
            else:
                chapter["pages"][1] = self.num_pages
            
            # Update section end pages
            for j, section in enumerate(chapter["sections"]):
                if j + 1 < len(chapter["sections"]):
                    section["pages"][1] = chapter["sections"][j + 1]["pages"][0] - 1
                else:
                    section["pages"][1] = chapter["pages"][1]
        
        return {"chapters": chapters}
    
    def _extract_toc_from_text(self) -> Dict[str, Any]:
        """
        Extract TOC by pattern matching (fallback method).
        Looks for patterns like:
        - Chapter 1: Title
        - 1.1 Section Title
        """
        chapters = []
        
        # Patterns for chapters and sections
        chapter_pattern = re.compile(r'^Chapter\s+(\d+)[:\.]?\s+(.+)$', re.MULTILINE | re.IGNORECASE)
        section_pattern = re.compile(r'^(\d+)\.(\d+)\s+(.+)$', re.MULTILINE)
        
        # Scan first 20 pages for TOC
        toc_text = ""
        for page_num in range(min(20, self.num_pages)):
            toc_text += self.extract_text_from_page(page_num + 1)
        
        # Find chapters
        for match in chapter_pattern.finditer(toc_text):
            chapter_num = int(match.group(1))
            chapter_title = match.group(2).strip()
            
            chapters.append({
                "id": f"ch{chapter_num}",
                "title": chapter_title,
                "pages": [1, self.num_pages],  # Placeholder
                "sections": []
            })
        
        # If no chapters found, create a default structure
        if not chapters:
            chapters = [{
                "id": "ch1",
                "title": "Full Document",
                "pages": [1, self.num_pages],
                "sections": [{
                    "id": "1.1",
                    "title": "Content",
                    "pages": [1, self.num_pages]
                }]
            }]
        
        return {"chapters": chapters}
    
    def extract_text_from_page(self, page_number: int) -> str:
        """
        Extract text from a single page.
        
        Args:
            page_number: Page number (1-indexed)
        
        Returns:
            Text content of the page
        """
        if page_number in self._page_cache:
            return self._page_cache[page_number]
        
        if page_number < 1 or page_number > self.num_pages:
            return ""
        
        page = self.doc[page_number - 1]  # PyMuPDF uses 0-indexing
        text = page.get_text()
        
        self._page_cache[page_number] = text
        return text
    
    def extract_text_range(self, start_page: int, end_page: int) -> str:
        """
        Extract text from a page range.
        
        Args:
            start_page: Starting page (1-indexed, inclusive)
            end_page: Ending page (1-indexed, inclusive)
        
        Returns:
            Combined text from all pages
        """
        texts = []
        for page_num in range(start_page, end_page + 1):
            texts.append(self.extract_text_from_page(page_num))
        
        return "\n\n".join(texts)
    
    def extract_definitions(self, text: str) -> List[str]:
        """
        Extract key definitions from text.
        
        Looks for patterns like:
        - "X is defined as..."
        - "Definition: X is..."
        - "X: a thing that..."
        
        Args:
            text: Text to analyze
        
        Returns:
            List of definition sentences
        """
        definitions = []
        
        # Pattern 1: "X is defined as"
        pattern1 = re.compile(r'([A-Z][a-z\s]+)\s+is\s+defined\s+as\s+([^.]+\.)', re.MULTILINE)
        definitions.extend([m.group(0) for m in pattern1.finditer(text)])
        
        # Pattern 2: "Definition:"
        pattern2 = re.compile(r'Definition:\s*([^.]+\.)', re.IGNORECASE)
        definitions.extend([m.group(0) for m in pattern2.finditer(text)])
        
        # Pattern 3: Bold term followed by colon or dash (common in textbooks)
        # This would require parsing formatting, simplified here
        
        return definitions[:10]  # Return top 10
    
    def extract_equations(self, text: str) -> List[str]:
        """
        Extract mathematical equations from text.
        
        Looks for patterns like:
        - F = ma
        - E = mc²
        - x = (-b ± √(b²-4ac)) / 2a
        
        Args:
            text: Text to analyze
        
        Returns:
            List of equations
        """
        equations = []
        
        # Pattern: variable = expression
        pattern = re.compile(r'([A-Za-z][\w]*)\s*=\s*([^,\n.]+)', re.MULTILINE)
        
        for match in pattern.finditer(text):
            equation = match.group(0).strip()
            # Filter out common false positives
            if len(equation) < 50 and '=' in equation:
                equations.append(equation)
        
        return equations[:15]  # Return top 15
    
    def find_exact_matches(
        self, 
        query: str, 
        context_window: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Find exact keyword matches across the document.
        
        Args:
            query: Keyword or phrase to search
            context_window: Characters before/after match
        
        Returns:
            List of matches with context
        """
        matches = []
        query_lower = query.lower()
        
        for page_num in range(1, self.num_pages + 1):
            page_text = self.extract_text_from_page(page_num)
            page_text_lower = page_text.lower()
            
            # Find all occurrences on this page
            start = 0
            while True:
                pos = page_text_lower.find(query_lower, start)
                if pos == -1:
                    break
                
                # Extract context
                context_start = max(0, pos - context_window)
                context_end = min(len(page_text), pos + len(query) + context_window)
                
                context_before = page_text[context_start:pos]
                matched_text = page_text[pos:pos + len(query)]
                context_after = page_text[pos + len(query):context_end]
                
                matches.append({
                    "text": matched_text,
                    "page_number": page_num,
                    "context_before": context_before,
                    "context_after": context_after,
                    "full_context": page_text[context_start:context_end]
                })
                
                start = pos + 1
        
        return matches
    
    def get_page_dimensions(self, page_number: int) -> Tuple[float, float]:
        """Get page width and height"""
        page = self.doc[page_number - 1]
        rect = page.rect
        return (rect.width, rect.height)
    
    def close(self):
        """Close the PDF document"""
        self.doc.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Example usage
if __name__ == "__main__":
    # Process a PDF
    with PDFProcessor("data/sample_textbook.pdf") as processor:
        # Extract TOC
        curriculum = processor.extract_curriculum_structure()
        print("📚 Curriculum Structure:")
        print(f"  Chapters: {len(curriculum['chapters'])}")
        
        # Extract a section
        section_text = processor.extract_text_range(1, 5)
        print(f"\n📄 Pages 1-5: {len(section_text)} characters")
        
        # Find definitions
        definitions = processor.extract_definitions(section_text)
        print(f"\n📖 Definitions found: {len(definitions)}")
        
        # Find equations
        equations = processor.extract_equations(section_text)
        print(f"\n🔢 Equations found: {len(equations)}")
        
        # Keyword search
        matches = processor.find_exact_matches("Newton's law", context_window=100)
        print(f"\n🔍 Keyword matches: {len(matches)}")
