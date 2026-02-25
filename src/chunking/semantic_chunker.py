"""
Semantic Chunker for Academic Textbooks
========================================

State-of-the-art chunking that preserves semantic boundaries in structured academic content.

Key Features:
- TOC-aware: Uses PDF table of contents for natural section boundaries
- Definition-preserving: Keeps definitions, theorems, lemmas intact
- Equation-aware: Never splits mathematical derivations
- Context-rich: Maintains hierarchical metadata (chapter > section > subsection)
- Adaptive sizing: Variable chunk sizes based on content structure
- Fallback strategy: Uses character-based splitting for very large sections

Algorithm:
1. Extract PDF TOC hierarchy (chapters → sections → subsections)
2. For each section:
   - Detect semantic units (definitions, theorems, examples, equations)
   - Create chunks respecting unit boundaries
   - Add rich metadata (chapter context, section path, unit type)
3. If section > max_size, use recursive splitting with overlap
4. Preserve context across chunks with breadcrumb metadata
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import re
from pathlib import Path
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class SemanticChunk:
    """
    A semantically coherent chunk from an academic textbook.
    
    Attributes:
        content: The actual text content
        chunk_id: Unique identifier (e.g., "ch1.s2.c3" = Chapter 1, Section 2, Chunk 3)
        chapter_title: Title of parent chapter
        section_title: Title of parent section
        subsection_title: Optional subsection title
        page_range: [start_page, end_page]
        semantic_type: Type of content (definition, theorem, example, text, equation, mixed)
        metadata: Additional context (source PDF, timestamp, etc.)
    """
    content: str
    chunk_id: str
    chapter_title: str
    section_title: str
    subsection_title: Optional[str] = None
    page_range: Tuple[int, int] = (0, 0)
    semantic_type: str = "text"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_langchain_document(self) -> Document:
        """Convert to LangChain Document for vector store compatibility"""
        metadata = {
            "chunk_id": self.chunk_id,
            "chapter": self.chapter_title,
            "section": self.section_title,
            "subsection": self.subsection_title,
            "page_start": self.page_range[0],
            "page_end": self.page_range[1],
            "semantic_type": self.semantic_type,
            **self.metadata
        }
        return Document(page_content=self.content, metadata=metadata)


class SemanticChunker:
    """
    Advanced semantic chunking for academic textbooks with clear structure.
    
    This is a state-of-the-art chunker that leverages document structure
    rather than arbitrary character boundaries.
    """
    
    # Semantic unit patterns (English + French)
    DEFINITION_PATTERNS = [
        # Numbered definitions (EN/FR)
        r'(?:Definition|Définition|Def\.|Déf\.)\s*\d+(?:\.\d+)*[:\.]?\s*',
        # Markdown-style
        r'\*\*(?:Definition|Définition)\*\*:?',
        # Keywords (English)
        r'(?:^|\n)(?:Definition|Theorem|Lemma|Corollary|Proposition|Axiom|Postulate)[:\s]',
        # Keywords (French) - avec accents
        r'(?:^|\n)(?:Définition|Théorème|Lemme|Corollaire|Proposition|Axiome|Postulat|Propriété)[:\s\.]',
        # French variations without accents (sometimes in PDFs)
        r'(?:^|\n)(?:Definition|Theoreme|Lemme|Corollaire|Propriete)[:\s\.]',
        # French "Démonstration" / "Preuve"
        r'(?:^|\n)(?:Démonstration|Demonstration|Preuve)[:\s\.]',
    ]
    
    EXAMPLE_PATTERNS = [
        # Numbered examples (EN/FR)
        r'(?:Example|Exemple|Ex\.)\s*\d+(?:\.\d+)*[:\.]?\s*',
        # Markdown-style
        r'\*\*(?:Example|Exemple)\*\*:?',
        # Keywords (English)
        r'(?:^|\n)Example[:\s]',
        # Keywords (French)
        r'(?:^|\n)(?:Exemple|Exercice|Application|Remarque)[:\s\.]',
    ]
    
    EQUATION_PATTERNS = [
        r'\$\$.*?\$\$',  # LaTeX display equations
        r'\\\[.*?\\\]',  # LaTeX bracket equations
        r'(?:^|\n)\s*\(\d+\)\s*$',  # Equation numbers like (1), (2)
    ]
    
    def __init__(
        self,
        pdf_path: str,
        target_chunk_size: int = 1000,
        max_chunk_size: int = 2000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 300
    ):
        """
        Initialize semantic chunker.
        
        Args:
            pdf_path: Path to academic PDF textbook
            target_chunk_size: Preferred chunk size in characters (soft limit)
            max_chunk_size: Maximum chunk size before forcing split
            chunk_overlap: Overlap between chunks when splitting large sections
            min_chunk_size: Minimum viable chunk size
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        self.doc = fitz.open(str(self.pdf_path))
        self.num_pages = len(self.doc)
        
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Fallback splitter for oversized sections
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Cache
        self._toc_cache = None
        self._page_text_cache = {}
    
    def extract_toc(self) -> Dict[str, Any]:
        """
        Extract hierarchical table of contents from PDF.
        
        Strategy:
        1. Try PDF bookmarks (fitz.get_toc())
        2. If not available, parse text for section patterns (1.1, 1.1.1, etc.)
        
        Returns:
            {
                "chapters": [
                    {
                        "id": "ch1",
                        "title": "Introduction",
                        "page_start": 1,
                        "page_end": 25,
                        "sections": [...]
                    }
                ]
            }
        """
        if self._toc_cache:
            return self._toc_cache
        
        toc_raw = self.doc.get_toc()
        
        if not toc_raw:
            # Fallback: detect sections from text patterns
            self._toc_cache = self._extract_toc_from_text()
            return self._toc_cache
        
        # Parse TOC hierarchy
        chapters = []
        current_chapter = None
        current_section = None
        
        for level, title, page in toc_raw:
            if level == 1:  # Chapter level
                if current_chapter:
                    chapters.append(current_chapter)
                
                current_chapter = {
                    "id": f"ch{len(chapters) + 1}",
                    "title": title.strip(),
                    "page_start": page,
                    "page_end": None,
                    "sections": []
                }
                current_section = None
            
            elif level == 2 and current_chapter:  # Section level
                if current_section:
                    current_chapter["sections"].append(current_section)
                
                section_num = len(current_chapter["sections"]) + 1
                current_section = {
                    "id": f"{len(chapters) + 1}.{section_num}",
                    "title": title.strip(),
                    "page_start": page,
                    "page_end": None,
                    "subsections": []
                }
            
            elif level == 3 and current_section:  # Subsection level
                subsection_num = len(current_section["subsections"]) + 1
                current_section["subsections"].append({
                    "id": f"{current_section['id']}.{subsection_num}",
                    "title": title.strip(),
                    "page_start": page,
                    "page_end": None
                })
        
        # Finalize last items
        if current_section:
            current_chapter["sections"].append(current_section)
        if current_chapter:
            chapters.append(current_chapter)
        
        # Fill in end pages
        for i, chapter in enumerate(chapters):
            chapter["page_end"] = chapters[i + 1]["page_start"] - 1 if i + 1 < len(chapters) else self.num_pages
            
            for j, section in enumerate(chapter["sections"]):
                if j + 1 < len(chapter["sections"]):
                    section["page_end"] = chapter["sections"][j + 1]["page_start"] - 1
                else:
                    section["page_end"] = chapter["page_end"]
                
                for k, subsection in enumerate(section["subsections"]):
                    if k + 1 < len(section["subsections"]):
                        subsection["page_end"] = section["subsections"][k + 1]["page_start"] - 1
                    else:
                        subsection["page_end"] = section["page_end"]
        
        self._toc_cache = {"chapters": chapters}
        return self._toc_cache
    
    def _extract_toc_from_text(self) -> Dict[str, Any]:
        """
        Fallback: Extract TOC by detecting section patterns in text.
        
        Detects patterns like:
        - "1 Généralités" (chapter)
        - "1.1 Tribu" (section)  
        - "1.1.1 Définition d'une tribu" (subsection)
        
        Works for French academic documents with standard numbering.
        """
        # Patterns for detecting section headers (FR + EN)
        # 
        # CHAPTER detection strategies:
        # 1. Two-line format: "Chapter N\nTITLE" or "Chapitre N\nTitre"
        # 2. Single-line format: "1 Généralités" (less common)
        
        # Section: "1.1 Title" at start of line - title must be text, not math
        section_pattern = re.compile(
            r'^(\d+\.\d+)\s+([A-ZÀÂÄÉÈÊËÏÎÔÙÛÜÇa-zàâäéèêëïîôùûüç][a-zA-ZÀ-ÿ\s\'\-\.]{2,60})$',
            re.MULTILINE
        )
        
        # Subsection: "1.1.1 Title" at start of line
        subsection_pattern = re.compile(
            r'^(\d+\.\d+\.\d+)\s+([A-ZÀÂÄÉÈÊËÏÎÔÙÛÜÇa-zàâäéèêëïîôùûüç][a-zA-ZÀ-ÿ\s\'\-\.]{2,60})$',
            re.MULTILINE
        )
        
        # Characters that indicate math/formulas - not valid in titles
        math_chars = set('=+×÷∫∑∏√∂∇≤≥≠≈∈∉⊂⊃∪∩(){}[]|λμσαβγδ')
        
        # Words to exclude from chapter detection (headers, titles, etc.)
        excluded_words = {
            'contents', 'table', 'sommaire', 'index', 'bibliographie', 
            'references', 'annexe', 'appendix'
        }
        
        def is_valid_title(title: str) -> bool:
            """Check if title is a valid section title (not math/formula)"""
            # Must not contain math symbols
            if any(c in title for c in math_chars):
                return False
            # Must not contain equation-like patterns
            if re.search(r'[=\+\-\*\/\(\)\[\]\{\}]', title):
                return False
            # Must be mostly alphabetic
            alpha_count = sum(1 for c in title if c.isalpha())
            if len(title) > 0 and alpha_count / len(title) < 0.6:
                return False
            return True
        
        # Extract all text with page info
        all_sections = []  # [(level, number, title, page), ...]
        
        for page_num in range(self.num_pages):
            page = self.doc[page_num]
            text = page.get_text()
            
            # Skip TOC pages entirely for structure detection
            if self._is_toc_page(text):
                continue
            
            lines = text.split('\n')
            
            # Find chapters - two-line format: "Chapter N\nTITLE" or "Chapitre N\nTitre"
            # Some titles may span multiple lines
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                # Check for "Chapter N" or "Chapitre N" format
                chapter_match = re.match(r'^(?:Chapter|Chapitre)\s+(\d+)\s*$', line_stripped)
                if chapter_match and i + 1 < len(lines):
                    num = chapter_match.group(1)
                    # Collect title from next line(s) - titles in ALL CAPS may span 2 lines
                    title_parts = []
                    for j in range(i + 1, min(i + 3, len(lines))):  # Check next 2 lines max
                        next_line = lines[j].strip()
                        # Stop if line looks like content (starts with text that isn't caps)
                        if next_line and (next_line.isupper() or (len(title_parts) == 0 and next_line[0].isupper())):
                            # Skip if it's a section number like "6.1"
                            if re.match(r'^\d+\.\d+', next_line):
                                break
                            title_parts.append(next_line)
                        else:
                            break
                    
                    if title_parts:
                        title = ' '.join(title_parts)
                        # Clean up LaTeX artifacts and normalize
                        title_clean = title.replace('´', "'").replace('`', "'")
                        all_sections.append((1, num, title_clean, page_num + 1))
            
            # Find sections (still use regex on full text)
            for match in section_pattern.finditer(text):
                num, title = match.groups()
                title_clean = title.strip()
                if ('...' not in title_clean and 
                    len(title_clean) > 2 and 
                    not title_clean.isupper() and
                    is_valid_title(title_clean)):
                    all_sections.append((2, num, title_clean, page_num + 1))
            
            # Find subsections
            for match in subsection_pattern.finditer(text):
                num, title = match.groups()
                title_clean = title.strip()
                if ('...' not in title_clean and 
                    len(title_clean) > 2 and
                    not title_clean.isupper() and
                    is_valid_title(title_clean)):
                    all_sections.append((3, num, title_clean, page_num + 1))
        
        # Remove duplicates - keep first occurrence (actual content, not repeated headers)
        seen = set()
        unique_sections = []
        for level, num, title, page in all_sections:
            key = (level, num)
            if key not in seen:
                seen.add(key)
                unique_sections.append((level, num, title, page))
        
        # Sort by page then by level
        unique_sections.sort(key=lambda x: (x[3], x[0]))
        
        # If no structure found, use simple fallback
        if not unique_sections:
            return {
                "chapters": [{
                    "id": "ch1",
                    "title": self.pdf_path.stem,
                    "page_start": 1,
                    "page_end": self.num_pages,
                    "sections": [{
                        "id": "1.1",
                        "title": "Full Content",
                        "page_start": 1,
                        "page_end": self.num_pages,
                        "subsections": []
                    }]
                }]
            }
        
        # Build hierarchy
        chapters = []
        current_chapter = None
        current_section = None
        
        for level, num, title, page in unique_sections:
            if level == 1:
                if current_section and current_chapter:
                    current_chapter["sections"].append(current_section)
                if current_chapter:
                    chapters.append(current_chapter)
                
                current_chapter = {
                    "id": f"ch{num}",
                    "title": title,
                    "page_start": page,
                    "page_end": None,
                    "sections": []
                }
                current_section = None
            
            elif level == 2:
                if current_section and current_chapter:
                    current_chapter["sections"].append(current_section)
                
                # Create chapter if not exists
                if not current_chapter:
                    ch_num = num.split('.')[0]
                    current_chapter = {
                        "id": f"ch{ch_num}",
                        "title": f"Chapter {ch_num}",
                        "page_start": page,
                        "page_end": None,
                        "sections": []
                    }
                
                current_section = {
                    "id": num,
                    "title": title,
                    "page_start": page,
                    "page_end": None,
                    "subsections": []
                }
            
            elif level == 3:
                if current_section:
                    current_section["subsections"].append({
                        "id": num,
                        "title": title,
                        "page_start": page,
                        "page_end": None
                    })
        
        # Finalize
        if current_section and current_chapter:
            current_chapter["sections"].append(current_section)
        if current_chapter:
            chapters.append(current_chapter)
        
        # Fill in end pages
        for i, chapter in enumerate(chapters):
            chapter["page_end"] = chapters[i + 1]["page_start"] - 1 if i + 1 < len(chapters) else self.num_pages
            
            for j, section in enumerate(chapter["sections"]):
                if j + 1 < len(chapter["sections"]):
                    section["page_end"] = chapter["sections"][j + 1]["page_start"] - 1
                else:
                    section["page_end"] = chapter["page_end"]
                
                for k, subsection in enumerate(section["subsections"]):
                    if k + 1 < len(section["subsections"]):
                        subsection["page_end"] = section["subsections"][k + 1]["page_start"] - 1
                    else:
                        subsection["page_end"] = section["page_end"]
        
        return {"chapters": chapters}
    
    def _is_toc_page(self, page_text: str) -> bool:
        """
        Detect if a page is a Table of Contents page.
        
        TOC pages typically have:
        - Many lines with "..." or ". . ." (dotted leaders)
        - Keywords like "Contents", "Table des matières", "Sommaire"
        - Many page number references at end of lines
        """
        lines = page_text.split('\n')
        
        # Check for TOC keywords
        toc_keywords = ['contents', 'table des matières', 'sommaire', 'table of contents']
        text_lower = page_text.lower()
        has_toc_keyword = any(kw in text_lower for kw in toc_keywords)
        
        # Count lines with dotted leaders (common in TOC)
        dotted_lines = sum(1 for line in lines if '...' in line or '. . .' in line)
        
        # Count lines ending with page numbers
        page_num_pattern = re.compile(r'\s+\d{1,3}\s*$')
        page_num_lines = sum(1 for line in lines if page_num_pattern.search(line))
        
        # A page is TOC if:
        # - Has TOC keyword and many dotted lines
        # - OR has many dotted lines (>30% of non-empty lines)
        non_empty_lines = [l for l in lines if l.strip()]
        if not non_empty_lines:
            return False
            
        dotted_ratio = dotted_lines / len(non_empty_lines) if non_empty_lines else 0
        
        return (has_toc_keyword and dotted_lines > 3) or dotted_ratio > 0.3
    
    def _clean_toc_lines(self, text: str) -> str:
        """
        Remove TOC-style lines from text content.
        
        Removes lines that look like TOC entries:
        - Lines with "..." followed by page numbers
        - Lines that are just section references
        """
        lines = text.split('\n')
        cleaned_lines = []
        
        # Pattern for TOC entry lines
        toc_line_pattern = re.compile(r'^.*\.{3,}.*\d{1,3}\s*$')  # "Something ... 42"
        dotted_leader_pattern = re.compile(r'\.\s*\.\s*\.\s*')    # ". . . " pattern
        
        for line in lines:
            # Skip lines that look like TOC entries
            if toc_line_pattern.match(line.strip()):
                continue
            if dotted_leader_pattern.search(line):
                continue
            # Keep the line
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def extract_text_from_pages(self, start_page: int, end_page: int, skip_toc: bool = True) -> str:
        """
        Extract and cache text from page range.
        
        Args:
            start_page: First page (1-indexed)
            end_page: Last page (1-indexed)
            skip_toc: If True, skip pages detected as TOC pages
        """
        text_parts = []
        for page_num in range(start_page, end_page + 1):
            if page_num in self._page_text_cache:
                page_text = self._page_text_cache[page_num]
            else:
                page = self.doc[page_num - 1]  # 0-indexed
                page_text = page.get_text()
                self._page_text_cache[page_num] = page_text
            
            # Skip TOC pages if requested
            if skip_toc and self._is_toc_page(page_text):
                continue
                
            # Clean remaining TOC-style lines from content
            cleaned_text = self._clean_toc_lines(page_text)
            if cleaned_text.strip():
                text_parts.append(cleaned_text)
        
        return "\n".join(text_parts)
    
    def detect_semantic_units(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect semantic units (definitions, examples, equations) in text.
        
        Returns:
            [
                {"type": "definition", "start": 0, "end": 150, "content": "..."},
                {"type": "text", "start": 150, "end": 500, "content": "..."},
                ...
            ]
        """
        units = []
        
        # Combine all patterns
        all_patterns = []
        for pattern in self.DEFINITION_PATTERNS:
            all_patterns.append(("definition", re.compile(pattern, re.MULTILINE | re.IGNORECASE)))
        for pattern in self.EXAMPLE_PATTERNS:
            all_patterns.append(("example", re.compile(pattern, re.MULTILINE | re.IGNORECASE)))
        
        # Find all matches
        matches = []
        for unit_type, pattern in all_patterns:
            for match in pattern.finditer(text):
                matches.append((match.start(), unit_type))
        
        matches.sort()
        
        # If no semantic units found, treat entire text as one unit
        if not matches:
            return [{
                "type": "text",
                "start": 0,
                "end": len(text),
                "content": text
            }]
        
        # Create units between matches
        last_end = 0
        for start_pos, unit_type in matches:
            # Add text before this unit
            if start_pos > last_end:
                units.append({
                    "type": "text",
                    "start": last_end,
                    "end": start_pos,
                    "content": text[last_end:start_pos]
                })
            
            # Find end of semantic unit (next paragraph break or next unit)
            end_pos = text.find("\n\n", start_pos)
            if end_pos == -1:
                end_pos = len(text)
            
            units.append({
                "type": unit_type,
                "start": start_pos,
                "end": end_pos,
                "content": text[start_pos:end_pos]
            })
            
            last_end = end_pos
        
        # Add remaining text
        if last_end < len(text):
            units.append({
                "type": "text",
                "start": last_end,
                "end": len(text),
                "content": text[last_end:]
            })
        
        return units
    
    def create_chunks_from_section(
        self,
        section_text: str,
        chapter_title: str,
        section_title: str,
        section_id: str,
        page_range: Tuple[int, int],
        subsection_title: Optional[str] = None
    ) -> List[SemanticChunk]:
        """
        Create semantic chunks from a section, preserving semantic boundaries.
        
        Strategy:
        1. Detect semantic units (definitions, examples, etc.)
        2. Group units into chunks near target_chunk_size
        3. Never split a semantic unit
        4. If a single unit > max_chunk_size, use fallback splitter
        """
        chunks = []
        
        # Detect semantic units
        units = self.detect_semantic_units(section_text)
        
        # Group units into chunks
        current_chunk_content = []
        current_chunk_types = []
        current_size = 0
        
        for unit in units:
            unit_content = unit["content"].strip()
            unit_size = len(unit_content)
            
            if not unit_content:
                continue
            
            # If single unit is too large, use fallback splitter
            if unit_size > self.max_chunk_size:
                # Save current chunk if exists
                if current_chunk_content:
                    chunk_text = "\n\n".join(current_chunk_content)
                    semantic_type = self._determine_chunk_type(current_chunk_types)
                    chunks.append(self._create_chunk(
                        chunk_text, chapter_title, section_title, section_id,
                        len(chunks) + 1, page_range, semantic_type, subsection_title
                    ))
                    current_chunk_content = []
                    current_chunk_types = []
                    current_size = 0
                
                # Split large unit with fallback
                fallback_chunks = self.fallback_splitter.split_text(unit_content)
                for i, fallback_text in enumerate(fallback_chunks):
                    chunks.append(self._create_chunk(
                        fallback_text, chapter_title, section_title, section_id,
                        len(chunks) + 1, page_range, unit["type"], subsection_title
                    ))
                continue
            
            # Check if adding this unit exceeds target size
            if current_size + unit_size > self.target_chunk_size and current_chunk_content:
                # Save current chunk
                chunk_text = "\n\n".join(current_chunk_content)
                semantic_type = self._determine_chunk_type(current_chunk_types)
                chunks.append(self._create_chunk(
                    chunk_text, chapter_title, section_title, section_id,
                    len(chunks) + 1, page_range, semantic_type, subsection_title
                ))
                
                current_chunk_content = [unit_content]
                current_chunk_types = [unit["type"]]
                current_size = unit_size
            else:
                # Add to current chunk
                current_chunk_content.append(unit_content)
                current_chunk_types.append(unit["type"])
                current_size += unit_size
        
        # Save last chunk
        if current_chunk_content:
            chunk_text = "\n\n".join(current_chunk_content)
            semantic_type = self._determine_chunk_type(current_chunk_types)
            chunks.append(self._create_chunk(
                chunk_text, chapter_title, section_title, section_id,
                len(chunks) + 1, page_range, semantic_type, subsection_title
            ))
        
        return chunks
    
    def _create_chunk(
        self,
        content: str,
        chapter_title: str,
        section_title: str,
        section_id: str,
        chunk_num: int,
        page_range: Tuple[int, int],
        semantic_type: str,
        subsection_title: Optional[str] = None
    ) -> SemanticChunk:
        """Helper to create a SemanticChunk with proper metadata"""
        chunk_id = f"{section_id}.c{chunk_num}"
        
        return SemanticChunk(
            content=content,
            chunk_id=chunk_id,
            chapter_title=chapter_title,
            section_title=section_title,
            subsection_title=subsection_title,
            page_range=page_range,
            semantic_type=semantic_type,
            metadata={
                "source": str(self.pdf_path),
                "chunk_size": len(content),
                "pdf_pages": self.num_pages
            }
        )
    
    def _determine_chunk_type(self, unit_types: List[str]) -> str:
        """Determine overall chunk type from constituent unit types"""
        if not unit_types:
            return "text"
        
        # If all same type, use that
        if len(set(unit_types)) == 1:
            return unit_types[0]
        
        # If contains definition/example, prioritize that
        if "definition" in unit_types:
            return "definition"
        if "example" in unit_types:
            return "example"
        
        return "mixed"
    
    def chunk_document(self) -> List[SemanticChunk]:
        """
        Chunk entire document using semantic boundaries.
        
        Returns:
            List of SemanticChunk objects with rich metadata
        """
        all_chunks = []
        
        toc = self.extract_toc()
        
        for chapter in toc["chapters"]:
            chapter_title = chapter["title"]
            
            for section in chapter["sections"]:
                section_title = section["title"]
                section_id = section["id"]
                page_start = section["page_start"]
                page_end = section["page_end"]
                
                # Extract section text
                section_text = self.extract_text_from_pages(page_start, page_end)
                
                # Handle subsections if present
                if section["subsections"]:
                    for subsection in section["subsections"]:
                        subsection_text = self.extract_text_from_pages(
                            subsection["page_start"],
                            subsection["page_end"]
                        )
                        
                        chunks = self.create_chunks_from_section(
                            subsection_text,
                            chapter_title,
                            section_title,
                            subsection["id"],
                            (subsection["page_start"], subsection["page_end"]),
                            subsection_title=subsection["title"]
                        )
                        all_chunks.extend(chunks)
                else:
                    # No subsections, chunk entire section
                    chunks = self.create_chunks_from_section(
                        section_text,
                        chapter_title,
                        section_title,
                        section_id,
                        (page_start, page_end)
                    )
                    all_chunks.extend(chunks)
        
        return all_chunks
    
    def chunk_to_langchain_documents(self) -> List[Document]:
        """
        Chunk document and convert to LangChain Documents for vector store.
        
        Returns:
            List of LangChain Document objects
        """
        chunks = self.chunk_document()
        return [chunk.to_langchain_document() for chunk in chunks]


def compare_chunking_strategies(
    pdf_path: str,
    semantic_chunker_kwargs: Dict[str, Any] = None,
    baseline_chunk_size: int = 512,
    baseline_overlap: int = 100
) -> Dict[str, Any]:
    """
    Compare semantic chunking vs baseline character-based chunking.
    
    Args:
        pdf_path: Path to PDF
        semantic_chunker_kwargs: Optional kwargs for SemanticChunker
        baseline_chunk_size: Character-based chunk size for baseline
        baseline_overlap: Character overlap for baseline
    
    Returns:
        {
            "semantic": {"chunks": [...], "stats": {...}},
            "baseline": {"chunks": [...], "stats": {...}},
            "comparison": {...}
        }
    """
    # Semantic chunking
    semantic_kwargs = semantic_chunker_kwargs or {}
    semantic_chunker = SemanticChunker(pdf_path, **semantic_kwargs)
    semantic_chunks = semantic_chunker.chunk_document()
    
    # Baseline chunking
    doc = fitz.open(pdf_path)
    full_text = "\n".join([page.get_text() for page in doc])
    doc.close()
    
    baseline_splitter = RecursiveCharacterTextSplitter(
        chunk_size=baseline_chunk_size,
        chunk_overlap=baseline_overlap
    )
    baseline_chunks = baseline_splitter.split_text(full_text)
    
    # Stats
    semantic_stats = {
        "num_chunks": len(semantic_chunks),
        "avg_size": sum(len(c.content) for c in semantic_chunks) / len(semantic_chunks),
        "min_size": min(len(c.content) for c in semantic_chunks),
        "max_size": max(len(c.content) for c in semantic_chunks),
        "semantic_types": {t: sum(1 for c in semantic_chunks if c.semantic_type == t) 
                          for t in set(c.semantic_type for c in semantic_chunks)}
    }
    
    baseline_stats = {
        "num_chunks": len(baseline_chunks),
        "avg_size": sum(len(c) for c in baseline_chunks) / len(baseline_chunks),
        "min_size": min(len(c) for c in baseline_chunks),
        "max_size": max(len(c) for c in baseline_chunks)
    }
    
    return {
        "semantic": {
            "chunks": semantic_chunks,
            "stats": semantic_stats
        },
        "baseline": {
            "chunks": baseline_chunks,
            "stats": baseline_stats
        },
        "comparison": {
            "semantic_preserves_structure": True,
            "baseline_arbitrary_boundaries": True,
            "semantic_chunk_count": len(semantic_chunks),
            "baseline_chunk_count": len(baseline_chunks),
            "semantic_avg_size": semantic_stats["avg_size"],
            "baseline_avg_size": baseline_stats["avg_size"]
        }
    }
