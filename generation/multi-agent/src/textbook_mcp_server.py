"""
Textbook MCP Server
===================
Exposes PDF textbook data through Model Context Protocol tools.

Tools:
1. read_curriculum_structure() - Get table of contents
2. read_section_content(section_id) - Get full text of a section
3. vector_search(query, k, filters) - Semantic search
4. keyword_search(query) - Exact match search
5. verify_citation(page_number, quote) - Validate citations
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from dataclasses import dataclass, asdict

from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

from vector_store import VectorStore
from pdf_processor import PDFProcessor


@dataclass
class Citation:
    """Citation verification result"""
    exists: bool
    similarity_score: float
    actual_text: str
    verdict: str  # ACCURATE, INACCURATE, NOT_FOUND


class TextbookMCPServer:
    """MCP Server for University Textbook Access"""
    
    def __init__(self, pdf_path: str, vector_store: VectorStore):
        """
        Initialize the textbook server.
        
        Args:
            pdf_path: Path to the PDF textbook
            vector_store: Vector store instance with indexed content
        """
        self.server = Server("textbook-server")
        self.pdf_processor = PDFProcessor(pdf_path)
        self.vector_store = vector_store
        
        # Cache curriculum structure
        self.curriculum = None
        
        # Register all tools
        self._register_tools()
    
    def _register_tools(self):
        """Register all MCP tools"""
        
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="read_curriculum_structure",
                    description="Get the complete table of contents with chapters, sections, and page ranges. "
                                "Use this to understand the textbook structure and select sections for question generation.",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="read_section_content",
                    description="Read the full text content of a specific section. "
                                "Returns raw text, key definitions, and equations. "
                                "Used by Question Generator to understand material before creating questions.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "section_id": {
                                "type": "string",
                                "description": "Section identifier (e.g., '1.2', '3.4')"
                            }
                        },
                        "required": ["section_id"]
                    }
                ),
                Tool(
                    name="vector_search",
                    description="Perform semantic search over textbook chunks. "
                                "Returns top-K most relevant chunks with metadata (page, section, score). "
                                "Used by Answer Generator to retrieve context for answering questions.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (question or keywords)"
                            },
                            "k": {
                                "type": "integer",
                                "description": "Number of results to return",
                                "default": 5
                            },
                            "filters": {
                                "type": "object",
                                "description": "Metadata filters (e.g., {'chapter': 'ch1'})",
                                "properties": {
                                    "chapter": {"type": "string"},
                                    "section": {"type": "string"},
                                    "page_min": {"type": "integer"},
                                    "page_max": {"type": "integer"}
                                }
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="keyword_search",
                    description="Find exact keyword matches in the textbook. "
                                "Critical for finding specific terminology, formulas, or definitions. "
                                "Returns chunks containing exact matches with context window.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Exact keyword or phrase to search for"
                            },
                            "context_window": {
                                "type": "integer",
                                "description": "Characters to include before/after match",
                                "default": 200
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="verify_citation",
                    description="Verify if a cited quote exists on the specified page. "
                                "Detects hallucinations by checking if the Answer Agent's citations are accurate. "
                                "Returns similarity score and actual text from the page.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "page_number": {
                                "type": "integer",
                                "description": "Page number being cited"
                            },
                            "quote_snippet": {
                                "type": "string",
                                "description": "The quoted text to verify"
                            },
                            "threshold": {
                                "type": "number",
                                "description": "Similarity threshold for accuracy (0-1)",
                                "default": 0.85
                            }
                        },
                        "required": ["page_number", "quote_snippet"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            """Handle tool calls"""
            
            if name == "read_curriculum_structure":
                result = self._read_curriculum_structure()
                
            elif name == "read_section_content":
                section_id = arguments["section_id"]
                result = self._read_section_content(section_id)
                
            elif name == "vector_search":
                query = arguments["query"]
                k = arguments.get("k", 5)
                filters = arguments.get("filters", {})
                result = self._vector_search(query, k, filters)
                
            elif name == "keyword_search":
                query = arguments["query"]
                context_window = arguments.get("context_window", 200)
                result = self._keyword_search(query, context_window)
                
            elif name == "verify_citation":
                page_number = arguments["page_number"]
                quote_snippet = arguments["quote_snippet"]
                threshold = arguments.get("threshold", 0.85)
                result = self._verify_citation(page_number, quote_snippet, threshold)
            
            else:
                raise ValueError(f"Unknown tool: {name}")
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
    
    def _read_curriculum_structure(self) -> Dict[str, Any]:
        """
        Extract table of contents from PDF.
        
        Returns:
            {
                "chapters": [
                    {
                        "id": "ch1",
                        "title": "Classical Mechanics",
                        "pages": [1, 45],
                        "sections": [
                            {"id": "1.1", "title": "Newton's Laws", "pages": [1, 12]},
                            ...
                        ]
                    }
                ]
            }
        """
        if self.curriculum is None:
            self.curriculum = self.pdf_processor.extract_curriculum_structure()
        
        return self.curriculum
    
    def _read_section_content(self, section_id: str) -> Dict[str, Any]:
        """
        Get full text of a section.
        
        Args:
            section_id: Section identifier (e.g., "1.2")
        
        Returns:
            {
                "section_id": "1.2",
                "title": "Energy Conservation",
                "pages": [13, 24],
                "full_text": "...",
                "key_definitions": [...],
                "equations": [...]
            }
        """
        section_info = self._find_section_in_curriculum(section_id)
        
        if not section_info:
            return {"error": f"Section {section_id} not found"}
        
        # Extract text from page range
        start_page, end_page = section_info["pages"]
        full_text = self.pdf_processor.extract_text_range(start_page, end_page)
        
        # Extract key information
        definitions = self.pdf_processor.extract_definitions(full_text)
        equations = self.pdf_processor.extract_equations(full_text)
        
        return {
            "section_id": section_id,
            "title": section_info["title"],
            "pages": section_info["pages"],
            "full_text": full_text,
            "key_definitions": definitions,
            "equations": equations,
            "word_count": len(full_text.split())
        }
    
    def _vector_search(
        self, 
        query: str, 
        k: int = 5, 
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Semantic search using vector embeddings.
        
        Args:
            query: Search query
            k: Number of results
            filters: Metadata filters
        
        Returns:
            {
                "chunks": [
                    {
                        "text": "...",
                        "page_number": 14,
                        "section_id": "1.2",
                        "chapter": "ch1",
                        "relevance_score": 0.94
                    }
                ]
            }
        """
        results = self.vector_store.search(
            query=query,
            k=k,
            where=filters
        )
        
        return {
            "query": query,
            "num_results": len(results),
            "chunks": [
                {
                    "text": r["text"],
                    "page_number": r["metadata"]["page_number"],
                    "section_id": r["metadata"]["section_id"],
                    "chapter": r["metadata"]["chapter"],
                    "relevance_score": float(r["distance"])
                }
                for r in results
            ]
        }
    
    def _keyword_search(
        self, 
        query: str, 
        context_window: int = 200
    ) -> Dict[str, Any]:
        """
        Exact keyword matching.
        
        Args:
            query: Keyword or phrase
            context_window: Characters before/after match
        
        Returns:
            {
                "exact_matches": [
                    {
                        "text": "...",
                        "page_number": 78,
                        "section_id": "2.3",
                        "context_before": "...",
                        "context_after": "..."
                    }
                ]
            }
        """
        matches = self.pdf_processor.find_exact_matches(query, context_window)
        
        return {
            "query": query,
            "num_matches": len(matches),
            "exact_matches": matches
        }
    
    def _verify_citation(
        self, 
        page_number: int, 
        quote_snippet: str,
        threshold: float = 0.85
    ) -> Dict[str, Any]:
        """
        Verify if a quote exists on a page.
        
        Args:
            page_number: Page to check
            quote_snippet: Text to verify
            threshold: Similarity threshold
        
        Returns:
            {
                "exists": true/false,
                "similarity_score": 0.96,
                "actual_text": "...",
                "verdict": "ACCURATE" / "INACCURATE" / "NOT_FOUND"
            }
        """
        # Get text from the page
        page_text = self.pdf_processor.extract_text_from_page(page_number)
        
        if not page_text:
            return {
                "exists": False,
                "similarity_score": 0.0,
                "actual_text": "",
                "verdict": "NOT_FOUND",
                "error": f"Page {page_number} not found or empty"
            }
        
        # Find the most similar substring
        similarity, best_match = self._find_most_similar_substring(
            quote_snippet, 
            page_text
        )
        
        # Determine verdict
        if similarity >= threshold:
            verdict = "ACCURATE"
            exists = True
        elif similarity >= 0.70:
            verdict = "INACCURATE"
            exists = True
        else:
            verdict = "NOT_FOUND"
            exists = False
        
        return {
            "exists": exists,
            "similarity_score": round(similarity, 3),
            "actual_text": best_match,
            "verdict": verdict,
            "page_number": page_number,
            "threshold_used": threshold
        }
    
    def _find_most_similar_substring(
        self, 
        query: str, 
        text: str
    ) -> tuple[float, str]:
        """
        Find the most similar substring in text using sliding window.
        
        Args:
            query: Text to find
            text: Text to search in
        
        Returns:
            (similarity_score, best_matching_substring)
        """
        from difflib import SequenceMatcher
        
        query_len = len(query)
        best_score = 0.0
        best_match = ""
        
        # Sliding window approach
        for i in range(len(text) - query_len + 1):
            window = text[i:i + query_len]
            score = SequenceMatcher(None, query.lower(), window.lower()).ratio()
            
            if score > best_score:
                best_score = score
                best_match = window
        
        return best_score, best_match
    
    def _find_section_in_curriculum(self, section_id: str) -> Optional[Dict[str, Any]]:
        """Find section info in curriculum structure"""
        curriculum = self._read_curriculum_structure()
        
        for chapter in curriculum["chapters"]:
            for section in chapter["sections"]:
                if section["id"] == section_id:
                    return section
        
        return None
    
    async def run(self):
        """Run the MCP server"""
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


# Example usage
if __name__ == "__main__":
    import asyncio
    from vector_store import VectorStore
    
    # Initialize components
    vector_store = VectorStore(collection_name="textbook_chunks")
    server = TextbookMCPServer(
        pdf_path="data/textbook.pdf",
        vector_store=vector_store
    )
    
    # Run server
    asyncio.run(server.run())
