"""
ENSTA RAG Parser - Adapted for Multi-Agent Synthetic Data Generation

This parser is taken from the ENSTA RAG project (2025 team) with minimal modifications.
It provides robust PDF text extraction with multiple backends and clean text processing.

Original Source: https://github.com/maxenceleguery/enstrag
Team: Maxence Leguéry, Antoine Domingues, Albin Joyeux, Mattéo Denis, Simon Zarka
License: CC BY-NC-SA 4.0

Modifications:
- Removed unused imports (unstructured library)
- Added type hints for clarity
- Simplified for our use case
"""

from PyPDF2 import PdfReader
import pymupdf
import requests
import os
import re
import json
from langchain_core.documents import Document
from typing import List, Literal
from hashlib import sha256
from dataclasses import dataclass


@dataclass
class FileDocument:
    """
    Represents a document file with metadata.
    
    Attributes:
        url: Optional URL where the PDF is hosted
        local_path: Optional local file system path
        name: Human-readable name for the document
        label: Category/domain label (e.g., "Machine Learning", "Physics")
    """
    url: str | None
    local_path: str | None
    name: str
    label: str


def store_filedoc(filedoc: FileDocument) -> None:
    """
    Store document metadata to a JSON database for tracking.
    
    Creates/updates a filedocs.json file in PERSIST_PATH to track all processed documents.
    Prevents duplicate entries based on URL, path, or name.
    
    Args:
        filedoc: FileDocument to store
    """
    if (folder := os.environ.get("PERSIST_PATH")) is None:
        return
    
    json_database = os.path.join(folder, "filedocs.json")
    if os.path.exists(json_database):
        with open(json_database, "r") as f:
            filedocs = json.load(f)
    else:
        filedocs = []
    
    # Check for duplicates
    for doc in filedocs:
        if doc["url"] == filedoc.url or doc["local_path"] == filedoc.local_path or doc["name"] == filedoc.name:
            return
        
    filedocs.append(
        {
            "url": filedoc.url,
            "local_path": filedoc.local_path,
            "name": filedoc.name,
            "label": filedoc.label
        }
    )
    
    with open(json_database, "w") as f:
        json.dump(filedocs, f, indent=2)


def load_filedocs() -> List[FileDocument]:
    """
    Load previously processed documents from JSON database.
    
    Returns:
        List of FileDocument objects, or empty list if no database exists
    """
    if (folder := os.environ.get("PERSIST_PATH")) is None:
        return []
    
    json_database = os.path.join(folder, "filedocs.json")
    if os.path.exists(json_database):
        with open(json_database, "r") as f:
            return [
                FileDocument(file["url"], file["local_path"], file["name"], file["label"]) 
                for file in json.load(f)
            ]
    return []


class Parser:
    """
    Robust PDF text extraction with multiple backends and text cleaning.
    
    Supports two extraction backends:
    - PyPDF2: Faster, works well for most PDFs
    - pymupdf: More accurate for complex layouts
    """
    
    def __init__(self):
        pass
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean extracted text by removing non-printable characters and normalizing whitespace.
        
        Preserves:
        - Standard ASCII printable characters (0x20-0x7E)
        - Accented characters (À-ÿ)
        - Superscripts (², ³, ¹)
        - Newlines
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text with normalized whitespace
        """
        # Remove non-printable characters except newlines
        text = re.sub(r'[^\x20-\x7E\u00C0-\u00FF\u00B2\u00B3\u00B9\n]', '', text)
        # Replace multiple spaces with single space, preserve newlines
        text = re.sub(r'[^\S\n]+', ' ', text).strip()
        return text

    @staticmethod
    def get_text_from_pdf(path_to_pdf: str, backend: Literal["PyPDF2", "pymupdf"] = "PyPDF2") -> str:
        """
        Extract text from a PDF file using specified backend.
        
        Args:
            path_to_pdf: Absolute path to PDF file
            backend: Extraction library to use ("PyPDF2" or "pymupdf")
            
        Returns:
            Extracted and cleaned text from all pages
            
        Raises:
            FileNotFoundError: If PDF doesn't exist
            ValueError: If file is not a PDF or backend is invalid
            
        Example:
            >>> text = Parser.get_text_from_pdf("textbook.pdf", backend="pymupdf")
        """
        if not os.path.exists(path_to_pdf):
            raise FileNotFoundError(f"File {path_to_pdf} does not exist")
        if not path_to_pdf.endswith(".pdf"):
            raise ValueError(f"PDF file is expected. Got {path_to_pdf}")
        
        texts = []
        
        if backend == "PyPDF2":
            reader = PdfReader(path_to_pdf)
            for _, page in enumerate(reader.pages):
                raw_text = page.extract_text()
                if raw_text:
                    cleaned_text = Parser.clean_text(raw_text)
                    texts.append(cleaned_text + "\n\n")

        elif backend == "pymupdf":
            doc = pymupdf.open(path_to_pdf)
            for page in doc:
                cleaned_text = Parser.clean_text(page.get_textpage().extractText())
                texts.append(cleaned_text + "\n\n")

        else:
            raise ValueError(f"Wrong pdf extraction backend. Got {backend} instead of 'PyPDF2' or 'pymupdf'")

        return " ".join(texts)

    @staticmethod
    def download_pdf(url: str, name: str = None) -> str:
        """
        Download a PDF from URL and cache it locally.
        
        Downloads to PERSIST_PATH/pdfs/ directory. Skips download if file already exists.
        
        Args:
            url: URL of the PDF to download
            name: Optional custom name for the file (defaults to sanitized URL)
            
        Returns:
            Local path to downloaded PDF, or empty string on failure
            
        Example:
            >>> path = Parser.download_pdf(
            ...     "http://example.com/book.pdf", 
            ...     name="ML_textbook"
            ... )
        """
        if os.environ.get("PERSIST_PATH") is None:
            return ""
        
        TMP_FOLDER = os.path.join(os.environ.get("PERSIST_PATH"), "pdfs") 
        os.makedirs(TMP_FOLDER, exist_ok=True)

        if name is None:
            name = url.replace("/", "_")
        name = name.replace(" ", "_")

        pdf_path = os.path.join(TMP_FOLDER, f'{name}.pdf')
        
        # Skip if already downloaded
        if not os.path.exists(pdf_path):
            with open(pdf_path, 'wb') as f:
                try:
                    response = requests.get(url)
                    f.write(response.content)
                except Exception as e:
                    print(f"Failed to download {url}. Error: {e}")
                    return ""

        return pdf_path

    @staticmethod
    def get_text_from_pdf_url(url: str, name: str = None) -> tuple[str, str]:
        """
        Download PDF from URL and extract text in one step.
        
        Args:
            url: URL of the PDF
            name: Optional custom name for caching
            
        Returns:
            Tuple of (extracted_text, local_path)
        """
        pdf_path = Parser.download_pdf(url, name)
        text = Parser.get_text_from_pdf(pdf_path)
        return text, pdf_path

    @staticmethod
    def get_document_from_filedoc(filedoc: FileDocument, get_pages_num: bool = True) -> Document:
        """
        Convert a FileDocument to a LangChain Document with metadata.
        
        Downloads PDF if not available locally, extracts text, and creates a Document
        with rich metadata including hash for deduplication.
        
        Args:
            filedoc: FileDocument specification
            get_pages_num: Unused parameter (kept for compatibility)
            
        Returns:
            LangChain Document with page_content and metadata
            
        Example:
            >>> filedoc = FileDocument(
            ...     url="http://example.com/ml.pdf",
            ...     local_path=None,
            ...     name="Bishop ML",
            ...     label="Machine Learning"
            ... )
            >>> doc = Parser.get_document_from_filedoc(filedoc)
            >>> print(doc.metadata["hash"])
        """
        # Download if needed
        if filedoc.local_path is None or not os.path.exists(filedoc.local_path):
            filedoc.local_path = Parser.download_pdf(filedoc.url, filedoc.name)
        
        # Extract text
        text = Parser.get_text_from_pdf(filedoc.local_path)
        
        # Store metadata
        store_filedoc(filedoc)    

        return Document(
            page_content=text,
            metadata={
                "hash": sha256(text.encode('utf-8')).hexdigest(),
                "name": str(filedoc.name),
                "label": str(filedoc.label),
                "url": str(filedoc.url),
                "path": str(filedoc.local_path),
            }
        )

    @staticmethod
    def get_documents_from_filedocs(filedocs: List[FileDocument], get_pages_num: bool = True) -> List[Document]:
        """
        Batch process multiple FileDocuments into LangChain Documents.
        
        Args:
            filedocs: List of FileDocument specifications
            get_pages_num: Unused parameter (kept for compatibility)
            
        Returns:
            List of LangChain Documents
            
        Example:
            >>> filedocs = [
            ...     FileDocument("http://example.com/ml.pdf", None, "ML Book", "ML"),
            ...     FileDocument("http://example.com/physics.pdf", None, "Physics", "Physics")
            ... ]
            >>> docs = Parser.get_documents_from_filedocs(filedocs)
            >>> print(len(docs))  # 2
        """
        docs = []
        for filedoc in filedocs:
            documents = Parser.get_document_from_filedoc(filedoc, get_pages_num)
            if isinstance(documents, Document):
                docs.append(documents)
            else:
                docs.extend(documents)
        return docs
