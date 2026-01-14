"""
Text Processing Utilities for SoG Pipeline

Utilities for document preprocessing, paragraph splitting, and text cleaning.
"""

from typing import List, Tuple
import re
from pathlib import Path


def load_text_file(file_path: str, encoding: str = 'utf-8') -> str:
    """
    Load text from a file.
    
    Args:
        file_path: Path to text file
        encoding: File encoding
        
    Returns:
        File contents as string
    """
    with open(file_path, 'r', encoding=encoding) as f:
        return f.read()


def split_into_paragraphs(text: str, min_length: int = 50) -> List[str]:
    """
    Split text into paragraphs.
    
    Args:
        text: Input text
        min_length: Minimum paragraph length in characters
        
    Returns:
        List of paragraph texts
    """
    # Split by double newlines or single newlines followed by indentation
    paragraphs = re.split(r'\n\s*\n+|\n(?=\s{4,})', text)
    
    # Clean and filter paragraphs
    cleaned = []
    for para in paragraphs:
        para = para.strip()
        if len(para) >= min_length:
            cleaned.append(para)
    
    return cleaned


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    # Simple sentence splitting (can be improved with NLTK or spaCy)
    sentences = re.split(r'[.!?]+\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def clean_text(text: str, remove_special_chars: bool = False) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text
        remove_special_chars: Whether to remove special characters
        
    Returns:
        Cleaned text
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters if requested
    if remove_special_chars:
        text = re.sub(r'[^\w\s.,!?;:()\[\]{}\'""-]', '', text)
    
    return text.strip()


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to end at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            boundary = max(last_period, last_newline)
            
            if boundary > chunk_size // 2:  # Only if boundary is in latter half
                chunk = chunk[:boundary + 1]
                end = start + boundary + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks


def extract_title_and_content(text: str) -> Tuple[str, str]:
    """
    Extract title and content from text.
    
    Assumes first line or first paragraph is the title.
    
    Args:
        text: Input text
        
    Returns:
        Tuple of (title, content)
    """
    lines = text.split('\n', 1)
    
    if len(lines) == 1:
        return "", text
    
    title = lines[0].strip()
    content = lines[1].strip()
    
    # If title is too long, treat as regular content
    if len(title) > 200:
        return "", text
    
    return title, content


def remove_citations(text: str) -> str:
    """
    Remove citation markers from text.
    
    Args:
        text: Input text
        
    Returns:
        Text without citations
    """
    # Remove [1], [2], etc.
    text = re.sub(r'\[\d+\]', '', text)
    
    # Remove (Author, Year) style citations
    text = re.sub(r'\([A-Z][a-z]+(?:\s+et\s+al\.)?,\s*\d{4}\)', '', text)
    
    return text


def merge_short_paragraphs(paragraphs: List[str], 
                          min_length: int = 100,
                          max_length: int = 1000) -> List[str]:
    """
    Merge consecutive short paragraphs.
    
    Args:
        paragraphs: List of paragraphs
        min_length: Minimum paragraph length
        max_length: Maximum merged paragraph length
        
    Returns:
        List of merged paragraphs
    """
    merged = []
    current = ""
    
    for para in paragraphs:
        if len(current) + len(para) <= max_length:
            current = current + " " + para if current else para
        else:
            if current:
                merged.append(current.strip())
            current = para
    
    if current:
        merged.append(current.strip())
    
    return merged


def count_tokens_approximate(text: str) -> int:
    """
    Approximate token count (words * 1.3 for English text).
    
    Args:
        text: Input text
        
    Returns:
        Approximate token count
    """
    words = len(text.split())
    return int(words * 1.3)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate text to approximate token limit.
    
    Args:
        text: Input text
        max_tokens: Maximum number of tokens
        
    Returns:
        Truncated text
    """
    words = text.split()
    max_words = int(max_tokens / 1.3)
    
    if len(words) <= max_words:
        return text
    
    return ' '.join(words[:max_words])
