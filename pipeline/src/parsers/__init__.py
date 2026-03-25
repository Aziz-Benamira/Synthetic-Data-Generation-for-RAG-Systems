"""
PDF parsing utilities for extracting text from academic documents.

This module provides parsers adapted from the ENSTA RAG project with
enhancements for multi-agent synthetic data generation.
"""

from .ensta_parser import Parser, FileDocument, load_filedocs, store_filedoc

__all__ = ["Parser", "FileDocument", "load_filedocs", "store_filedoc"]
