"""
src/multimodal — Multimodal extension for the RAG pipeline.

Adds support for extracting figures from PDFs and generating rich
textual descriptions via a Vision-Language Model (Qwen2-VL-7B-Instruct).

Components:
    ImageExtractor  — PyMuPDF-based figure extraction + caption detection
    VLCaptioner     — Qwen2-VL-7B-Instruct wrapper for image description
"""

from .image_extractor import ImageExtractor, FigureRecord
from .vl_captioner import VLCaptioner

__all__ = ["ImageExtractor", "FigureRecord", "VLCaptioner"]
