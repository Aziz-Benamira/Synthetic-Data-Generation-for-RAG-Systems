"""
image_extractor.py — Figure extraction from academic PDFs

Extracts figures (raster images) from a PDF using PyMuPDF and attempts to
locate the corresponding caption text (e.g. "Figure 1: The attention mechanism").

Usage::

    extractor = ImageExtractor("data/pdfs/Attention_Is_All_You_Need.pdf")
    figures = extractor.extract_all()
    for fig in figures:
        print(fig.figure_id, fig.caption)
        with open(f"fig_{fig.figure_id}.png", "wb") as f:
            f.write(fig.image_bytes)
"""

from __future__ import annotations

import io
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class FigureRecord:
    """
    One figure extracted from a PDF.

    Attributes:
        figure_id   : Unique identifier, e.g. "p7_img1" (page 7, image 1)
        page        : 0-indexed page number
        image_bytes : Raw PNG bytes (converted from whatever the PDF stores)
        width       : Image width in pixels
        height      : Image height in pixels
        bbox        : Bounding box on page (x0, y0, x1, y1) in PDF points
        caption     : Caption text detected near the image (empty string if none)
        section     : Closest section/chapter heading above the image (may be empty)
        xref        : PyMuPDF internal cross-reference (for debugging)
        metadata    : Arbitrary extra info dict
    """
    figure_id: str
    page: int
    image_bytes: bytes
    width: int
    height: int
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    caption: str = ""
    section: str = ""
    surrounding_text: str = ""   # text paragraphs above + below the image on the same page
    xref: int = -1
    metadata: dict = field(default_factory=dict)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def to_chunk_dict(self) -> dict:
        """
        Convert to a dict compatible with the SemanticChunk / Gold Dataset schema.

        The 'content' field contains the AI-generated description (filled by
        VLCaptioner) or falls back to the detected caption.  Callers should
        call VLCaptioner.describe(figure) and set figure.metadata['vl_description']
        before calling this method.
        """
        vl_desc = self.metadata.get("vl_description", "")
        content_parts = []
        if vl_desc:
            content_parts.append(f"[Vision-Language Description]\n{vl_desc}")
        if self.caption:
            content_parts.append(f"[Figure Caption]\n{self.caption}")
        if self.surrounding_text:
            content_parts.append(f"[Surrounding Context]\n{self.surrounding_text}")
        if not content_parts:
            content_parts.append(f"[Figure {self.figure_id} — no description available]")

        return {
            "chunk_id": self.figure_id,
            "content": "\n\n".join(content_parts),
            "semantic_type": "figure",
            "page_range": [self.page, self.page],
            # Keys expected by PipelineV4 (_process_chunk uses chunk.get("chapter") etc.)
            "chapter": self.section,
            "section": self.caption or self.figure_id,
            "metadata": {
                "source_figure_id": self.figure_id,
                "caption": self.caption,
                "surrounding_text": self.surrounding_text,
                "page": self.page,
                "width": self.width,
                "height": self.height,
                "vl_description": vl_desc,
            },
        }


# ── Caption patterns (multilingual) ──────────────────────────────────────────

_CAPTION_RE = re.compile(
    r"""
    (?:                         # match at start of a text span
        Figure\s*\d+            # Figure 1
      | Fig\.\s*\d+             # Fig. 1
      | Figure\s*:              # Figure :  (French sans numéro)
      | Figure\s+[A-Z]\d*       # Figure A1
      | FIGURE\s*\d+            # FIGURE 1 (caps)
      | Tableau\s*\d+           # Tableau 1  (French table)
      | Table\s*\d+             # Table 1
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Heading patterns to detect section titles above images
_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\s+)?[A-ZÀÂÉÈÊËÏÎÔÙŒÆ][^\n]{3,60}$",
    re.MULTILINE,
)


# ── Main class ────────────────────────────────────────────────────────────────

class ImageExtractor:
    """
    Extract figures from a PDF document with caption & section detection.

    Args:
        pdf_path          : Path to the PDF file.
        min_width         : Ignore images narrower than this (pixels).
        min_height        : Ignore images shorter than this (pixels).
        caption_search_px : How far below / above the image bbox (in PDF points)
                            to search for a caption line.
    """

    def __init__(
        self,
        pdf_path: str,
        min_width: int = 100,
        min_height: int = 100,
        caption_search_px: float = 80.0,
        context_chars: int = 600,
    ):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        self.doc = fitz.open(str(self.pdf_path))
        self.min_width = min_width
        self.min_height = min_height
        self.caption_search_px = caption_search_px
        self.context_chars = context_chars   # max chars to take from above/below image

        logger.info(
            "ImageExtractor ready: %s (%d pages)", self.pdf_path.name, len(self.doc)
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_all(self) -> List[FigureRecord]:
        """Extract all qualifying figures from every page."""
        all_figures: List[FigureRecord] = []
        for page_idx in range(len(self.doc)):
            page_figures = self._extract_page(page_idx)
            all_figures.extend(page_figures)

        logger.info("Total figures extracted: %d", len(all_figures))
        return all_figures

    def extract_pages(self, pages: List[int]) -> List[FigureRecord]:
        """Extract figures from a specific list of page indices (0-indexed)."""
        figures: List[FigureRecord] = []
        for page_idx in pages:
            figures.extend(self._extract_page(page_idx))
        return figures

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _extract_page(self, page_idx: int) -> List[FigureRecord]:
        """Extract images from a single page, with caption & section lookup."""
        page = self.doc[page_idx]
        image_list = page.get_images(full=True)  # list of (xref, smask, w, h, ...)

        figures: List[FigureRecord] = []
        page_text_blocks = self._get_text_blocks(page)

        for img_order, img_info in enumerate(image_list):
            xref = img_info[0]
            width = img_info[2]
            height = img_info[3]

            if width < self.min_width or height < self.min_height:
                logger.debug(
                    "Skipping small image xref=%d (%dx%d) on page %d",
                    xref, width, height, page_idx,
                )
                continue

            # Convert to PNG bytes
            try:
                image_bytes = self._to_png(xref)
            except Exception as exc:
                logger.warning("Could not extract image xref=%d: %s", xref, exc)
                continue

            # Find the image bbox on the page
            bbox = self._find_image_bbox(page, xref)

            figure_id = f"p{page_idx + 1}_img{img_order + 1}"

            caption = self._find_caption(bbox, page_text_blocks, page_idx)
            section = self._find_section(bbox, page_text_blocks)
            surrounding = self._find_surrounding_text(bbox, page_text_blocks, caption)

            rec = FigureRecord(
                figure_id=figure_id,
                page=page_idx,
                image_bytes=image_bytes,
                width=width,
                height=height,
                bbox=bbox,
                caption=caption,
                section=section,
                surrounding_text=surrounding,
                xref=xref,
            )
            figures.append(rec)
            logger.info(
                "  [%s] page=%d  %dx%d  caption=%r",
                figure_id, page_idx + 1, width, height, caption[:60] if caption else "",
            )

        return figures

    def _to_png(self, xref: int) -> bytes:
        """Extract raw image and convert to PNG bytes."""
        base_image = self.doc.extract_image(xref)
        img_bytes = base_image["image"]
        ext = base_image.get("ext", "png").lower()

        if ext == "png":
            return img_bytes

        # For JPEG / JBIG2 / etc. convert via fitz Pixmap
        pix = fitz.Pixmap(self.doc, xref)
        if pix.n > 4:  # CMYK → RGB
            pix = fitz.Pixmap(fitz.csRGB, pix)
        png_bytes = pix.tobytes("png")
        pix = None  # release
        return png_bytes

    def _find_image_bbox(
        self, page: fitz.Page, xref: int
    ) -> Tuple[float, float, float, float]:
        """Return the bounding box of an embedded image on a page."""
        # fitz can give us image placements
        for item in page.get_image_rects(xref):
            # item is a Rect
            return (item.x0, item.y0, item.x1, item.y1)
        # Fallback: full page rect
        r = page.rect
        return (r.x0, r.y0, r.x1, r.y1)

    def _get_text_blocks(self, page: fitz.Page) -> List[dict]:
        """
        Return a list of text blocks with bbox and text content.
        Each block: {'bbox': (x0,y0,x1,y1), 'text': str}
        """
        blocks = []
        raw = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
        for b in raw:
            if b[6] == 0:  # text block (not image)
                blocks.append({"bbox": (b[0], b[1], b[2], b[3]), "text": b[4].strip()})
        return blocks

    def _find_caption(
        self,
        img_bbox: Tuple[float, float, float, float],
        text_blocks: List[dict],
        page_idx: int,
    ) -> str:
        """
        Search for a caption text block near the image.

        Strategy:
        1. Look for blocks whose top edge is within caption_search_px BELOW img_bbox.y1
        2. Also look ABOVE img_bbox.y0 (some figures have captions above)
        3. Among candidates, prefer blocks matching _CAPTION_RE.
        """
        x0, y0, x1, y1 = img_bbox
        candidates = []

        for block in text_blocks:
            bx0, by0, bx1, by1 = block["bbox"]
            text = block["text"]
            if not text:
                continue

            # Horizontal overlap check — the caption should be in a similar x range
            h_overlap = min(bx1, x1) - max(bx0, x0)
            if h_overlap < (x1 - x0) * 0.3:
                continue

            # Vertical proximity: below image
            if 0 <= by0 - y1 <= self.caption_search_px:
                score = 10 if _CAPTION_RE.search(text) else 1
                candidates.append((score, by0, text))
            # Vertical proximity: above image
            elif 0 <= y0 - by1 <= self.caption_search_px:
                score = 10 if _CAPTION_RE.search(text) else 1
                candidates.append((score, by0, text))

        if not candidates:
            return ""

        # Sort by score desc, then by vertical position asc
        candidates.sort(key=lambda c: (-c[0], c[1]))
        return candidates[0][2].replace("\n", " ").strip()

    def _find_section(
        self,
        img_bbox: Tuple[float, float, float, float],
        text_blocks: List[dict],
    ) -> str:
        """
        Return the closest heading/section title that appears ABOVE the image.
        """
        _x0, y0, _x1, _y1 = img_bbox
        best_text = ""
        best_y = -1.0

        for block in text_blocks:
            _bx0, by0, _bx1, by1 = block["bbox"]
            text = block["text"].replace("\n", " ").strip()
            if not text:
                continue
            # Must be above the image
            if by1 > y0:
                continue
            # Heuristic: heading-like block (short, title-case or numbered)
            if _HEADING_RE.search(text) and by0 > best_y:
                best_y = by0
                best_text = text

        return best_text

    def _find_surrounding_text(
        self,
        img_bbox: Tuple[float, float, float, float],
        text_blocks: List[dict],
        caption: str,
    ) -> str:
        """
        Collect the text paragraphs that immediately surround the image on the page.

        Strategy:
        - ABOVE the image: take text blocks whose bottom edge is above the image top,
          closest first, up to context_chars characters total.
        - BELOW the image: take text blocks whose top edge is below the image bottom,
          closest first, up to context_chars characters total.
        - Skip: the caption (already captured), very short blocks (page numbers /
          running headers < 20 chars), and the section heading (already in `section`).

        Returns a single string with an "--- above ---" / "--- below ---" separator.
        """
        _x0, y0, _x1, y1 = img_bbox
        caption_norm = caption.strip().lower()

        above_blocks: List[Tuple[float, str]] = []   # (distance_to_image, text)
        below_blocks: List[Tuple[float, str]] = []

        for block in text_blocks:
            _bx0, by0, _bx1, by1 = block["bbox"]
            text = block["text"].replace("\n", " ").strip()

            # Skip empty, very short (headers/page numbers), or the caption itself
            if len(text) < 20:
                continue
            if text.lower() == caption_norm:
                continue

            if by1 <= y0:                          # block is entirely ABOVE image
                above_blocks.append((y0 - by1, text))
            elif by0 >= y1:                        # block is entirely BELOW image
                below_blocks.append((by0 - y1, text))

        # Sort by proximity (closest first)
        above_blocks.sort(key=lambda x: x[0])
        below_blocks.sort(key=lambda x: x[0])

        # Accumulate up to context_chars chars for each direction
        def _collect(blocks: List[Tuple[float, str]], max_chars: int) -> str:
            parts, total = [], 0
            for _, text in blocks:
                if total >= max_chars:
                    break
                parts.append(text)
                total += len(text)
            return " ".join(parts)[:max_chars]

        above_text = _collect(above_blocks, self.context_chars)
        below_text = _collect(below_blocks, self.context_chars)

        parts = []
        if above_text:
            parts.append(f"[Before figure]\n{above_text}")
        if below_text:
            parts.append(f"[After figure]\n{below_text}")

        return "\n\n".join(parts)
