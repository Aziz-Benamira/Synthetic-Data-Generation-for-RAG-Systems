"""
vl_captioner.py — Vision-Language description generator for PDF figures

Uses Qwen2-VL-7B-Instruct (local) to generate rich textual descriptions of
figures extracted from academic PDFs.  The description is stored in
FigureRecord.metadata['vl_description'] and included in the chunk content.

Usage::

    captioner = VLCaptioner("/home/ensta/data/Qwen2-VL-7B-Instruct")
    captioner.load()          # loads model onto GPU

    for figure in figures:
        captioner.describe(figure)   # fills figure.metadata['vl_description']
        print(figure.metadata['vl_description'])

    captioner.unload()        # frees VRAM when done
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)


# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are an expert in scientific figure analysis for academic documents. "
    "Your task is to describe figures from research papers and textbooks in a "
    "way that is informative, precise, and suitable for question-answering systems. "
    "Describe what is shown, its purpose, key elements, and any labels or values "
    "visible in the figure."
)

_USER_PROMPT_TEMPLATE = """\
Please provide a detailed description of this figure from an academic document.

{context_block}

Describe:
1. What type of figure this is (diagram, graph, architecture, table, equation, etc.)
2. The key visual elements and their relationships
3. Any labels, annotations, axes, or numerical values present
4. The apparent purpose or main message of the figure in its academic context

Be specific and factual. Do not speculate beyond what is visible."""


def _build_user_prompt(caption: str = "", section: str = "") -> str:
    context_parts = []
    if section:
        context_parts.append(f"Section context: {section}")
    if caption:
        context_parts.append(f"Figure caption from PDF: {caption}")
    context_block = "\n".join(context_parts) if context_parts else "(No caption available)"
    return _USER_PROMPT_TEMPLATE.format(context_block=context_block)


# ── Main class ────────────────────────────────────────────────────────────────

class VLCaptioner:
    """
    Wrapper around Qwen2-VL-7B-Instruct for figure description.

    Args:
        model_path    : Path to the local Qwen2-VL-7B-Instruct directory
                        (default: /home/ensta/data/Qwen2-VL-7B-Instruct)
        max_new_tokens: Maximum tokens to generate per description.
        device        : 'cuda' (default) or 'cpu'.
    """

    DEFAULT_MODEL_PATH = "/home/ensta/data/Qwen2-VL-7B-Instruct"

    def __init__(
        self,
        model_path: Optional[str] = None,
        max_new_tokens: int = 512,
        device: str = "cuda",
    ):
        self.model_path = Path(model_path or self.DEFAULT_MODEL_PATH)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Qwen2-VL model directory not found: {self.model_path}\n"
                "Download from HuggingFace: Qwen/Qwen2-VL-7B-Instruct"
            )
        self.max_new_tokens = max_new_tokens
        self.device = device

        self._model = None
        self._processor = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load model and processor onto GPU (call once before describe())."""
        if self._model is not None:
            return  # already loaded

        logger.info("Loading Qwen2-VL from %s …", self.model_path)
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self._model.eval()

        self._processor = AutoProcessor.from_pretrained(
            str(self.model_path),
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )
        logger.info("Qwen2-VL loaded successfully.")

    def unload(self) -> None:
        """Release GPU memory."""
        if self._model is not None:
            import torch
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            torch.cuda.empty_cache()
            logger.info("Qwen2-VL unloaded.")

    # ── Core method ───────────────────────────────────────────────────────────

    def describe(self, figure) -> str:
        """
        Generate a VL description for a FigureRecord.

        Stores result in figure.metadata['vl_description'] and also returns it.

        Args:
            figure: A FigureRecord instance (from image_extractor.py).

        Returns:
            The generated description string.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call captioner.load() first.")

        description = self._generate(
            image_bytes=figure.image_bytes,
            caption=figure.caption,
            section=figure.section,
        )
        figure.metadata["vl_description"] = description
        return description

    def describe_batch(self, figures: List) -> List[str]:
        """
        Describe multiple figures sequentially (one at a time; VL models are memory-heavy).
        """
        descriptions = []
        for i, fig in enumerate(figures):
            logger.info("[%d/%d] Describing %s …", i + 1, len(figures), fig.figure_id)
            desc = self.describe(fig)
            descriptions.append(desc)
        return descriptions

    # ── Internal ──────────────────────────────────────────────────────────────

    def _generate(self, image_bytes: bytes, caption: str = "", section: str = "") -> str:
        """
        Run Qwen2-VL inference for a single image.

        Args:
            image_bytes: PNG/JPEG bytes of the figure.
            caption    : Detected caption text (may be empty).
            section    : Section heading above the image (may be empty).

        Returns:
            Generated description string.
        """
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        user_text = _build_user_prompt(caption=caption, section=section)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": user_text},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        import torch
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        # Decode only the newly generated tokens
        new_ids = output_ids[:, inputs.input_ids.shape[1]:]
        description = self._processor.batch_decode(
            new_ids, skip_special_tokens=True
        )[0].strip()

        return description
