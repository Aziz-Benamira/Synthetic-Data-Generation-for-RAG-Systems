"""
deepseek_wrapper.py
===================

Thin wrapper around a llama_cpp Llama instance that strips
<think>...</think> blocks from every create_chat_completion response.

DeepSeek R1 (and other reasoning models) emit chain-of-thought reasoning
wrapped in <think>...</think> tags BEFORE the final answer.  Downstream
code (CriticV4 metrics, generators) expects clean JSON without those tags.

Usage:
    from deepseek_wrapper import DeepSeekR1Wrapper

    raw_llm = Llama(model_path=..., n_gpu_layers=-1, n_ctx=16384)
    llm = DeepSeekR1Wrapper(raw_llm)   # drop-in replacement

    # All callers automatically get clean output:
    #   create_chat_completion() → think tags stripped from content
"""

import re
import logging

logger = logging.getLogger(__name__)

# Matches <think>...</think> including multiline and nested content.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Remove all <think>...</think> blocks and strip surrounding whitespace."""
    return _THINK_RE.sub("", text).strip()


class DeepSeekR1Wrapper:
    """
    Drop-in replacement for a llama_cpp Llama object.

    Strips <think>...</think> from the 'content' field of every
    create_chat_completion() response before returning it.

    All other attributes/methods are forwarded to the underlying LLM.
    """

    def __init__(self, llm):
        # Use object.__setattr__ to avoid triggering __setattr__ below
        object.__setattr__(self, "_llm", llm)

    def create_chat_completion(self, messages, **kwargs):
        response = self._llm.create_chat_completion(messages=messages, **kwargs)

        for choice in response.get("choices", []):
            msg = choice.get("message", {})
            content = msg.get("content", "")
            if content and "<think>" in content:
                cleaned = strip_think(content)
                msg["content"] = cleaned
                logger.debug(
                    f"[DeepSeekR1Wrapper] Stripped think block "
                    f"({len(content) - len(cleaned)} chars removed)"
                )

        return response

    # ── Forward everything else to the underlying LLM ────────────────────────

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_llm"), name)

    def __setattr__(self, name, value):
        if name == "_llm":
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_llm"), name, value)
