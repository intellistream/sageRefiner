"""
LongLLMLingua Algorithm Implementation
======================================

LongLLMLingua is an extension of LLMLingua optimized for long document scenarios.
It features question-aware context ranking and dynamic compression.

Key Features:
    - Question-aware context ranking (rank_method="longllmlingua")
    - Condition evaluation after question (condition_in_question="after")
    - Context reordering by relevance (reorder_context="sort")
    - Dynamic compression ratio adjustment

References:
    LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression
    https://arxiv.org/abs/2310.06839
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .compressor import DEFAULT_LONG_LLMLINGUA_CONFIG, LongLLMLinguaCompressor

__all__ = [
    "LongLLMLinguaCompressor",
    "DEFAULT_LONG_LLMLINGUA_CONFIG",
]

# Type hint for optional import
if TYPE_CHECKING:
    from .operator import LongLLMLinguaRefinerOperator as _LongLLMLinguaRefinerOperator

# Optional: SAGE operator (only available inside SAGE framework)
LongLLMLinguaRefinerOperator: type[_LongLLMLinguaRefinerOperator] | None
try:
    from .operator import LongLLMLinguaRefinerOperator

    __all__.append("LongLLMLinguaRefinerOperator")
except ImportError:
    LongLLMLinguaRefinerOperator = None
