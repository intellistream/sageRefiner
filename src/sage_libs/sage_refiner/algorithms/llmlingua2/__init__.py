"""
LLMLingua-2 Algorithm Implementation
====================================

LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression.

LLMLingua-2 is a BERT-based token classification approach for fast prompt compression.
Unlike LLMLingua (which uses LLM perplexity), LLMLingua-2 uses a fine-tuned BERT model
to classify which tokens to keep, resulting in much faster compression.

Features:
    - Fast compression via token classification (no LLM inference needed)
    - Multilingual support via XLM-RoBERTa
    - Token-level precise compression
    - Compatible with context-level filtering

References:
    LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression
    Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, et al.
    arXiv preprint arXiv:2403.12968 (2024)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .compressor import LLMLingua2Compressor

__all__ = [
    "LLMLingua2Compressor",
]

# Type hint for optional import
if TYPE_CHECKING:
    from .operator import LLMLingua2RefinerOperator as _LLMLingua2RefinerOperator

# Optional: SAGE operator (only available inside SAGE framework)
LLMLingua2RefinerOperator: type[_LLMLingua2RefinerOperator] | None
try:
    from .operator import LLMLingua2RefinerOperator

    __all__.append("LLMLingua2RefinerOperator")
except ImportError:
    LLMLingua2RefinerOperator = None
