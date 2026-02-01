"""
RECOMP Abstractive Algorithm Implementation
============================================

RECOMP (Retrieval-Oriented Compression) Abstractive algorithm for RAG context compression.

核心思路：
使用微调的 T5 模型生成检索文档的摘要，
将检索文档压缩为简洁的摘要作为压缩后的上下文。

支持的模型：
- fangyuan/nq_abstractive_compressor (NQ 数据集微调)
- fangyuan/tqa_abstractive_compressor (TriviaQA 微调)
- fangyuan/hotpotqa_abstractive (HotpotQA 微调)
- t5-large / t5-base (通用摘要模型)

References:
    RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation
    https://arxiv.org/pdf/2310.04408.pdf
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .compressor import RECOMPAbstractiveCompressor

__all__ = [
    "RECOMPAbstractiveCompressor",
]

# Type hint for optional import
if TYPE_CHECKING:
    from .operator import RECOMPAbstractiveRefinerOperator as _RECOMPAbstractiveRefinerOperator

# Optional: SAGE operator (only available inside SAGE framework)
RECOMPAbstractiveRefinerOperator: type[_RECOMPAbstractiveRefinerOperator] | None
try:
    from .operator import RECOMPAbstractiveRefinerOperator

    __all__.append("RECOMPAbstractiveRefinerOperator")
except ImportError:
    RECOMPAbstractiveRefinerOperator = None
