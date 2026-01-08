"""
RECOMP Extractive Algorithm Implementation
===========================================

RECOMP (Retrieval-Oriented Compression) Extractive algorithm for RAG context compression.

核心思路：
使用双编码器（Contriever/DPR）对检索文档进行句子级打分，
选择与 query 最相关的 top-k 句子作为压缩后的上下文。

支持的模型：
- fangyuan/nq_extractive_compressor (NQ 数据集微调)
- fangyuan/tqa_extractive_compressor (TriviaQA 微调)
- fangyuan/hotpotqa_extractive_compressor (HotpotQA 微调)
- facebook/contriever-msmarco (通用检索模型)

References:
    RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation
    https://arxiv.org/pdf/2310.04408.pdf
"""

from .compressor import RECOMPExtractiveCompressor

__all__ = [
    "RECOMPExtractiveCompressor",
]

# Optional: SAGE operator (only available inside SAGE framework)
try:
    from .operator import RECOMPExtractiveRefinerOperator

    __all__.append("RECOMPExtractiveRefinerOperator")
except ImportError:
    RECOMPExtractiveRefinerOperator = None
