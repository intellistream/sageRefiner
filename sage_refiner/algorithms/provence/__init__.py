"""
Provence Algorithm Implementation
==================================

Provence: Sentence-level context pruning for RAG.

Based on the BERGEN framework's implementation of the Provence compressor.
Uses a pre-trained model to score and prune sentences from retrieved contexts.

References:
    https://arxiv.org/abs/2501.16214
    BERGEN: Benchmarking RAG Pipelines
"""

from .compressor import ProvenceCompressor
from .operator import ProvenceRefinerOperator

__all__ = [
    "ProvenceCompressor",
    "ProvenceRefinerOperator",
]
