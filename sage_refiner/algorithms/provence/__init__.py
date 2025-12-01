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

__all__ = [
    "ProvenceCompressor",
]

# Optional: SAGE operator (only when running inside SAGE framework)
try:
    from .operator import ProvenceRefinerOperator

    __all__.append("ProvenceRefinerOperator")
except ImportError:
    # Running standalone without SAGE - operator not available
    ProvenceRefinerOperator = None
