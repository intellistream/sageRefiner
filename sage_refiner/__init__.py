"""
sage_refiner - Intelligent Context Compression for RAG
======================================================

Standalone library providing state-of-the-art context compression algorithms.

Quick Start:
    >>> from sage_refiner import LongRefinerCompressor, RefinerConfig
    >>> config = RefinerConfig(algorithm="long_refiner", budget=2048)
    >>> refiner = LongRefinerCompressor(config.to_dict())
    >>> result = refiner.compress(question, documents, budget=2048)

Available Algorithms:
    - LongRefinerCompressor: Advanced selective compression with LLM-based importance scoring
    - REFORMCompressor: Efficient attention-based compression
    - ProvenceCompressor: Sentence-level context pruning
    - AdaptiveCompressor: Query-aware multi-granularity compression (NEW)

For SAGE framework integration, use sage-middleware's RefinerAdapter instead.
"""

__version__ = "0.1.0"
__author__ = "SAGE Team"
__license__ = "Apache-2.0"

from .algorithms.LongRefiner.compressor import LongRefinerCompressor
from .algorithms.provence.compressor import ProvenceCompressor
from .algorithms.reform.compressor import REFORMCompressor
from .config import RefinerAlgorithm, RefinerConfig

# Aliases for convenience
LongRefiner = LongRefinerCompressor
ReformCompressor = REFORMCompressor

__all__ = [
    # Config
    "RefinerConfig",
    "RefinerAlgorithm",
    # Algorithms
    "LongRefinerCompressor",
    "REFORMCompressor",
    "ProvenceCompressor",
    # Aliases
    "LongRefiner",
    "ReformCompressor",
    # Metadata
    "__version__",
]

# AdaptiveCompressor (new algorithm)
try:
    from .algorithms.adaptive.compressor import AdaptiveCompressor

    __all__.append("AdaptiveCompressor")
except ImportError:
    AdaptiveCompressor = None
