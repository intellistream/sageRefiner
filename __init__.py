"""
sageRefiner - Intelligent Context Compression for RAG
======================================================

Standalone library providing state-of-the-art context compression algorithms.

Quick Start:
    >>> from sageRefiner import LongRefiner, RefinerConfig
    >>> config = RefinerConfig(algorithm="long_refiner", budget=2048)
    >>> refiner = LongRefiner(config.to_dict())
    >>> refiner.initialize()
    >>> result = refiner.refine(query, documents, budget=2048)

Available Algorithms:
    - LongRefiner: Advanced selective compression with LLM-based importance scoring
    - Reform: Efficient reformulation-based compression

For SAGE framework integration, use sage-middleware's RefinerAdapter instead.
"""

__version__ = "0.1.0"
__author__ = "SAGE Team"
__license__ = "Apache-2.0"

# Configuration
from .config import RefinerAlgorithm, RefinerConfig

# Algorithms
from .algorithms.LongRefiner.compressor import LongRefiner
from .algorithms.reform.compressor import ReformCompressor

__all__ = [
    # Config
    "RefinerConfig",
    "RefinerAlgorithm",
    # Algorithms
    "LongRefiner",
    "ReformCompressor",
    # Metadata
    "__version__",
]
