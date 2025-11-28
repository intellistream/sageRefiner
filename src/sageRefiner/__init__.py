"""
sageRefiner - Intelligent Context Compression for RAG
======================================================

Standalone library providing state-of-the-art context compression algorithms.

Quick Start:
    >>> from sageRefiner import LongRefinerCompressor, RefinerConfig
    >>> config = RefinerConfig(algorithm="long_refiner", budget=2048)
    >>> refiner = LongRefinerCompressor(config.to_dict())
    >>> result = refiner.compress(question, documents, budget=2048)

Available Algorithms:
    - LongRefinerCompressor: Advanced selective compression with LLM-based importance scoring
    - REFORMCompressor: Efficient attention-based compression

For SAGE framework integration, use sage-middleware's RefinerAdapter instead.
"""

__version__ = "0.1.0"
__author__ = "SAGE Team"
__license__ = "Apache-2.0"

# Configuration
from .config import RefinerAlgorithm, RefinerConfig

# Algorithms
from .algorithms.LongRefiner.compressor import LongRefinerCompressor
from .algorithms.reform.compressor import REFORMCompressor

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
    # Aliases
    "LongRefiner",
    "ReformCompressor",
    # Metadata
    "__version__",
]
