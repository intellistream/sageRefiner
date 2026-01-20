"""
sage_refiner - Intelligent Context Compression for LLM
======================================================

Standalone library providing state-of-the-art context compression algorithms.

Quick Start:
    >>> from sage_refiner import LLMLingua2Compressor
    >>> compressor = LLMLingua2Compressor()
    >>> result = compressor.compress(context, question, target_token=500)

Available Algorithms:
    - LongRefinerCompressor: LLM-based selective compression
    - ProvenceCompressor: Sentence-level pruning
    - LLMLingua2Compressor: Fast BERT-based compression
    - LongLLMLinguaCompressor: Perplexity-based compression
    - RECOMPAbstractiveCompressor: T5-based summarization
    - RECOMPExtractiveCompressor: Sentence extraction
    - EHPCCompressor: Evaluator heads compression
"""

from ._version import __author__, __email__, __version__

__license__ = "Apache-2.0"

from .algorithms.llmlingua2 import LLMLingua2RefinerOperator
from .algorithms.llmlingua2.compressor import LLMLingua2Compressor
from .algorithms.longllmlingua import LongLLMLinguaRefinerOperator
from .algorithms.longllmlingua.compressor import (
    DEFAULT_LONG_LLMLINGUA_CONFIG,
    LongLLMLinguaCompressor,
)
from .algorithms.LongRefiner.compressor import LongRefinerCompressor
from .algorithms.provence.compressor import ProvenceCompressor
from .algorithms.recomp_abst import RECOMPAbstractiveRefinerOperator
from .algorithms.recomp_abst.compressor import RECOMPAbstractiveCompressor
from .algorithms.recomp_extr import RECOMPExtractiveRefinerOperator
from .algorithms.recomp_extr.compressor import RECOMPExtractiveCompressor
from .config import RefinerAlgorithm, RefinerConfig

# Aliases for convenience
LongRefiner = LongRefinerCompressor

__all__ = [
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    # Config
    "RefinerConfig",
    "RefinerAlgorithm",
    # Algorithms - Compressors
    "LongRefinerCompressor",
    "LLMLingua2Compressor",
    "LongLLMLinguaCompressor",
    "ProvenceCompressor",
    "RECOMPAbstractiveCompressor",
    "RECOMPExtractiveCompressor",
    # Algorithms - Operators
    "LLMLingua2RefinerOperator",
    "LongLLMLinguaRefinerOperator",
    "RECOMPAbstractiveRefinerOperator",
    "RECOMPExtractiveRefinerOperator",
    # Config constants
    "DEFAULT_LONG_LLMLINGUA_CONFIG",
    # Aliases
    "LongRefiner",
]

# AdaptiveCompressor (new algorithm)
try:
    from .algorithms.adaptive.compressor import AdaptiveCompressor

    __all__.append("AdaptiveCompressor")
except ImportError:
    AdaptiveCompressor = None
