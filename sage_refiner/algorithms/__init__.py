"""
Refiner Algorithms
==================

Collection of context compression and refinement algorithms.

Algorithms:
- LongRefiner: Three-stage LLM-driven compression
- REFORM: Attention-head based token-level compression
- Provence: DeBERTa sentence-level pruning
- Adaptive: Multi-granularity query-aware compression
- LLMLingua: Perplexity-based token-level compression (Microsoft)
"""

import logging

# Core compressors (always available)
from .LongRefiner import LongRefinerCompressor
from .provence import ProvenceCompressor
from .reform import AttentionHookExtractor, REFORMCompressor

__all__ = [
    # REFORM
    "REFORMCompressor",
    "AttentionHookExtractor",
    # LongRefiner
    "LongRefinerCompressor",
    # Provence
    "ProvenceCompressor",
]

_logger = logging.getLogger(__name__)

# Adaptive compressor (new algorithm)
try:
    from .adaptive import (
        AdaptiveCompressor,
        Granularity,  # noqa: F401
        GranularitySelector,  # noqa: F401
        InformationDensityAnalyzer,  # noqa: F401
        QueryClassifier,
        QueryType,  # noqa: F401
    )

    __all__.extend(
        [
            "AdaptiveCompressor",
            "QueryClassifier",
            "QueryType",
            "GranularitySelector",
            "Granularity",
            "InformationDensityAnalyzer",
        ]
    )
except ImportError as e:
    # Dependencies not available
    AdaptiveCompressor = None
    QueryClassifier = None
    _logger.debug(f"Adaptive compressor not available: {e}")

# LLMLingua compressor (optional dependency: llmlingua)
try:
    from .llmlingua import LLMLinguaCompressor

    __all__.append("LLMLinguaCompressor")
except ImportError as e:
    # llmlingua not installed
    LLMLinguaCompressor = None
    _logger.debug(f"LLMLingua compressor not available: {e}")

# Optional: SAGE operators (only when running inside SAGE framework)
try:
    from .adaptive import AdaptiveRefinerOperator
    from .LongRefiner import LongRefinerOperator
    from .provence import ProvenceRefinerOperator
    from .reform import REFORMRefinerOperator

    __all__.extend(
        [
            "LongRefinerOperator",
            "REFORMRefinerOperator",
            "ProvenceRefinerOperator",
            "AdaptiveRefinerOperator",
        ]
    )
except ImportError:
    # Running standalone without SAGE - operators not available
    LongRefinerOperator = None
    REFORMRefinerOperator = None
    ProvenceRefinerOperator = None
    AdaptiveRefinerOperator = None

# LLMLingua operator (optional)
try:
    from .llmlingua import LLMLinguaRefinerOperator

    __all__.append("LLMLinguaRefinerOperator")
except ImportError:
    LLMLinguaRefinerOperator = None
