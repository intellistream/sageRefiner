"""
Refiner Algorithms
==================

Collection of context compression and refinement algorithms.
"""

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

# Optional: SAGE operators (only when running inside SAGE framework)
try:
    from .LongRefiner import LongRefinerOperator
    from .provence import ProvenceRefinerOperator
    from .reform import REFORMRefinerOperator

    __all__.extend(["LongRefinerOperator", "REFORMRefinerOperator", "ProvenceRefinerOperator"])
except ImportError:
    # Running standalone without SAGE - operators not available
    LongRefinerOperator = None
    REFORMRefinerOperator = None
    ProvenceRefinerOperator = None
