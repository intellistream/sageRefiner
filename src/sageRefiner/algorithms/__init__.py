"""
Refiner Algorithms
==================

Collection of context compression and refinement algorithms.
"""

# Core compressors (always available)
from .LongRefiner import LongRefinerCompressor
from .reform import AttentionHookExtractor, REFORMCompressor

__all__ = [
    # REFORM
    "REFORMCompressor",
    "AttentionHookExtractor",
    # LongRefiner
    "LongRefinerCompressor",
]

# Optional: SAGE operators (only when running inside SAGE framework)
try:
    from .LongRefiner import LongRefinerOperator
    from .reform import REFORMRefinerOperator

    __all__.extend(["LongRefinerOperator", "REFORMRefinerOperator"])
except ImportError:
    # Running standalone without SAGE - operators not available
    LongRefinerOperator = None
    REFORMRefinerOperator = None
