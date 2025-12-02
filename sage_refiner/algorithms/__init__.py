"""
Refiner Algorithms
==================

Collection of context compression and refinement algorithms.

Available Algorithms:
    - REFORM: Attention-head driven token selection for RAG context compression
    - LongRefiner: Long document refinement with sliding window
    - Provence: Provenance-aware context compression
    - LongLLMLingua: Question-aware prompt compression for long documents
    - LLMLingua2: Fast BERT-based token classification compression
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

# LongLLMLingua: Question-aware compression for long documents
try:
    from .longllmlingua import LongLLMLinguaCompressor

    __all__.append("LongLLMLinguaCompressor")
except ImportError:
    LongLLMLinguaCompressor = None

# Optional: LLMLingua-2 compressor (requires LLMLingua dependencies)
try:
    from .llmlingua2 import LLMLingua2Compressor

    __all__.append("LLMLingua2Compressor")
except ImportError:
    LLMLingua2Compressor = None

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

# LongLLMLingua operator (requires SAGE framework + LLMLingua)
try:
    from .longllmlingua import LongLLMLinguaOperator

    __all__.append("LongLLMLinguaOperator")
except ImportError:
    LongLLMLinguaOperator = None

# Optional: LLMLingua-2 operator (requires SAGE framework + LLMLingua)
try:
    from .llmlingua2 import LLMLingua2Operator

    __all__.append("LLMLingua2Operator")
except ImportError:
    LLMLingua2Operator = None
