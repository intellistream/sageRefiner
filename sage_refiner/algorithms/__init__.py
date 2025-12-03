"""
Refiner Algorithms
==================

Collection of context compression and refinement algorithms.

Available Algorithms:
    - REFORM: Attention-head driven token selection for RAG context compression
    - RECOMP Extractive: Sentence-level extractive compression with dual encoders
    - RECOMP Abstractive: T5-based abstractive summarization compression
    - LongRefiner: Long document refinement with sliding window
    - Provence: Provenance-aware context compression
    - LongLLMLingua: Question-aware prompt compression for long documents
    - LLMLingua2: Fast BERT-based token classification compression
"""

# Core compressors (always available)
from .LongRefiner import LongRefinerCompressor
from .provence import ProvenceCompressor
from .recomp_abst import RECOMPAbstractiveCompressor
from .recomp_extr import RECOMPExtractiveCompressor
from .reform import AttentionHookExtractor, REFORMCompressor

__all__ = [
    # REFORM
    "REFORMCompressor",
    "AttentionHookExtractor",
    # RECOMP Extractive
    "RECOMPExtractiveCompressor",
    # RECOMP Abstractive
    "RECOMPAbstractiveCompressor",
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
    from .recomp_abst import RECOMPAbstractiveOperator
    from .recomp_extr import RECOMPExtractiveOperator
    from .reform import REFORMRefinerOperator

    __all__.extend([
        "LongRefinerOperator",
        "REFORMRefinerOperator",
        "ProvenceRefinerOperator",
        "RECOMPExtractiveOperator",
        "RECOMPAbstractiveOperator",
    ])
except ImportError:
    # Running standalone without SAGE - operators not available
    LongRefinerOperator = None
    REFORMRefinerOperator = None
    ProvenceRefinerOperator = None
    RECOMPExtractiveOperator = None
    RECOMPAbstractiveOperator = None

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
