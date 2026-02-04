"""
Refiner Algorithms
==================

Collection of context compression and refinement algorithms.

Available Algorithms:
    - REFORM: Attention-head based efficient compression with KV cache optimization
    - RECOMP Extractive: Sentence-level extractive compression with dual encoders
    - RECOMP Abstractive: T5-based abstractive summarization compression
    - LongRefiner: Long document refinement with sliding window
    - Provence: Provenance-aware context compression
    - LongLLMLingua: Question-aware prompt compression for long documents
    - LLMLingua2: Fast BERT-based token classification compression
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

# Type hints for optional imports
if TYPE_CHECKING:
    from .llmlingua2.compressor import LLMLingua2Compressor as _LLMLingua2Compressor
    from .llmlingua2.operator import LLMLingua2RefinerOperator as _LLMLingua2RefinerOperator
    from .longllmlingua.compressor import LongLLMLinguaCompressor as _LongLLMLinguaCompressor
    from .longllmlingua.operator import (
        LongLLMLinguaRefinerOperator as _LongLLMLinguaRefinerOperator,
    )
    from .LongRefiner.operator import LongRefinerOperator as _LongRefinerOperator
    from .provence.operator import ProvenceRefinerOperator as _ProvenceRefinerOperator
    from .recomp_abst.operator import (
        RECOMPAbstractiveRefinerOperator as _RECOMPAbstractiveRefinerOperator,
    )
    from .recomp_extr.operator import (
        RECOMPExtractiveRefinerOperator as _RECOMPExtractiveRefinerOperator,
    )
    from .reform.operator import REFORMRefinerOperator as _REFORMRefinerOperator

# LongLLMLingua: Question-aware compression for long documents
LongLLMLinguaCompressor: type[_LongLLMLinguaCompressor] | None
try:
    from .longllmlingua import LongLLMLinguaCompressor

    __all__.append("LongLLMLinguaCompressor")
except ImportError:
    LongLLMLinguaCompressor = None

# Optional: LLMLingua-2 compressor (requires LLMLingua dependencies)
LLMLingua2Compressor: type[_LLMLingua2Compressor] | None
try:
    from .llmlingua2 import LLMLingua2Compressor

    __all__.append("LLMLingua2Compressor")
except ImportError:
    LLMLingua2Compressor = None

# Optional: SAGE operators (only when running inside SAGE framework)
LongRefinerOperator: type[_LongRefinerOperator] | None
ProvenceRefinerOperator: type[_ProvenceRefinerOperator] | None
RECOMPExtractiveRefinerOperator: type[_RECOMPExtractiveRefinerOperator] | None
RECOMPAbstractiveRefinerOperator: type[_RECOMPAbstractiveRefinerOperator] | None
REFORMRefinerOperator: type[_REFORMRefinerOperator] | None
try:
    from .LongRefiner import LongRefinerOperator
    from .provence import ProvenceRefinerOperator
    from .recomp_abst import RECOMPAbstractiveRefinerOperator
    from .recomp_extr import RECOMPExtractiveRefinerOperator
    from .reform import REFORMRefinerOperator

    __all__.extend(
        [
            "LongRefinerOperator",
            "ProvenceRefinerOperator",
            "RECOMPExtractiveRefinerOperator",
            "RECOMPAbstractiveRefinerOperator",
            "REFORMRefinerOperator",
        ]
    )
except ImportError:
    # Running standalone without SAGE - operators not available
    LongRefinerOperator = None
    ProvenceRefinerOperator = None
    RECOMPExtractiveRefinerOperator = None
    RECOMPAbstractiveRefinerOperator = None
    REFORMRefinerOperator = None

# LongLLMLingua operator (requires SAGE framework + LLMLingua)
LongLLMLinguaRefinerOperator: type[_LongLLMLinguaRefinerOperator] | None
try:
    from .longllmlingua import LongLLMLinguaRefinerOperator

    __all__.append("LongLLMLinguaRefinerOperator")
except ImportError:
    LongLLMLinguaRefinerOperator = None

# Optional: LLMLingua-2 operator (requires SAGE framework + LLMLingua)
LLMLingua2RefinerOperator: type[_LLMLingua2RefinerOperator] | None
try:
    from .llmlingua2 import LLMLingua2RefinerOperator

    __all__.append("LLMLingua2RefinerOperator")
except ImportError:
    LLMLingua2RefinerOperator = None
