"""
REFORM Algorithm Implementation
================================

REFORM (Compress, Gather, and Recompute) algorithm for RAG context compression.

References:
    REFORM: Compress, Gather, and Recompute (Appendix B.5)
"""

from .compressor import REFORMCompressor
from .model_utils import AttentionHookExtractor

__all__ = [
    "REFORMCompressor",
    "AttentionHookExtractor",
]

# Optional: SAGE operator (only available inside SAGE framework)
try:
    from .operator import REFORMRefinerOperator

    __all__.append("REFORMRefinerOperator")
except ImportError:
    REFORMRefinerOperator = None
