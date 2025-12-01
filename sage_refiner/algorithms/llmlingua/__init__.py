"""
LLMLingua Compressor Integration
================================

微软 LLMLingua 提示词压缩算法的 SAGE 集成。

LLMLingua 是一种基于 perplexity 的提示词压缩方法：
- LLMLingua-1: 使用小型 LLM 计算 token perplexity，移除低信息量 token
- LLMLingua-2: 使用 BERT 进行 token 级分类，判断是否保留

论文:
- LLMLingua: https://arxiv.org/abs/2310.05736
- LLMLingua-2: https://arxiv.org/abs/2403.12968

官方库: https://github.com/microsoft/LLMLingua
"""

from .compressor import LLMLinguaCompressor

__all__ = [
    "LLMLinguaCompressor",
]

# Optional: SAGE Operator (only when running inside SAGE framework)
try:
    from .operator import LLMLinguaRefinerOperator

    __all__.append("LLMLinguaRefinerOperator")
except ImportError:
    # Running standalone without SAGE - operator not available
    LLMLinguaRefinerOperator = None
