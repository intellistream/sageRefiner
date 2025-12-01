"""
Adaptive Context Compressor (自适应上下文压缩器)
================================================

一种新的 RAG 上下文压缩算法，核心创新点：

1. **多粒度压缩 (Multi-Granularity)**
   - 段落级 → 句子级 → 短语级 的级联压缩
   - 根据压缩预算动态选择粒度

2. **Query 感知策略 (Query-Aware)**
   - 事实型 Query → 精确匹配，保留关键实体上下文
   - 推理型 Query → 保留因果链和逻辑结构
   - 多跳型 Query → 保留桥接实体和中间步骤

3. **信息密度优化 (Information Density)**
   - 基于 Pointwise Mutual Information (PMI) 的冗余检测
   - 去除重复信息，保留互补信息

4. **动态预算分配 (Dynamic Budget)**
   - 根据 query 复杂度分配压缩预算
   - 复杂 query 保留更多上下文
"""

from .compressor import AdaptiveCompressor
from .density import InformationDensityAnalyzer
from .granularity import Granularity, GranularitySelector
from .query_classifier import QueryClassifier, QueryType

__all__ = [
    "AdaptiveCompressor",
    "QueryClassifier",
    "QueryType",
    "GranularitySelector",
    "Granularity",
    "InformationDensityAnalyzer",
]

# Optional: SAGE Operator (only when running inside SAGE framework)
try:
    from .operator import AdaptiveRefinerOperator

    __all__.append("AdaptiveRefinerOperator")
except ImportError:
    # Running standalone without SAGE - operator not available
    AdaptiveRefinerOperator = None
