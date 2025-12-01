"""
Adaptive Context Compressor (自适应上下文压缩器)
================================================

核心压缩器实现，整合所有组件:
1. Query 分类
2. 粒度选择
3. 信息密度优化
4. 动态预算分配

创新点:
- **Query 感知**: 根据 query 类型调整压缩策略
- **多粒度级联**: 段落→句子→短语 级联压缩
- **MMR 多样性**: 避免冗余，最大化信息覆盖
- **动态预算**: 复杂 query 自动增加预算
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CompressionResult:
    """压缩结果"""

    compressed_context: str
    original_tokens: int
    compressed_tokens: int
    compression_rate: float

    # 额外信息
    query_type: str = ""
    granularity_used: str = ""
    num_units_selected: int = 0
    num_units_total: int = 0
    processing_time_ms: float = 0.0

    # 统计
    relevance_stats: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "compressed_context": self.compressed_context,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "compression_rate": self.compression_rate,
            "query_type": self.query_type,
            "granularity_used": self.granularity_used,
            "num_units_selected": self.num_units_selected,
            "num_units_total": self.num_units_total,
            "processing_time_ms": self.processing_time_ms,
            "relevance_stats": self.relevance_stats,
        }


class AdaptiveCompressor:
    """
    自适应上下文压缩器

    整合 Query 分类、粒度选择、信息密度分析的完整压缩器。

    使用流程:
    1. 分析 Query → 确定类型、策略、预算乘数
    2. 调整预算 → 基础预算 × 预算乘数
    3. 选择粒度 → 根据调整后预算选择段落/句子/短语级
    4. 分割文本 → 按选定粒度分割
    5. 多样性选择 → MMR 算法选择片段
    6. 组装结果 → 拼接并返回

    Usage:
        compressor = AdaptiveCompressor()
        result = compressor.compress(
            context="...",
            question="What is quantum computing?",
            budget=2048,
        )
        print(result.compressed_context)
        print(f"Compression: {result.compression_rate:.1%}")
    """

    def __init__(
        self,
        tokenizer=None,
        encoder=None,
        use_query_classifier: bool = True,
        use_ner: bool = False,  # NER 较慢，默认关闭
        similarity_threshold: float = 0.85,
        diversity_weight: float = 0.3,
        min_budget: int = 256,
        max_budget_multiplier: float = 2.0,
    ):
        """
        初始化

        Args:
            tokenizer: Tokenizer (如果为 None，自动加载)
            encoder: 编码器 (如果为 None，自动加载)
            use_query_classifier: 是否启用 Query 分类
            use_ner: 是否启用 NER (较慢)
            similarity_threshold: 相似度阈值
            diversity_weight: 多样性权重
            min_budget: 最小预算
            max_budget_multiplier: 最大预算乘数
        """
        self.use_query_classifier = use_query_classifier
        self.similarity_threshold = similarity_threshold
        self.diversity_weight = diversity_weight
        self.min_budget = min_budget
        self.max_budget_multiplier = max_budget_multiplier

        # 加载 tokenizer
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self._load_tokenizer()

        # 初始化组件
        self._init_components(encoder, use_ner)

        logger.info(
            f"AdaptiveCompressor initialized: "
            f"query_classifier={use_query_classifier}, "
            f"diversity_weight={diversity_weight}"
        )

    def _load_tokenizer(self) -> None:
        """加载默认 tokenizer"""
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            logger.info("Loaded default tokenizer: bert-base-uncased")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise

    def _init_components(self, encoder, use_ner: bool) -> None:
        """初始化子组件"""
        from .query_classifier import QueryClassifier
        from .granularity import GranularitySelector
        from .density import InformationDensityAnalyzer

        # Query 分类器
        if self.use_query_classifier:
            self.query_classifier = QueryClassifier(use_ner=use_ner)
        else:
            self.query_classifier = None

        # 粒度选择器
        self.granularity_selector = GranularitySelector()

        # 信息密度分析器
        self.density_analyzer = InformationDensityAnalyzer(
            encoder=encoder,
            similarity_threshold=self.similarity_threshold,
            diversity_weight=self.diversity_weight,
        )

    def compress(
        self,
        context: str,
        question: str,
        budget: int = 2048,
        force_granularity: str | None = None,
    ) -> CompressionResult:
        """
        压缩上下文

        Args:
            context: 原始上下文
            question: 问题
            budget: 基础 token 预算
            force_granularity: 强制使用的粒度 (paragraph/sentence/phrase)

        Returns:
            CompressionResult
        """
        start_time = time.time()

        # Step 0: 计算原始 token 数
        original_tokens = len(self.tokenizer.encode(context))

        # 如果原始上下文已经小于预算，直接返回
        if original_tokens <= budget:
            return CompressionResult(
                compressed_context=context,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_rate=1.0,
                query_type="N/A",
                granularity_used="none",
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        # Step 1: Query 分析
        if self.use_query_classifier and self.query_classifier:
            query_analysis = self.query_classifier.analyze(question)
            query_type = query_analysis.query_type.value
            budget_multiplier = min(
                query_analysis.budget_multiplier, self.max_budget_multiplier
            )
            compression_strategy = query_analysis.compression_strategy
        else:
            query_type = "unknown"
            budget_multiplier = 1.0
            compression_strategy = "balanced"

        # Step 2: 调整预算
        adjusted_budget = max(int(budget * budget_multiplier), self.min_budget)
        logger.debug(
            f"Budget adjustment: {budget} → {adjusted_budget} "
            f"(multiplier={budget_multiplier:.2f}, query_type={query_type})"
        )

        # Step 3: 选择粒度
        if force_granularity:
            from .granularity import Granularity

            granularity = Granularity(force_granularity)
        else:
            granularity = self.granularity_selector.recommend_granularity(
                original_tokens=original_tokens,
                budget=adjusted_budget,
                query_type=query_type,
            )

        logger.debug(f"Selected granularity: {granularity.value}")

        # Step 4: 分割文本
        units = self.granularity_selector.cascade_segment(
            text=context,
            target_budget=adjusted_budget,
            tokenizer=self.tokenizer,
        )

        if not units:
            # 无法分割，返回原始（截断）
            truncated = self.tokenizer.decode(
                self.tokenizer.encode(context)[:adjusted_budget]
            )
            return CompressionResult(
                compressed_context=truncated,
                original_tokens=original_tokens,
                compressed_tokens=adjusted_budget,
                compression_rate=adjusted_budget / original_tokens,
                query_type=query_type,
                granularity_used="truncation",
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        # Step 5: 多样性选择
        selected_units = self.density_analyzer.select_diverse(
            query=question,
            units=units,
            budget=adjusted_budget,
            tokenizer=self.tokenizer,
        )

        # Step 6: 组装结果
        # 按原始位置排序
        selected_units.sort(key=lambda u: u.start_char)

        # 拼接
        compressed_text = self._assemble_text(selected_units)

        # 计算压缩后 token 数
        compressed_tokens = len(self.tokenizer.encode(compressed_text))

        # 统计
        relevance_stats = {}
        if selected_units:
            scores = [u.score for u in selected_units if u.score > 0]
            if scores:
                import numpy as np

                relevance_stats = {
                    "mean": float(np.mean(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                }

        processing_time = (time.time() - start_time) * 1000

        return CompressionResult(
            compressed_context=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_rate=compressed_tokens / original_tokens,
            query_type=query_type,
            granularity_used=granularity.value,
            num_units_selected=len(selected_units),
            num_units_total=len(units),
            processing_time_ms=processing_time,
            relevance_stats=relevance_stats,
        )

    def _assemble_text(self, units: list) -> str:
        """
        组装压缩后的文本

        根据单元之间的距离决定分隔符:
        - 相邻 → 空格
        - 跳过内容 → 换行
        """
        if not units:
            return ""

        parts = []
        prev_end = 0

        for unit in units:
            if parts and unit.start_char > prev_end + 50:
                # 跳过了较多内容，用换行分隔
                parts.append("\n\n")
            elif parts:
                parts.append(" ")

            parts.append(unit.text.strip())
            prev_end = unit.end_char

        return "".join(parts)

    def batch_compress(
        self,
        contexts: list[str],
        questions: list[str],
        budget: int = 2048,
    ) -> list[CompressionResult]:
        """
        批量压缩

        Args:
            contexts: 上下文列表
            questions: 问题列表
            budget: Token 预算

        Returns:
            CompressionResult 列表
        """
        results = []
        for ctx, q in zip(contexts, questions):
            result = self.compress(ctx, q, budget)
            results.append(result)
        return results
