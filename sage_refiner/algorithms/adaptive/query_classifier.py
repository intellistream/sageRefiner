"""
Query Classifier (查询分类器)
============================

对输入 Query 进行分类，决定压缩策略。

Query 类型:
- FACTUAL: 事实型 (What/Who/When/Where) - 需要精确匹配
- REASONING: 推理型 (Why/How) - 需要因果链
- MULTI_HOP: 多跳型 (复合问题) - 需要桥接信息
- COMPARISON: 比较型 (A vs B) - 需要多个实体信息
- AGGREGATION: 聚合型 (统计/汇总) - 需要广泛覆盖
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Query 类型枚举"""

    FACTUAL = "factual"  # What is X? Who is Y?
    REASONING = "reasoning"  # Why did X happen? How does Y work?
    MULTI_HOP = "multi_hop"  # X did something, what was the result?
    COMPARISON = "comparison"  # Compare X and Y
    AGGREGATION = "aggregation"  # How many? What are all the...?
    UNKNOWN = "unknown"


@dataclass
class QueryAnalysis:
    """Query 分析结果"""

    query_type: QueryType
    complexity_score: float  # 0-1, 越高越复杂
    key_entities: list[str]  # 关键实体
    expected_answer_type: str  # 期望的答案类型 (person/date/number/explanation)
    compression_strategy: str  # 推荐的压缩策略
    budget_multiplier: float  # 预算乘数 (复杂query需要更多上下文)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_type": self.query_type.value,
            "complexity_score": self.complexity_score,
            "key_entities": self.key_entities,
            "expected_answer_type": self.expected_answer_type,
            "compression_strategy": self.compression_strategy,
            "budget_multiplier": self.budget_multiplier,
        }


class QueryClassifier:
    """
    Query 分类器

    使用规则 + 轻量级模型混合方法进行分类。

    分类流程:
    1. 规则匹配 (疑问词、句式)
    2. 实体识别 (NER)
    3. 复杂度评估 (子句数量、实体数量)
    4. 策略推荐

    Usage:
        classifier = QueryClassifier()
        analysis = classifier.analyze("Why did the Roman Empire fall?")
        print(analysis.query_type)  # QueryType.REASONING
        print(analysis.budget_multiplier)  # 1.5
    """

    # 疑问词模式
    FACTUAL_PATTERNS = [
        r"^what (is|are|was|were)\b",
        r"^who (is|are|was|were)\b",
        r"^when (did|was|were|is)\b",
        r"^where (did|was|were|is)\b",
        r"^which\b",
        r"^name\b",
    ]

    REASONING_PATTERNS = [
        r"^why\b",
        r"^how (did|does|do|can|could)\b",
        r"^explain\b",
        r"^what (caused|led to|resulted in)\b",
        r"\breason\b.*\?",
        r"\bcause\b.*\?",
    ]

    MULTI_HOP_PATTERNS = [
        r"and (then|also|what)\b",
        r"after .+ what\b",
        r"which .+ that\b",
        r"\bthen\b.*\?",
        r"based on .+ what\b",
    ]

    COMPARISON_PATTERNS = [
        r"\bcompare\b",
        r"\bvs\.?\b",
        r"\bversus\b",
        r"\bdifference between\b",
        r"\bsimilar(ity|ities)?\b.*\bwith\b",
        r"(more|less|better|worse) than\b",
    ]

    AGGREGATION_PATTERNS = [
        r"^how many\b",
        r"^how much\b",
        r"\ball\b.*\?",
        r"\blist\b",
        r"\bcount\b",
        r"^what are (the|all)\b",
    ]

    # 策略映射
    STRATEGY_MAP = {
        QueryType.FACTUAL: "entity_focused",  # 聚焦关键实体周围
        QueryType.REASONING: "causal_chain",  # 保留因果链
        QueryType.MULTI_HOP: "bridge_preserving",  # 保留桥接信息
        QueryType.COMPARISON: "multi_entity",  # 保留多个实体
        QueryType.AGGREGATION: "broad_coverage",  # 广泛覆盖
        QueryType.UNKNOWN: "balanced",  # 平衡策略
    }

    # 预算乘数 (复杂query需要更多token)
    BUDGET_MULTIPLIER = {
        QueryType.FACTUAL: 0.8,  # 事实型可以压缩更多
        QueryType.REASONING: 1.3,  # 推理型需要更多上下文
        QueryType.MULTI_HOP: 1.5,  # 多跳型需要最多上下文
        QueryType.COMPARISON: 1.2,  # 比较型适中
        QueryType.AGGREGATION: 1.4,  # 聚合型需要广泛覆盖
        QueryType.UNKNOWN: 1.0,  # 默认
    }

    def __init__(self, use_ner: bool = True, ner_model: str | None = None):
        """
        初始化分类器

        Args:
            use_ner: 是否使用 NER 进行实体识别
            ner_model: NER 模型名称 (默认使用 spaCy)
        """
        self.use_ner = use_ner
        self.ner_pipeline = None

        if use_ner:
            self._load_ner_model(ner_model)

        # 编译正则表达式
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """编译正则表达式"""
        self.factual_re = [re.compile(p, re.IGNORECASE) for p in self.FACTUAL_PATTERNS]
        self.reasoning_re = [re.compile(p, re.IGNORECASE) for p in self.REASONING_PATTERNS]
        self.multi_hop_re = [re.compile(p, re.IGNORECASE) for p in self.MULTI_HOP_PATTERNS]
        self.comparison_re = [re.compile(p, re.IGNORECASE) for p in self.COMPARISON_PATTERNS]
        self.aggregation_re = [re.compile(p, re.IGNORECASE) for p in self.AGGREGATION_PATTERNS]

    def _load_ner_model(self, ner_model: str | None) -> None:
        """加载 NER 模型"""
        try:
            # 优先使用 transformers pipeline
            from transformers import pipeline

            model_name = ner_model or "dslim/bert-base-NER"
            self.ner_pipeline = pipeline("ner", model=model_name, aggregation_strategy="simple")
            logger.info(f"Loaded NER model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load NER model: {e}. Entity extraction disabled.")
            self.use_ner = False

    def analyze(self, query: str) -> QueryAnalysis:
        """
        分析 Query

        Args:
            query: 输入查询

        Returns:
            QueryAnalysis 分析结果
        """
        query = query.strip()

        # Step 1: 分类
        query_type = self._classify_type(query)

        # Step 2: 提取实体
        entities = self._extract_entities(query) if self.use_ner else []

        # Step 3: 计算复杂度
        complexity = self._compute_complexity(query, entities)

        # Step 4: 推断答案类型
        answer_type = self._infer_answer_type(query, query_type)

        # Step 5: 获取策略和预算
        strategy = self.STRATEGY_MAP.get(query_type, "balanced")
        budget_mult = self.BUDGET_MULTIPLIER.get(query_type, 1.0)

        # 复杂度调整预算
        budget_mult *= 1.0 + 0.3 * complexity

        return QueryAnalysis(
            query_type=query_type,
            complexity_score=complexity,
            key_entities=entities,
            expected_answer_type=answer_type,
            compression_strategy=strategy,
            budget_multiplier=min(budget_mult, 2.0),  # 上限2倍
        )

    def _classify_type(self, query: str) -> QueryType:
        """规则分类"""
        # 按优先级检查
        if any(p.search(query) for p in self.multi_hop_re):
            return QueryType.MULTI_HOP

        if any(p.search(query) for p in self.comparison_re):
            return QueryType.COMPARISON

        if any(p.search(query) for p in self.aggregation_re):
            return QueryType.AGGREGATION

        if any(p.search(query) for p in self.reasoning_re):
            return QueryType.REASONING

        if any(p.search(query) for p in self.factual_re):
            return QueryType.FACTUAL

        return QueryType.UNKNOWN

    def _extract_entities(self, query: str) -> list[str]:
        """提取命名实体"""
        if not self.ner_pipeline:
            return []

        try:
            ner_results = self.ner_pipeline(query)
            # 提取实体文本，去重
            entities = list({r["word"] for r in ner_results})
            return entities[:10]  # 最多10个
        except Exception as e:
            logger.warning(f"NER failed: {e}")
            return []

    def _compute_complexity(self, query: str, entities: list[str]) -> float:
        """
        计算 Query 复杂度 (0-1)

        考虑因素:
        - 词数
        - 子句数 (逗号、连词)
        - 实体数量
        - 嵌套结构
        """
        words = query.split()
        word_count = len(words)

        # 子句指标
        clause_indicators = query.count(",") + query.count(" and ") + query.count(" or ")

        # 实体数量
        entity_count = len(entities)

        # 综合评分
        word_score = min(word_count / 30, 1.0)  # 30词以上为复杂
        clause_score = min(clause_indicators / 3, 1.0)  # 3个子句以上为复杂
        entity_score = min(entity_count / 5, 1.0)  # 5个实体以上为复杂

        complexity = 0.4 * word_score + 0.3 * clause_score + 0.3 * entity_score

        return complexity

    def _infer_answer_type(self, query: str, query_type: QueryType) -> str:
        """推断期望的答案类型"""
        query_lower = query.lower()

        # 基于疑问词推断
        if query_lower.startswith("who"):
            return "person"
        if query_lower.startswith("when"):
            return "date"
        if re.match(r"^how (many|much)\b", query_lower):
            return "number"
        if query_lower.startswith("where"):
            return "location"

        # 基于类型推断
        if query_type == QueryType.REASONING:
            return "explanation"
        if query_type == QueryType.COMPARISON:
            return "comparison"
        if query_type == QueryType.AGGREGATION:
            return "list"

        return "text"
