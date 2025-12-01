"""
Granularity Selector (粒度选择器)
================================

动态选择压缩粒度：段落级 → 句子级 → 短语级

核心思想:
- 大预算 → 段落级压缩 (保留完整语义)
- 中等预算 → 句子级压缩 (标准压缩)
- 小预算 → 短语级压缩 (极限压缩)

级联压缩:
1. 先用粗粒度快速筛选
2. 再用细粒度精细压缩
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class Granularity(str, Enum):
    """压缩粒度"""

    PARAGRAPH = "paragraph"  # 段落级 (1-N 句子)
    SENTENCE = "sentence"  # 句子级
    PHRASE = "phrase"  # 短语级 (子句/关键片段)
    TOKEN = "token"  # Token级 (最细粒度，类似REFORM)


@dataclass
class TextUnit:
    """文本单元"""

    text: str
    granularity: Granularity
    start_char: int
    end_char: int
    score: float = 0.0
    metadata: dict[str, Any] | None = None

    @property
    def length(self) -> int:
        return len(self.text)


class GranularitySelector:
    """
    粒度选择器

    根据压缩预算和 query 类型选择最优粒度。

    策略:
    - compression_rate < 0.3 → 短语级
    - compression_rate 0.3-0.6 → 句子级
    - compression_rate > 0.6 → 段落级

    Usage:
        selector = GranularitySelector()
        units = selector.segment(text, Granularity.SENTENCE)
        granularity = selector.recommend_granularity(
            original_tokens=5000,
            budget=1000,
            query_type="reasoning"
        )
    """

    # 句子分割正则
    SENTENCE_PATTERN = re.compile(
        r"(?<=[.!?])\s+(?=[A-Z])|"  # 标点后跟大写
        r"(?<=[.!?])\s*\n+|"  # 标点后换行
        r"\n{2,}"  # 多个换行 (段落边界)
    )

    # 短语分割正则 (子句)
    PHRASE_PATTERN = re.compile(
        r"(?<=[,;:])\s+|"  # 逗号、分号、冒号
        r"\s+(?:and|or|but|which|that|because|although|however)\s+"  # 连词
    )

    # 段落分割
    PARAGRAPH_PATTERN = re.compile(r"\n{2,}|\r\n{2,}")

    def __init__(
        self,
        min_sentence_length: int = 10,
        max_sentence_length: int = 500,
        min_phrase_length: int = 5,
    ):
        """
        初始化

        Args:
            min_sentence_length: 最小句子长度 (字符)
            max_sentence_length: 最大句子长度
            min_phrase_length: 最小短语长度
        """
        self.min_sentence_length = min_sentence_length
        self.max_sentence_length = max_sentence_length
        self.min_phrase_length = min_phrase_length

    def recommend_granularity(
        self,
        original_tokens: int,
        budget: int,
        query_type: str = "unknown",
    ) -> Granularity:
        """
        推荐压缩粒度

        Args:
            original_tokens: 原始 token 数
            budget: 目标 token 数
            query_type: Query 类型

        Returns:
            推荐的粒度
        """
        if original_tokens <= 0:
            return Granularity.PARAGRAPH

        compression_rate = budget / original_tokens

        # Query 类型调整
        type_adjustment = {
            "factual": -0.1,  # 事实型可以更激进
            "reasoning": 0.1,  # 推理型需要更多上下文
            "multi_hop": 0.15,
            "comparison": 0.05,
            "aggregation": 0.1,
        }
        adjustment = type_adjustment.get(query_type, 0.0)
        adjusted_rate = compression_rate + adjustment

        # 选择粒度
        if adjusted_rate < 0.2:
            return Granularity.PHRASE
        elif adjusted_rate < 0.4:
            return Granularity.SENTENCE
        else:
            return Granularity.PARAGRAPH

    def segment(
        self,
        text: str,
        granularity: Granularity,
    ) -> list[TextUnit]:
        """
        按指定粒度分割文本

        Args:
            text: 输入文本
            granularity: 目标粒度

        Returns:
            TextUnit 列表
        """
        if granularity == Granularity.PARAGRAPH:
            return self._segment_paragraphs(text)
        elif granularity == Granularity.SENTENCE:
            return self._segment_sentences(text)
        elif granularity == Granularity.PHRASE:
            return self._segment_phrases(text)
        else:
            # Token 级不在这里处理
            return [
                TextUnit(
                    text=text,
                    granularity=Granularity.PARAGRAPH,
                    start_char=0,
                    end_char=len(text),
                )
            ]

    def _segment_paragraphs(self, text: str) -> list[TextUnit]:
        """段落分割"""
        paragraphs = self.PARAGRAPH_PATTERN.split(text)
        units = []
        pos = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            start = text.find(para, pos)
            end = start + len(para)
            units.append(
                TextUnit(
                    text=para,
                    granularity=Granularity.PARAGRAPH,
                    start_char=start,
                    end_char=end,
                )
            )
            pos = end

        return units

    def _segment_sentences(self, text: str) -> list[TextUnit]:
        """句子分割"""
        # 先按段落分割，再按句子分割
        sentences = self.SENTENCE_PATTERN.split(text)
        units = []
        pos = 0

        for sent in sentences:
            sent = sent.strip()
            if len(sent) < self.min_sentence_length:
                continue

            # 过长的句子进一步分割
            if len(sent) > self.max_sentence_length:
                sub_units = self._segment_phrases(sent)
                for sub in sub_units:
                    sub.start_char += pos
                    sub.end_char += pos
                units.extend(sub_units)
            else:
                start = text.find(sent, pos)
                if start == -1:
                    start = pos
                end = start + len(sent)
                units.append(
                    TextUnit(
                        text=sent,
                        granularity=Granularity.SENTENCE,
                        start_char=start,
                        end_char=end,
                    )
                )
                pos = end

        return units

    def _segment_phrases(self, text: str) -> list[TextUnit]:
        """短语分割"""
        phrases = self.PHRASE_PATTERN.split(text)
        units = []
        pos = 0

        for phrase in phrases:
            phrase = phrase.strip()
            if len(phrase) < self.min_phrase_length:
                continue

            start = text.find(phrase, pos)
            if start == -1:
                start = pos
            end = start + len(phrase)
            units.append(
                TextUnit(
                    text=phrase,
                    granularity=Granularity.PHRASE,
                    start_char=start,
                    end_char=end,
                )
            )
            pos = end

        return units

    def cascade_segment(
        self,
        text: str,
        target_budget: int,
        tokenizer,
    ) -> list[TextUnit]:
        """
        级联分割: 先粗后细

        1. 段落级分割
        2. 对超出预算的部分进行句子级分割
        3. 如果仍超出，进行短语级分割

        Args:
            text: 输入文本
            target_budget: 目标 token 数
            tokenizer: Tokenizer 实例

        Returns:
            TextUnit 列表 (混合粒度)
        """
        # Step 1: 段落级分割
        paragraphs = self._segment_paragraphs(text)

        # 计算 token 数
        total_tokens = 0
        for unit in paragraphs:
            unit.metadata = {"tokens": len(tokenizer.encode(unit.text))}
            total_tokens += unit.metadata["tokens"]

        # 如果已经满足预算，返回段落级
        if total_tokens <= target_budget:
            return paragraphs

        # Step 2: 句子级分割
        sentences = []
        for para in paragraphs:
            para_sentences = self._segment_sentences(para.text)
            for sent in para_sentences:
                sent.start_char += para.start_char
                sent.end_char += para.start_char
                sent.metadata = {"tokens": len(tokenizer.encode(sent.text))}
            sentences.extend(para_sentences)

        total_tokens = sum(u.metadata["tokens"] for u in sentences)
        if total_tokens <= target_budget * 1.5:  # 允许一定余量
            return sentences

        # Step 3: 短语级分割
        phrases = []
        for sent in sentences:
            sent_phrases = self._segment_phrases(sent.text)
            for phrase in sent_phrases:
                phrase.start_char += sent.start_char
                phrase.end_char += sent.start_char
                phrase.metadata = {"tokens": len(tokenizer.encode(phrase.text))}
            phrases.extend(sent_phrases)

        return phrases
