"""
Information Density Analyzer (信息密度分析器)
=============================================

检测和去除冗余信息，保留高密度信息。

核心思想:
1. **语义去重** - 相似度高的片段只保留最相关的
2. **PMI 分析** - 使用 Pointwise Mutual Information 检测冗余
3. **覆盖度优化** - 确保信息覆盖最大化

算法:
- 构建片段-Query 相关性矩阵
- 构建片段-片段 相似度矩阵
- 贪婪选择: 每次选择边际增益最大的片段
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class DensityScore:
    """信息密度得分"""

    relevance: float  # 与 Query 的相关性 (0-1)
    redundancy: float  # 与已选片段的冗余度 (0-1)
    novelty: float  # 新颖性 = 1 - redundancy
    marginal_gain: float  # 边际增益 = relevance * novelty


class InformationDensityAnalyzer:
    """
    信息密度分析器

    结合相关性和冗余度进行智能片段选择。

    算法流程:
    1. 计算所有片段与 Query 的相关性
    2. 计算片段间的相似度矩阵
    3. 贪婪选择: 每步选择边际增益最大的片段
    4. 更新冗余度并继续

    Usage:
        analyzer = InformationDensityAnalyzer(encoder)
        selected = analyzer.select_diverse(
            query="What is quantum computing?",
            units=text_units,
            budget=1000,
            tokenizer=tokenizer,
        )
    """

    def __init__(
        self,
        encoder=None,
        similarity_threshold: float = 0.8,
        diversity_weight: float = 0.3,
    ):
        """
        初始化

        Args:
            encoder: 编码器 (sentence-transformers 模型)
            similarity_threshold: 相似度阈值，超过则认为冗余
            diversity_weight: 多样性权重 (0-1)
        """
        self.encoder = encoder
        self.similarity_threshold = similarity_threshold
        self.diversity_weight = diversity_weight

        if encoder is None:
            self._load_default_encoder()

    def _load_default_encoder(self) -> None:
        """加载默认编码器"""
        try:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
            logger.info("Loaded default encoder: BAAI/bge-small-en-v1.5")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            self.encoder = None

    def compute_relevance(
        self,
        query: str,
        texts: list[str],
    ) -> np.ndarray:
        """
        计算文本与 Query 的相关性

        Args:
            query: 查询
            texts: 文本列表

        Returns:
            相关性分数数组 [num_texts]
        """
        if self.encoder is None:
            # 回退: 简单的词重叠
            return self._compute_relevance_overlap(query, texts)

        # 编码
        query_emb = self.encoder.encode([query], convert_to_tensor=True)
        text_embs = self.encoder.encode(texts, convert_to_tensor=True)

        # 余弦相似度
        similarities = F.cosine_similarity(
            query_emb.expand(len(texts), -1), text_embs, dim=1
        )

        # 归一化到 [0, 1]
        scores = (similarities + 1) / 2

        return scores.cpu().numpy()

    def _compute_relevance_overlap(
        self,
        query: str,
        texts: list[str],
    ) -> np.ndarray:
        """基于词重叠的简单相关性"""
        query_words = set(query.lower().split())
        scores = []

        for text in texts:
            text_words = set(text.lower().split())
            if not query_words:
                scores.append(0.0)
            else:
                overlap = len(query_words & text_words) / len(query_words)
                scores.append(overlap)

        return np.array(scores)

    def compute_similarity_matrix(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """
        计算文本间的相似度矩阵

        Args:
            texts: 文本列表

        Returns:
            相似度矩阵 [num_texts, num_texts]
        """
        if self.encoder is None:
            # 回退: 简单的 Jaccard 相似度
            return self._compute_jaccard_matrix(texts)

        # 编码
        embeddings = self.encoder.encode(texts, convert_to_tensor=True)

        # 两两余弦相似度
        sim_matrix = F.cosine_similarity(
            embeddings.unsqueeze(0), embeddings.unsqueeze(1), dim=2
        )

        return sim_matrix.cpu().numpy()

    def _compute_jaccard_matrix(self, texts: list[str]) -> np.ndarray:
        """Jaccard 相似度矩阵"""
        n = len(texts)
        word_sets = [set(t.lower().split()) for t in texts]
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                if not word_sets[i] or not word_sets[j]:
                    sim = 0.0
                else:
                    intersection = len(word_sets[i] & word_sets[j])
                    union = len(word_sets[i] | word_sets[j])
                    sim = intersection / union if union > 0 else 0.0
                matrix[i, j] = sim
                matrix[j, i] = sim

        return matrix

    def select_diverse(
        self,
        query: str,
        units: list,  # List[TextUnit]
        budget: int,
        tokenizer,
    ) -> list:
        """
        多样性选择: 在满足预算的情况下最大化覆盖度

        使用 Maximal Marginal Relevance (MMR) 算法变体:
        score = λ * relevance + (1-λ) * novelty

        Args:
            query: 查询
            units: TextUnit 列表
            budget: Token 预算
            tokenizer: Tokenizer

        Returns:
            选中的 TextUnit 列表
        """
        if not units:
            return []

        # 提取文本
        texts = [u.text for u in units]
        n = len(texts)

        # 计算相关性
        relevance_scores = self.compute_relevance(query, texts)

        # 计算相似度矩阵
        sim_matrix = self.compute_similarity_matrix(texts)

        # 计算 token 数
        token_counts = [len(tokenizer.encode(t)) for t in texts]

        # 贪婪选择
        selected_indices = []
        selected_tokens = 0
        remaining = set(range(n))

        while remaining and selected_tokens < budget:
            best_idx = -1
            best_gain = -float("inf")

            for idx in remaining:
                # 检查预算
                if selected_tokens + token_counts[idx] > budget * 1.1:  # 10% 余量
                    continue

                # 计算冗余度 (与已选片段的最大相似度)
                if selected_indices:
                    redundancy = max(sim_matrix[idx, s] for s in selected_indices)
                else:
                    redundancy = 0.0

                # 边际增益
                novelty = 1.0 - redundancy
                gain = (
                    (1 - self.diversity_weight) * relevance_scores[idx]
                    + self.diversity_weight * novelty
                )

                # 如果相似度太高，降低增益
                if redundancy > self.similarity_threshold:
                    gain *= 0.1  # 大幅惩罚

                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx

            if best_idx == -1:
                break

            # 选择
            selected_indices.append(best_idx)
            selected_tokens += token_counts[best_idx]
            remaining.remove(best_idx)

            # 更新 unit 的分数
            units[best_idx].score = best_gain

        # 按原始顺序返回
        selected_indices.sort()
        return [units[i] for i in selected_indices]

    def analyze_density(
        self,
        query: str,
        texts: list[str],
    ) -> list[DensityScore]:
        """
        分析每个文本的信息密度

        Args:
            query: 查询
            texts: 文本列表

        Returns:
            DensityScore 列表
        """
        n = len(texts)

        # 相关性
        relevance_scores = self.compute_relevance(query, texts)

        # 相似度矩阵
        sim_matrix = self.compute_similarity_matrix(texts)

        scores = []
        for i in range(n):
            # 冗余度: 与其他文本的平均相似度
            other_sims = [sim_matrix[i, j] for j in range(n) if j != i]
            redundancy = np.mean(other_sims) if other_sims else 0.0

            novelty = 1.0 - redundancy
            marginal_gain = relevance_scores[i] * novelty

            scores.append(
                DensityScore(
                    relevance=float(relevance_scores[i]),
                    redundancy=float(redundancy),
                    novelty=float(novelty),
                    marginal_gain=float(marginal_gain),
                )
            )

        return scores
