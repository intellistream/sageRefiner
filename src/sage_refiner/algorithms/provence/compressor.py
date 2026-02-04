"""
Provence Compressor
===================

Provence是一种基于句子级别的上下文剪枝算法，用于RAG场景的上下文压缩。

核心思路：
1. 使用预训练的句子级评分模型评估每个句子与查询的相关性
2. 根据阈值过滤低相关性句子
3. 可选的重排序功能，将最相关的内容排在前面

特点：
- 基于 DeBERTa-v3 的预训练模型
- 支持批量处理
- 可配置的阈值和重排序选项
- 标题自动保留

References:
    https://arxiv.org/abs/2501.16214
"""

import logging
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class ProvenceCompressor:
    """Provence上下文压缩器

    使用预训练的provence模型对检索到的文档进行句子级别的剪枝。

    Attributes:
        model_name: Provence模型名称/路径
        threshold: 相关性阈值，低于此值的句子将被过滤
        batch_size: 批处理大小
        always_select_title: 是否始终保留标题
        reorder: 是否根据重排序分数重新排列文档
        top_k: 重排序时保留的top-k文档数
    """

    def __init__(
        self,
        model_name: str = "naver/provence-reranker-debertav3-v1",
        threshold: float = 0.1,
        batch_size: int = 32,
        always_select_title: bool = True,
        enable_warnings: bool = True,
        reorder: bool = False,
        top_k: int = 5,
        device: str | None = None,
    ):
        """初始化Provence压缩器

        Args:
            model_name: Provence模型名称，默认使用官方预训练模型
            threshold: 相关性阈值 (0-1)，低于此值的句子将被过滤
            batch_size: 处理批次大小
            always_select_title: 是否始终保留文档标题
            enable_warnings: 是否启用警告信息
            reorder: 是否对压缩后的文档进行重排序
            top_k: 重排序时保留的文档数量
            device: 计算设备 (cuda/cpu)，None则自动检测
        """
        self.model_name = model_name
        self.threshold = threshold
        self.batch_size = batch_size
        self.always_select_title = always_select_title
        self.enable_warnings = enable_warnings
        self.reorder = reorder
        self.top_k = top_k

        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 加载模型
        logger.info(f"Loading Provence model: {model_name}")
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        logger.info(
            f"Provence Compressor initialized: threshold={threshold}, "
            f"reorder={reorder}, top_k={top_k}, device={self.device}"
        )

    def compress(
        self,
        context: str,
        question: str,
    ) -> dict[str, Any]:
        """压缩单个上下文

        Args:
            context: 原始上下文文本（多个文档拼接）
            question: 问题文本

        Returns:
            {
                "compressed_context": str,
                "original_tokens": int,
                "compressed_tokens": int,
                "compression_rate": float,
                "pruned_docs": List[str],
            }
        """
        # 将context分割成文档列表
        # 假设文档之间用双换行分隔
        docs = [doc.strip() for doc in context.split("\n\n") if doc.strip()]

        if not docs:
            original_tokens = len(self.tokenizer.encode(context)) if context else 0
            return {
                "compressed_context": context,
                "original_tokens": original_tokens,
                "compressed_tokens": original_tokens,
                "compression_rate": 1.0,
                "pruned_docs": [],
            }

        # 调用批量压缩
        # 注意：context_list 的外层长度必须与 question_list 相同
        # 每个元素是该 question 对应的多个文档列表
        result = self.batch_compress(
            question_list=[question],
            context_list=[docs],  # 外层长度为 1，与 question_list 匹配
        )

        return result[0]

    def batch_compress(
        self,
        question_list: list[str],
        context_list: list[list[str]],
    ) -> list[dict[str, Any]]:
        """批量压缩上下文

        Args:
            question_list: 问题列表
            context_list: 上下文列表，每个元素是该问题对应的文档列表

        Returns:
            压缩结果列表
        """
        # 使用provence模型的process接口
        try:
            provence_out = self.model.process(
                question_list,
                context_list,
                threshold=self.threshold,
                batch_size=self.batch_size,
                always_select_title=self.always_select_title,
                enable_warnings=self.enable_warnings,
                reorder=False,  # 先不重排序，后续手动处理
            )
        except Exception as e:
            logger.error(f"Provence model processing failed: {e}")
            # 回退：返回原始上下文
            results = []
            for _question, contexts in zip(question_list, context_list):
                original_text = "\n\n".join(contexts)
                original_tokens = len(self.tokenizer.encode(original_text))
                results.append(
                    {
                        "compressed_context": original_text,
                        "original_tokens": original_tokens,
                        "compressed_tokens": original_tokens,
                        "compression_rate": 1.0,
                        "pruned_docs": contexts,
                    }
                )
            return results

        processed_contexts = provence_out["pruned_context"]
        reranking_scores = provence_out.get("reranking_score", [])

        results = []
        for i, (original, processed) in enumerate(zip(context_list, processed_contexts)):
            # 如果启用重排序，手动对文档进行重排序并截取 top_k
            if self.reorder and i < len(reranking_scores):
                scores = reranking_scores[i]
                idxs = np.argsort(scores)[::-1][: self.top_k]
                processed = [processed[j] for j in idxs if j < len(processed)]

            # 构建压缩后的上下文
            compressed_text = "\n\n".join(processed)
            original_text = "\n\n".join(original)

            # 计算token数
            original_tokens = len(self.tokenizer.encode(original_text))
            compressed_tokens = len(self.tokenizer.encode(compressed_text))

            # compression_rate 为 0~1 的比例，可用 :.1% 打印百分比
            compression_rate = compressed_tokens / original_tokens if original_tokens > 0 else 1.0

            results.append(
                {
                    "compressed_context": compressed_text,
                    "original_tokens": original_tokens,
                    "compressed_tokens": compressed_tokens,
                    "compression_rate": compression_rate,
                    "pruned_docs": processed,
                }
            )

        return results
