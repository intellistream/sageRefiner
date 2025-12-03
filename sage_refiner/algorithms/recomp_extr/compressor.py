"""
RECOMP Extractive Compressor
=============================

基于 RECOMP 论文的抽取式上下文压缩算法实现。

核心思路：
1. 使用双编码器（Contriever/DPR）对检索文档进行句子级打分
2. 计算每个句子与 query 的相似度分数（dot product，与原始 RECOMP 一致）
3. 选择 top-k 高分句子作为压缩后的上下文
4. 保持原文句子顺序进行拼接

支持的模型：
- fangyuan/nq_extractive_compressor (NQ 数据集微调)
- fangyuan/tqa_extractive_compressor (TriviaQA 微调)
- fangyuan/hotpotqa_extractive_compressor (HotpotQA 微调)
- facebook/contriever-msmarco (通用检索模型)

References:
    RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation
    https://arxiv.org/pdf/2310.04408.pdf
"""

from __future__ import annotations

import logging
import re
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling with attention mask.

    Args:
        token_embeddings: Token embeddings from model output [batch, seq_len, hidden_dim]
        attention_mask: Attention mask [batch, seq_len]

    Returns:
        Sentence embeddings [batch, hidden_dim]
    """
    # Mask out padding tokens
    token_embeddings = token_embeddings.masked_fill(~attention_mask[..., None].bool(), 0.0)
    # Sum and divide by number of non-padding tokens
    sentence_embeddings = token_embeddings.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    return sentence_embeddings


class RECOMPExtractiveCompressor:
    """RECOMP Extractive Compressor

    使用预训练的双编码器（Contriever/DPR）对检索文档进行句子级打分，
    选择与 query 最相关的 top-k 句子作为压缩后的上下文。

    Attributes:
        model: 双编码器模型
        tokenizer: 分词器
        device: 运行设备
        top_k: 选择的句子数
        score_threshold: 分数阈值
        max_length: 最大序列长度
        use_l2_norm: 是否使用 L2 归一化（默认 False，与原始 RECOMP 一致）
    """

    def __init__(
        self,
        model_path: str = "fangyuan/nq_extractive_compressor",
        device: str | None = None,
        top_k: int = 5,
        score_threshold: float = 0.0,
        max_length: int = 512,
        cache_dir: str | None = None,
        use_l2_norm: bool = False,
    ):
        """初始化 RECOMP Extractive Compressor.

        Args:
            model_path: 模型路径或 HuggingFace Hub 模型名
            device: 运行设备 ("cuda", "cpu", 或 None 自动检测)
            top_k: 选择的句子数量
            score_threshold: 分数阈值，低于此分数的句子不会被选中
            max_length: tokenizer 最大序列长度
            cache_dir: 模型缓存目录
            use_l2_norm: 是否对 embeddings 进行 L2 归一化（默认 False）
        """
        self.model_path = model_path
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.max_length = max_length
        self.use_l2_norm = use_l2_norm

        # Device setup
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading RECOMP Extractive model: {model_path}")
        logger.info(
            f"Device: {self.device}, top_k: {top_k}, "
            f"score_threshold: {score_threshold}, use_l2_norm: {use_l2_norm}"
        )

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_path, cache_dir=cache_dir)
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"RECOMP Extractive Compressor initialized: {model_path}")

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences.

        使用简单的规则进行句子分割，支持中英文。
        尝试使用 NLTK，如果不可用则使用正则表达式。

        Args:
            text: 输入文本

        Returns:
            句子列表
        """
        # Try NLTK first
        try:
            import nltk

            try:
                sentences = nltk.sent_tokenize(text)
                # Filter out empty sentences
                sentences = [s.strip() for s in sentences if s.strip()]
                if sentences:
                    return sentences
            except LookupError:
                # NLTK data not downloaded, fall through to regex
                logger.debug("NLTK punkt data not available, using regex fallback")
        except ImportError:
            logger.debug("NLTK not available, using regex fallback")

        # Regex fallback for English and Chinese
        # Split on sentence-ending punctuation followed by space or newline
        # Handles: . ! ? 。 ! ？ etc.
        pattern = r"(?<=[.!?。！？])\s+"
        sentences = re.split(pattern, text)

        # Additional split for Chinese without spaces
        result = []
        for sent in sentences:
            # Split Chinese sentences that don't have spaces
            chinese_pattern = r"(?<=[。！？])"
            sub_sents = re.split(chinese_pattern, sent)
            for sub in sub_sents:
                sub = sub.strip()
                if sub:
                    result.append(sub)

        return result if result else [text]

    def _compute_scores(self, sentences: list[str], query: str) -> torch.Tensor:
        """Compute similarity scores between sentences and query.

        使用 mean pooling + dot product 计算句子与 query 的相似度分数。
        与原始 RECOMP 实现一致，默认不进行 L2 归一化。

        Args:
            sentences: 句子列表
            query: 查询文本

        Returns:
            每个句子的相似度分数 Tensor [num_sentences]
        """
        if not sentences:
            return torch.tensor([])

        # Batch encode: query + all sentences
        texts = [query] + sentences

        with torch.no_grad():
            # Tokenize
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            # Forward pass
            outputs = self.model(**inputs)

            # Mean pooling to get sentence embeddings
            embeddings = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])

            # Optional L2 normalization (disabled by default to match original RECOMP)
            if self.use_l2_norm:
                embeddings = F.normalize(embeddings, p=2, dim=-1)

            # Query embedding is the first one
            query_emb = embeddings[0]  # [hidden_dim]
            sentence_embs = embeddings[1:]  # [num_sentences, hidden_dim]

            # Compute dot product scores (matches original RECOMP behavior)
            scores = torch.mv(sentence_embs, query_emb)  # [num_sentences]

            return scores

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: 输入文本

        Returns:
            Token 数量
        """
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def score_sentences(
        self,
        context: str,
        question: str,
    ) -> tuple[list[str], torch.Tensor]:
        """Split context into sentences and compute relevance scores.

        将 context 分割为句子，并计算每个句子与 question 的相关性分数。
        此方法分离了 "scoring" 逻辑，便于复用和调试。

        Args:
            context: 原始上下文文本
            question: 问题文本

        Returns:
            (sentences, scores) 元组:
            - sentences: 句子列表
            - scores: 每个句子的相关性分数 Tensor [num_sentences]
        """
        # Handle empty context
        if not context or not context.strip():
            logger.warning("Empty context provided to score_sentences")
            return [], torch.tensor([])

        # Split context into sentences
        sentences = self._split_sentences(context)

        if not sentences:
            logger.warning("No sentences extracted from context")
            return [], torch.tensor([])

        # Compute scores
        scores = self._compute_scores(sentences, question)

        logger.debug(
            f"Scored {len(sentences)} sentences: "
            f"min={scores.min().item():.4f}, max={scores.max().item():.4f}, "
            f"mean={scores.mean().item():.4f}"
        )

        return sentences, scores

    def select_sentences(
        self,
        sentences: list[str],
        scores: torch.Tensor,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> tuple[list[str], list[int]]:
        """Select top-k sentences based on scores.

        根据分数选择 top-k 句子，此方法分离了 "selection" 逻辑。

        Args:
            sentences: 句子列表
            scores: 每个句子的分数 Tensor
            top_k: 选择的句子数（默认使用 self.top_k）
            score_threshold: 分数阈值（默认使用 self.score_threshold）

        Returns:
            (selected_sentences, selected_indices) 元组:
            - selected_sentences: 选中的句子列表（按原文顺序）
            - selected_indices: 选中句子的原始索引列表
        """
        if top_k is None:
            top_k = self.top_k
        if score_threshold is None:
            score_threshold = self.score_threshold

        if not sentences or len(scores) == 0:
            return [], []

        # Convert to CPU for selection
        scores_cpu = scores.cpu()

        # Create list of (index, score) pairs
        indexed_scores = [(idx, scores_cpu[idx].item()) for idx in range(len(sentences))]

        # Filter by threshold
        filtered_scores = [
            (idx, score) for idx, score in indexed_scores if score >= score_threshold
        ]

        if not filtered_scores:
            logger.warning(
                f"No sentences above threshold {score_threshold}, "
                f"selecting top {top_k} anyway"
            )
            filtered_scores = indexed_scores

        # Sort by score (descending) and select top-k
        sorted_scores = sorted(filtered_scores, key=lambda x: x[1], reverse=True)
        selected = sorted_scores[:top_k]

        # Sort selected by original index to maintain order
        selected_sorted = sorted(selected, key=lambda x: x[0])
        selected_indices = [idx for idx, _ in selected_sorted]
        selected_sentences = [sentences[idx] for idx in selected_indices]

        logger.debug(
            f"Selected {len(selected_indices)}/{len(sentences)} sentences, "
            f"indices: {selected_indices}"
        )

        return selected_sentences, selected_indices

    def compress(self, context: str, question: str) -> dict[str, Any]:
        """Compress context by selecting top-k relevant sentences.

        压缩上下文，选择与问题最相关的 top-k 句子。

        步骤:
        1. 调用 score_sentences 分割句子并计算分数
        2. 调用 select_sentences 选择 top-k 句子
        3. 按原文顺序拼接选中的句子

        Args:
            context: 原始上下文文本（检索到的文档拼接）
            question: 问题文本

        Returns:
            {
                "compressed_context": str,
                "original_tokens": int,
                "compressed_tokens": int,
                "compression_rate": float,
                "num_selected_sentences": int,
                "total_sentences": int,
                "sentence_scores": List[float],
                "selected_indices": List[int],
            }
        """
        logger.debug(
            f"RECOMPExtractiveCompressor.compress called with "
            f"context length: {len(context)}, question: '{question[:50]}...'"
        )

        # Handle empty context
        if not context or not context.strip():
            logger.warning("Empty context provided")
            return {
                "compressed_context": "",
                "original_tokens": 0,
                "compressed_tokens": 0,
                "compression_rate": 1.0,
                "num_selected_sentences": 0,
                "total_sentences": 0,
                "sentence_scores": [],
                "selected_indices": [],
            }

        # Step 1: Score sentences
        sentences, scores = self.score_sentences(context, question)
        total_sentences = len(sentences)

        if not sentences:
            logger.warning("No sentences extracted from context")
            original_tokens = self._count_tokens(context)
            return {
                "compressed_context": context,
                "original_tokens": original_tokens,
                "compressed_tokens": original_tokens,
                "compression_rate": 1.0,
                "num_selected_sentences": 0,
                "total_sentences": 0,
                "sentence_scores": [],
                "selected_indices": [],
            }

        # Step 2: Select top-k sentences
        selected_sentences, selected_indices = self.select_sentences(sentences, scores)

        # Step 3: Concatenate selected sentences in original order
        compressed_context = " ".join(selected_sentences)

        # Step 4: Compute statistics
        original_tokens = self._count_tokens(context)
        compressed_tokens = self._count_tokens(compressed_context)
        compression_rate = compressed_tokens / original_tokens if original_tokens > 0 else 1.0

        logger.info(
            f"RECOMP Extractive: {original_tokens} -> {compressed_tokens} tokens "
            f"({compression_rate:.2%}), {len(selected_indices)}/{total_sentences} sentences"
        )

        return {
            "compressed_context": compressed_context,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_rate": compression_rate,
            "num_selected_sentences": len(selected_indices),
            "total_sentences": total_sentences,
            "sentence_scores": scores.cpu().tolist(),
            "selected_indices": selected_indices,
        }

    def compress_batch(
        self, contexts: list[str], questions: list[str]
    ) -> list[dict[str, Any]]:
        """Batch compress multiple context-question pairs.

        Args:
            contexts: 上下文列表
            questions: 问题列表

        Returns:
            压缩结果列表
        """
        if len(contexts) != len(questions):
            raise ValueError(
                f"Length mismatch: contexts={len(contexts)}, questions={len(questions)}"
            )

        results = []
        for context, question in zip(contexts, questions):
            result = self.compress(context, question)
            results.append(result)

        return results
