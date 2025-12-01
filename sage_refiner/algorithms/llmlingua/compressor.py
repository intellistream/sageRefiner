"""
LLMLingua Compressor
====================

封装微软 LLMLingua 库，提供与 SAGE 其他压缩器一致的接口。

支持:
- LLMLingua-2 (BERT-based, 推荐): 快速、准确
- LLMLingua-1 (LLM-based): 需要 GPU，较慢但保留语义更好

依赖:
    pip install llmlingua

模型:
- LLMLingua-2: microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank
- LLMLingua-1: 需要指定小型 LLM (如 meta-llama/Llama-2-7b)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CompressionResult:
    """压缩结果，与 AdaptiveCompressor.CompressionResult 兼容"""

    compressed_context: str
    original_tokens: int
    compressed_tokens: int
    compression_rate: float

    # LLMLingua 特有信息
    origin_tokens: int = 0  # 原始 token 数（llmlingua 返回）
    compressed_tokens_llmlingua: int = 0  # 压缩后 token 数（llmlingua 返回）
    ratio: float = 0.0  # 压缩比例
    saving: str = ""  # 节省信息

    # 通用信息
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "compressed_context": self.compressed_context,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "compression_rate": self.compression_rate,
            "origin_tokens": self.origin_tokens,
            "compressed_tokens_llmlingua": self.compressed_tokens_llmlingua,
            "ratio": self.ratio,
            "saving": self.saving,
            "processing_time_ms": self.processing_time_ms,
        }


class LLMLinguaCompressor:
    """
    LLMLingua 上下文压缩器

    封装微软 LLMLingua 库，提供统一的压缩接口。
    如果 LLMLingua 不可用，回退到简单截断。

    Usage:
        compressor = LLMLinguaCompressor()
        result = compressor.compress(
            context="Long context...",
            question="What is...?",
            budget=2048,
        )
        print(result.compressed_context)
        print(f"Compression: {result.compression_rate:.1%}")
    """

    def __init__(
        self,
        model_name: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        use_llmlingua2: bool = True,
        device: str = "cuda",
        target_token: int = -1,
        context_budget: str = "+100",
        use_sentence_level_filter: bool = False,
        use_context_level_filter: bool = True,
        use_token_level_filter: bool = True,
        force_tokens: list[str] | None = None,
        drop_consecutive: bool = False,
    ):
        """
        初始化 LLMLingua 压缩器

        Args:
            model_name: LLMLingua 模型名称
                - LLMLingua-2: "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
                - LLMLingua-1: 需要 LLM 模型如 "meta-llama/Llama-2-7b-hf"
            use_llmlingua2: 是否使用 LLMLingua-2 (BERT-based)
            device: 设备 ("cuda", "cpu", "cuda:0", etc.)
            target_token: 目标 token 数（-1 表示使用 rate）
            context_budget: 上下文预算调整（如 "+100"）
            use_sentence_level_filter: 是否使用句子级过滤
            use_context_level_filter: 是否使用上下文级过滤
            use_token_level_filter: 是否使用 token 级过滤
            force_tokens: 强制保留的 token 列表
            drop_consecutive: 是否合并连续空格
        """
        self.model_name = model_name
        self.use_llmlingua2 = use_llmlingua2
        self.device = device
        self.target_token = target_token
        self.context_budget = context_budget
        self.use_sentence_level_filter = use_sentence_level_filter
        self.use_context_level_filter = use_context_level_filter
        self.use_token_level_filter = use_token_level_filter
        self.force_tokens = force_tokens or ["\n", ".", "?", "!", ","]
        self.drop_consecutive = drop_consecutive

        # 延迟加载压缩器
        self._compressor = None
        self._llmlingua_available = None
        self._fallback_tokenizer = None

    def _load_compressor(self) -> bool:
        """
        延迟加载 LLMLingua 压缩器

        Returns:
            是否加载成功
        """
        if self._llmlingua_available is not None:
            return self._llmlingua_available

        try:
            from llmlingua import PromptCompressor

            self._compressor = PromptCompressor(
                model_name=self.model_name,
                use_llmlingua2=self.use_llmlingua2,
                device_map=self.device,
            )
            self._llmlingua_available = True
            logger.info(
                f"LLMLingua loaded: model={self.model_name}, "
                f"llmlingua2={self.use_llmlingua2}, device={self.device}"
            )
            return True

        except ImportError:
            logger.warning(
                "LLMLingua not installed. Install with: pip install llmlingua. "
                "Falling back to simple truncation."
            )
            self._llmlingua_available = False
            return False

        except Exception as e:
            logger.warning(f"Failed to load LLMLingua: {e}. Falling back to simple truncation.")
            self._llmlingua_available = False
            return False

    def _load_fallback_tokenizer(self):
        """加载后备 tokenizer（用于 token 计数和截断）"""
        if self._fallback_tokenizer is not None:
            return

        try:
            from transformers import AutoTokenizer

            self._fallback_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        except Exception:
            # 极简后备：按空格分词
            self._fallback_tokenizer = None

    def _count_tokens(self, text: str) -> int:
        """计算 token 数"""
        if self._compressor is not None:
            # 使用 LLMLingua 的 tokenizer
            try:
                return len(self._compressor.tokenizer.encode(text))
            except Exception:
                pass

        self._load_fallback_tokenizer()
        if self._fallback_tokenizer is not None:
            return len(self._fallback_tokenizer.encode(text))

        # 极简后备：按空格分词
        return len(text.split())

    def compress(
        self,
        context: str,
        question: str,
        budget: int = 2048,
        compression_ratio: float | None = None,
    ) -> CompressionResult:
        """
        压缩上下文

        Args:
            context: 原始上下文
            question: 问题/查询
            budget: Token 预算
            compression_ratio: 压缩比例 (0-1)，如果指定则忽略 budget

        Returns:
            CompressionResult
        """
        start_time = time.time()

        # 计算原始 token 数
        original_tokens = self._count_tokens(context)

        # 如果原始上下文已经小于预算，直接返回
        if original_tokens <= budget:
            return CompressionResult(
                compressed_context=context,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_rate=1.0,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        # 计算压缩率
        if compression_ratio is not None:
            rate = compression_ratio
        else:
            rate = budget / original_tokens
            rate = max(0.1, min(0.9, rate))  # 限制在合理范围

        # 尝试使用 LLMLingua
        if self._load_compressor():
            return self._compress_with_llmlingua(
                context=context,
                question=question,
                rate=rate,
                original_tokens=original_tokens,
                start_time=start_time,
            )
        return self._compress_fallback(
            context=context,
            budget=budget,
            original_tokens=original_tokens,
            start_time=start_time,
        )

    def _compress_with_llmlingua(
        self,
        context: str,
        question: str,
        rate: float,
        original_tokens: int,
        start_time: float,
    ) -> CompressionResult:
        """使用 LLMLingua 压缩"""
        try:
            # 调用 LLMLingua
            result = self._compressor.compress_prompt(
                context,
                question=question,
                rate=rate,
                target_token=self.target_token,
                context_budget=self.context_budget,
                use_sentence_level_filter=self.use_sentence_level_filter,
                use_context_level_filter=self.use_context_level_filter,
                use_token_level_filter=self.use_token_level_filter,
                force_tokens=self.force_tokens,
                drop_consecutive=self.drop_consecutive,
            )

            compressed_text = result.get("compressed_prompt", context)
            compressed_tokens = self._count_tokens(compressed_text)

            # LLMLingua 返回的统计
            origin_tokens = result.get("origin_tokens", original_tokens)
            compressed_tokens_llm = result.get("compressed_tokens", compressed_tokens)
            ratio = result.get("ratio", rate)
            saving = result.get("saving", "")

            processing_time = (time.time() - start_time) * 1000

            logger.info(
                f"LLMLingua compression: {original_tokens} -> {compressed_tokens} tokens "
                f"({compressed_tokens / original_tokens:.1%}), time={processing_time:.0f}ms"
            )

            return CompressionResult(
                compressed_context=compressed_text,
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                compression_rate=compressed_tokens / original_tokens
                if original_tokens > 0
                else 1.0,
                origin_tokens=origin_tokens,
                compressed_tokens_llmlingua=compressed_tokens_llm,
                ratio=ratio,
                saving=saving,
                processing_time_ms=processing_time,
            )

        except Exception as e:
            logger.error(f"LLMLingua compression failed: {e}", exc_info=True)
            # 回退到简单截断
            return self._compress_fallback(
                context=context,
                budget=int(original_tokens * rate),
                original_tokens=original_tokens,
                start_time=start_time,
            )

    def _compress_fallback(
        self,
        context: str,
        budget: int,
        original_tokens: int,
        start_time: float,
    ) -> CompressionResult:
        """后备压缩：简单截断"""
        logger.warning("Using fallback truncation compression")

        self._load_fallback_tokenizer()

        if self._fallback_tokenizer is not None:
            # 使用 tokenizer 截断
            tokens = self._fallback_tokenizer.encode(context)
            truncated_tokens = tokens[:budget]
            compressed_text = self._fallback_tokenizer.decode(
                truncated_tokens,
                skip_special_tokens=True,
            )
            compressed_tokens = len(truncated_tokens)
        else:
            # 极简后备：按字符截断
            # 假设每个 token 约 4 个字符
            target_chars = budget * 4
            compressed_text = context[:target_chars]
            compressed_tokens = budget

        processing_time = (time.time() - start_time) * 1000

        return CompressionResult(
            compressed_context=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_rate=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            processing_time_ms=processing_time,
        )

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

    @property
    def is_available(self) -> bool:
        """检查 LLMLingua 是否可用"""
        return self._load_compressor()
