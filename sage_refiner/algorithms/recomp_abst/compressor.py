"""
RECOMP Abstractive Compressor
==============================

基于 RECOMP 论文的摘要生成式上下文压缩算法实现。

核心思路：
1. 使用微调的 T5 模型生成检索文档的摘要
2. 输入格式: "Question: {question}\n Document: {passages}\n Summary: "
3. 模型生成简洁的摘要作为压缩后的上下文

支持的模型：
- fangyuan/nq_abstractive_compressor (NQ 数据集微调)
- fangyuan/tqa_abstractive_compressor (TriviaQA 微调)
- fangyuan/hotpotqa_abstractive (HotpotQA 微调)
- t5-large / t5-base (通用摘要模型，需要适配 prompt)

References:
    RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation
    https://arxiv.org/pdf/2310.04408.pdf
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)


class RECOMPAbstractiveCompressor:
    """RECOMP Abstractive Compressor

    使用微调的 T5 模型生成检索文档的摘要，将检索文档压缩为简洁的摘要。

    Attributes:
        model: T5 序列到序列模型
        tokenizer: 分词器
        device: 运行设备
        max_source_length: 输入最大长度
        max_target_length: 生成摘要最大长度
        num_beams: beam search 的 beam 数量
    """

    def __init__(
        self,
        model_path: str = "fangyuan/nq_abstractive_compressor",
        device: str | None = None,
        max_source_length: int = 1024,
        max_target_length: int = 512,
        num_beams: int = 4,
        cache_dir: str | None = None,
        torch_dtype: str | None = None,
    ):
        """初始化 RECOMP Abstractive Compressor.

        Args:
            model_path: 模型路径或 HuggingFace Hub 模型名
            device: 运行设备 ("cuda", "cpu", 或 None 自动检测)
            max_source_length: 输入的最大 token 长度
            max_target_length: 生成摘要的最大 token 长度
            num_beams: beam search 的 beam 数量
            cache_dir: 模型缓存目录
            torch_dtype: 模型精度 ("float16", "bfloat16", "float32", 或 None 自动选择)
        """
        self.model_path = model_path
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.num_beams = num_beams

        # Device setup
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Dtype setup
        if torch_dtype is not None:
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            self.torch_dtype = dtype_map.get(torch_dtype, None)
        else:
            # Auto select based on device
            if self.device == "cuda" and torch.cuda.is_available():
                # Use bfloat16 on CUDA for memory efficiency if supported
                if torch.cuda.is_bf16_supported():
                    self.torch_dtype = torch.bfloat16
                else:
                    self.torch_dtype = torch.float16
            else:
                self.torch_dtype = torch.float32

        logger.info(f"Loading RECOMP Abstractive model: {model_path}")
        logger.info(
            f"Device: {self.device}, dtype: {self.torch_dtype}, "
            f"max_source_length: {max_source_length}, max_target_length: {max_target_length}, "
            f"num_beams: {num_beams}"
        )

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            torch_dtype=self.torch_dtype,
        )
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"RECOMP Abstractive Compressor initialized: {model_path}")

    def _format_input(self, context: str, question: str) -> str:
        """Format input for the T5 model.

        使用 RECOMP 论文中定义的输入格式:
        "Question: {question}\n Document: {passages}\n Summary: "

        Args:
            context: 检索到的文档（已拼接）
            question: 问题文本

        Returns:
            格式化后的输入字符串
        """
        return f"Question: {question}\n Document: {context}\n Summary: "

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: 输入文本

        Returns:
            Token 数量
        """
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def generate_summary(self, context: str, question: str) -> str:
        """Generate abstractive summary for the given context and question.

        生成摘要的核心方法，使用 T5 模型进行 seq2seq 生成。

        Args:
            context: 检索到的文档（已拼接）
            question: 问题文本

        Returns:
            生成的摘要文本
        """
        # Format input
        input_text = self._format_input(context, question)

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_source_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=self.max_target_length,
                num_beams=self.num_beams,
                early_stopping=True,
                do_sample=False,  # Deterministic generation
            )

        # Decode
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return summary.strip()

    def compress(self, context: str, question: str) -> dict[str, Any]:
        """Compress context by generating an abstractive summary.

        压缩上下文，生成摘要式的压缩文本。

        步骤:
        1. 构造 T5 输入格式
        2. 使用模型生成摘要
        3. 返回压缩结果

        Args:
            context: 原始上下文文本（检索到的文档拼接）
            question: 问题文本

        Returns:
            {
                "compressed_context": str,
                "original_tokens": int,
                "compressed_tokens": int,
                "compression_rate": float,
            }
        """
        logger.debug(
            f"RECOMPAbstractiveCompressor.compress called with "
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
            }

        # Count original tokens
        original_tokens = self._count_tokens(context)

        # Generate summary
        try:
            summary = self.generate_summary(context, question)
        except Exception as e:
            logger.error(f"Summary generation failed: {e}", exc_info=True)
            # Fallback: return original context truncated to max_target_length tokens
            summary = context
            logger.warning("Fallback to original context due to generation error")

        # Handle empty summary (model may generate empty output)
        if not summary or not summary.strip():
            logger.warning("Model generated empty summary, using placeholder")
            summary = "This passage doesn't contain relevant information to the question."

        # Count compressed tokens
        compressed_tokens = self._count_tokens(summary)
        compression_rate = compressed_tokens / original_tokens if original_tokens > 0 else 1.0

        logger.info(
            f"RECOMP Abstractive: {original_tokens} -> {compressed_tokens} tokens "
            f"({compression_rate:.2%})"
        )

        return {
            "compressed_context": summary,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_rate": compression_rate,
        }

    def compress_batch(
        self, contexts: list[str], questions: list[str], batch_size: int = 4
    ) -> list[dict[str, Any]]:
        """Batch compress multiple context-question pairs.

        批量压缩多个 context-question 对，支持 batch 推理以提高效率。

        Args:
            contexts: 上下文列表
            questions: 问题列表
            batch_size: 批处理大小

        Returns:
            压缩结果列表
        """
        if len(contexts) != len(questions):
            raise ValueError(
                f"Length mismatch: contexts={len(contexts)}, questions={len(questions)}"
            )

        results = []
        num_samples = len(contexts)

        for i in range(0, num_samples, batch_size):
            batch_contexts = contexts[i : i + batch_size]
            batch_questions = questions[i : i + batch_size]

            # Format inputs
            batch_inputs = [
                self._format_input(ctx, q) for ctx, q in zip(batch_contexts, batch_questions)
            ]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_inputs,
                max_length=self.max_source_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            # Generate batch
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=self.max_target_length,
                    num_beams=self.num_beams,
                    early_stopping=True,
                    do_sample=False,
                )

            # Decode and compute stats
            for j, (ctx, output_ids) in enumerate(zip(batch_contexts, outputs)):
                summary = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

                # Handle empty summary
                if not summary:
                    summary = "This passage doesn't contain relevant information to the question."

                original_tokens = self._count_tokens(ctx)
                compressed_tokens = self._count_tokens(summary)
                compression_rate = (
                    compressed_tokens / original_tokens if original_tokens > 0 else 1.0
                )

                results.append(
                    {
                        "compressed_context": summary,
                        "original_tokens": original_tokens,
                        "compressed_tokens": compressed_tokens,
                        "compression_rate": compression_rate,
                    }
                )

        return results
