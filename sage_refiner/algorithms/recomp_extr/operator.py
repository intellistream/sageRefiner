"""
RECOMP Extractive Compression Operator for SAGE Pipeline
==========================================================

将 RECOMP Extractive 压缩算法封装为 SAGE MapOperator，用于 RAG pipeline。
基于双编码器的句子级抽取压缩。
"""

import logging

from sage.common.core.functions import MapFunction as MapOperator

from .compressor import RECOMPExtractiveCompressor

logger = logging.getLogger(__name__)


class RECOMPExtractiveOperator(MapOperator):
    """RECOMP Extractive Refiner 算子

    在 RAG pipeline 中使用 RECOMP Extractive 算法压缩检索到的上下文。
    选择与 query 最相关的 top-k 句子作为压缩后的上下文。

    输入格式:
        {
            "query": str,
            "retrieval_results": List[str or dict],  # 检索到的文档
        }

    输出格式:
        {
            "query": str,
            "retrieval_results": List[str],  # 原始文档（保留）
            "refining_results": List[str],   # 压缩后的句子列表
            "compressed_context": str,        # 压缩后的上下文
            "original_tokens": int,
            "compressed_tokens": int,
            "compression_rate": float,
            "num_selected_sentences": int,
            "total_sentences": int,
        }

    Config 参数:
        enabled: bool - 是否启用压缩（False 时为 baseline 模式）
        model_path: str - 模型路径或 HuggingFace Hub 模型名
        device: str - 运行设备 ("cuda", "cpu", 或 None 自动检测)
        top_k: int - 选择的句子数量
        score_threshold: float - 分数阈值
        max_length: int - tokenizer 最大序列长度
        cache_dir: str - 模型缓存目录
        use_l2_norm: bool - 是否使用 L2 归一化（默认 False）
    """

    def __init__(self, config: dict, ctx=None):
        """初始化 RECOMP Extractive 算子.

        Args:
            config: 配置字典
            ctx: SAGE 上下文（可选）
        """
        super().__init__(config=config, ctx=ctx)
        self.cfg = config
        self.enabled = config.get("enabled", True)

        if self.enabled:
            self._init_compressor()
            logger.info("RECOMP Extractive Compression Operator initialized")
        else:
            logger.info("RECOMP Extractive Compression disabled (baseline mode)")

    def _init_compressor(self):
        """初始化 RECOMP Extractive 压缩器."""
        import torch

        # Get model path
        model_path = self.cfg.get("model_path", "fangyuan/nq_extractive_compressor")

        # Get device
        device = self.cfg.get("device")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Get other params
        top_k = self.cfg.get("top_k", 5)
        score_threshold = self.cfg.get("score_threshold", 0.0)
        max_length = self.cfg.get("max_length", 512)
        cache_dir = self.cfg.get("cache_dir")
        use_l2_norm = self.cfg.get("use_l2_norm", False)

        # Create compressor
        self.compressor = RECOMPExtractiveCompressor(
            model_path=model_path,
            device=device,
            top_k=top_k,
            score_threshold=score_threshold,
            max_length=max_length,
            cache_dir=cache_dir,
            use_l2_norm=use_l2_norm,
        )

        logger.info(f"RECOMP Extractive Compressor initialized: {model_path}")
        logger.info(
            f"Parameters: top_k={top_k}, score_threshold={score_threshold}, "
            f"use_l2_norm={use_l2_norm}"
        )

    def _extract_text_from_results(self, retrieval_results: list) -> list[str]:
        """从检索结果中提取文本.

        Args:
            retrieval_results: 检索结果列表，可以是字符串或字典

        Returns:
            文本列表
        """
        docs_text = []
        for result in retrieval_results:
            if isinstance(result, dict):
                # Try common keys
                text = (
                    result.get("text")
                    or result.get("contents")
                    or result.get("content")
                    or result.get("passage")
                    or str(result)
                )
            else:
                text = str(result)
            docs_text.append(text)
        return docs_text

    def execute(self, data: dict) -> dict:
        """执行压缩

        Args:
            data: 包含 query 和 retrieval_results 的字典

        Returns:
            添加了 refining_results 等压缩结果的字典
        """
        if not isinstance(data, dict):
            logger.error(f"Unexpected input format: {type(data)}")
            return data

        query = data.get("query", "")
        retrieval_results = data.get("retrieval_results", [])

        # Handle empty retrieval results
        if not retrieval_results:
            logger.warning(f"No retrieval results for query: '{query[:50]}...'")
            result_data = data.copy()
            result_data["refining_results"] = []
            result_data["compressed_context"] = ""
            result_data["original_tokens"] = 0
            result_data["compressed_tokens"] = 0
            result_data["compression_rate"] = 1.0
            result_data["num_selected_sentences"] = 0
            result_data["total_sentences"] = 0
            return result_data

        # Extract text from retrieval results
        docs_text = self._extract_text_from_results(retrieval_results)

        if not self.enabled:
            # Baseline mode: use original retrieval_results as refining_results
            result_data = data.copy()
            result_data["refining_results"] = docs_text
            logger.info("RECOMP Extractive disabled - passing through original documents")
            return result_data

        # Construct original context
        original_context = "\n\n".join(docs_text)

        # Log input statistics
        logger.info(
            f"RECOMP Extractive: Processing {len(docs_text)} documents, "
            f"query: '{query[:50]}...'"
        )

        try:
            # Compress
            compress_result = self.compressor.compress(
                context=original_context,
                question=query,
            )

            compressed_text = compress_result["compressed_context"]
            original_tokens = compress_result["original_tokens"]
            compressed_tokens = compress_result["compressed_tokens"]
            compression_rate = compress_result["compression_rate"]
            num_selected = compress_result["num_selected_sentences"]
            total_sentences = compress_result["total_sentences"]

            # Log compression results
            logger.info(
                f"RECOMP Extractive Compression: {original_tokens} -> {compressed_tokens} tokens "
                f"({compression_rate:.1%}), {num_selected}/{total_sentences} sentences"
            )
            logger.debug(f"Compressed text preview: {compressed_text[:200]}...")

            # Build result - use refining_results for promptor
            result_data = data.copy()
            result_data["refining_results"] = [compressed_text]  # Promptor expects list
            result_data["compressed_context"] = compressed_text
            result_data["original_tokens"] = original_tokens
            result_data["compressed_tokens"] = compressed_tokens
            result_data["compression_rate"] = compression_rate
            result_data["num_selected_sentences"] = num_selected
            result_data["total_sentences"] = total_sentences

            return result_data

        except Exception as e:
            logger.error(f"RECOMP Extractive Compression failed: {e}", exc_info=True)

            # Fallback: use original documents as refining_results
            # Compute meaningful statistics for fallback case
            # Note: We use compressor's token counting and sentence splitting for consistency
            original_tokens = self.compressor._count_tokens(original_context)
            # Since we're not compressing, compressed_tokens equals original_tokens
            compressed_tokens = original_tokens
            # Compression rate is 1.0 (no compression)
            compression_rate = 1.0
            # Count sentences using compressor's method for consistency
            # This gives accurate sentence count rather than using doc count as proxy
            sentences = self.compressor._split_sentences(original_context)
            total_sentences = len(sentences)

            result_data = data.copy()
            result_data["refining_results"] = docs_text
            result_data["compressed_context"] = original_context
            result_data["original_tokens"] = original_tokens
            result_data["compressed_tokens"] = compressed_tokens
            result_data["compression_rate"] = compression_rate
            result_data["num_selected_sentences"] = total_sentences  # All sentences kept
            result_data["total_sentences"] = total_sentences

            logger.warning(
                f"Fallback to original documents due to compression error. "
                f"Stats: {original_tokens} tokens, {total_sentences} sentences"
            )
            return result_data
