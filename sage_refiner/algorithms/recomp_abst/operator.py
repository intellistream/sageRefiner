"""
RECOMP Abstractive Compression Operator for SAGE Pipeline
===========================================================

将 RECOMP Abstractive 压缩算法封装为 SAGE MapOperator，用于 RAG pipeline。
基于 T5 模型的摘要生成式压缩。
"""

import logging

from sage.common.core.functions import MapFunction as MapOperator

from .compressor import RECOMPAbstractiveCompressor

logger = logging.getLogger(__name__)


class RECOMPAbstractiveRefinerOperator(MapOperator):
    """RECOMP Abstractive Refiner 算子

    在 RAG pipeline 中使用 RECOMP Abstractive 算法压缩检索到的上下文。
    使用 T5 模型生成摘要作为压缩后的上下文。

    输入格式:
        {
            "query": str,
            "retrieval_results": List[str or dict],  # 检索到的文档
        }

    输出格式:
        {
            "query": str,
            "retrieval_results": List[str],  # 原始文档（保留）
            "refining_results": List[str],   # 压缩后的摘要列表
            "compressed_context": str,        # 压缩后的上下文
            "original_tokens": int,
            "compressed_tokens": int,
            "compression_rate": float,
        }

    Config 参数:
        enabled: bool - 是否启用压缩（False 时为 baseline 模式）
        model_path: str - 模型路径或 HuggingFace Hub 模型名
        device: str - 运行设备 ("cuda", "cpu", 或 None 自动检测)
        max_source_length: int - 输入最大长度
        max_target_length: int - 生成摘要最大长度
        num_beams: int - beam search 的 beam 数量
        cache_dir: str - 模型缓存目录
        torch_dtype: str - 模型精度 ("float16", "bfloat16", "float32")
    """

    def __init__(self, config: dict, ctx=None):
        """初始化 RECOMP Abstractive 算子.

        Args:
            config: 配置字典
            ctx: SAGE 上下文（可选）
        """
        super().__init__(config=config, ctx=ctx)
        self.cfg = config
        self.enabled = config.get("enabled", True)

        if self.enabled:
            self._init_compressor()
            logger.info("RECOMP Abstractive Compression Operator initialized")
        else:
            logger.info("RECOMP Abstractive Compression disabled (baseline mode)")

    def _init_compressor(self):
        """初始化 RECOMP Abstractive 压缩器."""
        import torch

        # Get model path
        model_path = self.cfg.get("model_path", "fangyuan/nq_abstractive_compressor")

        # Get device
        device = self.cfg.get("device")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Get other params
        max_source_length = self.cfg.get("max_source_length", 1024)
        max_target_length = self.cfg.get("max_target_length", 512)
        num_beams = self.cfg.get("num_beams", 4)
        cache_dir = self.cfg.get("cache_dir")
        torch_dtype = self.cfg.get("torch_dtype")

        # Create compressor
        self.compressor = RECOMPAbstractiveCompressor(
            model_path=model_path,
            device=device,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            num_beams=num_beams,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
        )

        logger.info(f"RECOMP Abstractive Compressor initialized: {model_path}")
        logger.info(
            f"Parameters: max_source_length={max_source_length}, "
            f"max_target_length={max_target_length}, num_beams={num_beams}"
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
            return result_data

        # Extract text from retrieval results
        docs_text = self._extract_text_from_results(retrieval_results)

        if not self.enabled:
            # Baseline mode: use original retrieval_results as refining_results
            result_data = data.copy()
            result_data["refining_results"] = docs_text
            logger.info("RECOMP Abstractive disabled - passing through original documents")
            return result_data

        # Construct original context
        original_context = "\n\n".join(docs_text)

        # Log input statistics
        logger.info(
            f"RECOMP Abstractive: Processing {len(docs_text)} documents, query: '{query[:50]}...'"
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

            # Log compression results
            logger.info(
                f"RECOMP Abstractive Compression: {original_tokens} -> {compressed_tokens} tokens "
                f"({compression_rate:.1%})"
            )
            logger.debug(f"Compressed text preview: {compressed_text[:200]}...")

            # Build result - use refining_results for promptor
            result_data = data.copy()
            result_data["refining_results"] = [compressed_text]  # Promptor expects list
            result_data["compressed_context"] = compressed_text
            result_data["original_tokens"] = original_tokens
            result_data["compressed_tokens"] = compressed_tokens
            result_data["compression_rate"] = compression_rate

            return result_data

        except Exception as e:
            logger.error(f"RECOMP Abstractive Compression failed: {e}", exc_info=True)

            # Fallback: use original documents as refining_results
            original_tokens = self.compressor._count_tokens(original_context)

            result_data = data.copy()
            result_data["refining_results"] = docs_text
            result_data["compressed_context"] = original_context
            result_data["original_tokens"] = original_tokens
            result_data["compressed_tokens"] = original_tokens
            result_data["compression_rate"] = 1.0

            logger.warning(
                f"Fallback to original documents due to compression error. "
                f"Stats: {original_tokens} tokens"
            )
            return result_data
