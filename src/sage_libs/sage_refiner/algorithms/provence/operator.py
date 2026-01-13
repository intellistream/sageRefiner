"""
Provence Compression Operator for SAGE Pipeline
=================================================

将Provence压缩算法封装为SAGE MapOperator，用于RAG pipeline。
基于句子级别的上下文剪枝算法。
"""

import logging

from sage.common.core.functions import MapFunction as MapOperator

from .compressor import ProvenceCompressor

logger = logging.getLogger(__name__)


class ProvenceRefinerOperator(MapOperator):
    """Provence Refiner算子

    在RAG pipeline中使用Provence算法压缩检索到的上下文。

    输入格式:
        {
            "query": str,
            "retrieval_results": List[str] or List[dict],  # 检索到的文档
        }

    输出格式:
        {
            "query": str,
            "retrieval_results": List[str],  # 原始文档（保留）
            "refining_results": List[str],   # 压缩后的文档列表
            "compressed_context": str,        # 压缩后的上下文
            "original_tokens": int,
            "compressed_tokens": int,
            "compression_rate": float,
        }
    """

    def __init__(self, config: dict, ctx=None):
        super().__init__(config=config, ctx=ctx)
        self.cfg = config
        self.enabled = config.get("enabled", True)

        # 无论 enabled 与否，都初始化 compressor，这样 baseline 和 fallback 也能用 tokenizer
        self._init_compressor()

        if self.enabled:
            logger.info("Provence Compression Operator initialized")
        else:
            logger.info("Provence Compression disabled (baseline mode)")

    def _init_compressor(self):
        """初始化Provence压缩器"""
        self.compressor = ProvenceCompressor(
            model_name=self.cfg.get("model_name", "naver/provence-reranker-debertav3-v1"),
            threshold=self.cfg.get("threshold", 0.1),
            batch_size=self.cfg.get("batch_size", 32),
            always_select_title=self.cfg.get("always_select_title", True),
            enable_warnings=self.cfg.get("enable_warnings", True),
            reorder=self.cfg.get("reorder", False),
            top_k=self.cfg.get("top_k", 5),
            device=self.cfg.get("device"),
        )

        logger.info(
            f"Provence Compressor initialized: model={self.cfg.get('model_name')}, "
            f"threshold={self.cfg.get('threshold', 0.1)}"
        )

    def execute(self, data: dict) -> dict:
        """执行压缩

        Args:
            data: 包含query和retrieval_results的字典

        Returns:
            添加了refining_results等压缩结果的字典
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

        if not self.enabled:
            # Baseline mode: use original retrieval_results as refining_results
            result_data = data.copy()
            # Extract text from retrieval results for refining_results
            docs_text = []
            for result in retrieval_results:
                if isinstance(result, dict):
                    text = result.get("text") or result.get("contents") or str(result)
                else:
                    text = str(result)
                docs_text.append(text)

            # 构建与启用模式一致的输出字段
            original_text = "\n\n".join(docs_text)
            original_tokens = len(self.compressor.tokenizer.encode(original_text))

            result_data["refining_results"] = docs_text
            result_data["compressed_context"] = original_text
            result_data["original_tokens"] = original_tokens
            result_data["compressed_tokens"] = original_tokens
            result_data["compression_rate"] = 1.0  # 未压缩

            logger.info("Provence disabled - passing through original documents")
            return result_data

        # Extract text from retrieval results
        docs_text = []
        for result in retrieval_results:
            if isinstance(result, dict):
                text = result.get("text") or result.get("contents") or str(result)
            else:
                text = str(result)
            docs_text.append(text)

        # Log input statistics
        logger.info(f"Provence: Processing {len(docs_text)} documents, query: '{query[:50]}...'")

        try:
            # 使用批量压缩接口
            compress_results = self.compressor.batch_compress(
                question_list=[query],
                context_list=[docs_text],
            )

            compress_result = compress_results[0]
            compressed_text = compress_result["compressed_context"]
            pruned_docs = compress_result["pruned_docs"]
            original_tokens = compress_result["original_tokens"]
            compressed_tokens = compress_result["compressed_tokens"]
            compression_rate = compress_result["compression_rate"]

            # Log compression results
            logger.info(
                f"Provence Compression: {original_tokens} → {compressed_tokens} tokens "
                f"({compression_rate:.1%})"
            )
            logger.debug(f"Compressed text preview: {compressed_text[:200]}...")

            # Build result - use refining_results for promptor
            result_data = data.copy()
            result_data["refining_results"] = pruned_docs  # List of pruned document strings
            result_data["compressed_context"] = compressed_text
            result_data["original_tokens"] = original_tokens
            result_data["compressed_tokens"] = compressed_tokens
            result_data["compression_rate"] = compression_rate

            return result_data

        except Exception as e:
            logger.error(f"Provence Compression failed: {e}", exc_info=True)
            # Fallback: use original documents as refining_results
            result_data = data.copy()
            original_text = "\n\n".join(docs_text)
            original_tokens = len(self.compressor.tokenizer.encode(original_text))

            result_data["refining_results"] = docs_text
            result_data["compressed_context"] = original_text
            result_data["original_tokens"] = original_tokens
            result_data["compressed_tokens"] = original_tokens
            result_data["compression_rate"] = 1.0  # 未压缩（回退）

            logger.warning("Fallback to original documents due to compression error")
            return result_data
