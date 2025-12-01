"""
LLMLingua Compression Operator for SAGE Pipeline
=================================================

将 LLMLingua 压缩算法封装为 SAGE MapOperator，用于 RAG pipeline。
基于 perplexity 的 token 级压缩算法。

特性:
- LLMLingua-2: 快速 BERT-based 压缩
- LLMLingua-1: 高质量 LLM-based 压缩
- 自动回退: 如果 LLMLingua 不可用，使用简单截断
"""

import logging
import time

from sage.common.core.functions import MapFunction as MapOperator

from .compressor import LLMLinguaCompressor

logger = logging.getLogger(__name__)


class LLMLinguaRefinerOperator(MapOperator):
    """LLMLingua Refiner Operator

    在 RAG pipeline 中使用 LLMLingua 算法压缩检索到的上下文。

    特性:
        - Perplexity-based: 基于困惑度移除低信息量 token
        - Query-aware: 支持问题感知压缩
        - Token-level: 细粒度 token 级压缩

    输入格式:
        {
            "query": str,
            "retrieval_results": List[dict],  # 检索到的文档
        }

    输出格式:
        {
            "query": str,
            "retrieval_results": List[dict],  # 原始文档（保留）
            "refining_results": List[str],    # 压缩后的文档列表
            "compressed_context": str,         # 压缩后的完整上下文
            "original_tokens": int,
            "compressed_tokens": int,
            "compression_rate": float,
            "refine_time": float,             # 压缩耗时(秒)
        }
    """

    def __init__(self, config: dict, ctx=None):
        super().__init__(config=config, ctx=ctx)
        self.cfg = config
        self.enabled = config.get("enabled", True)

        if self.enabled:
            self._init_compressor()
            logger.info("LLMLingua Refiner Operator initialized")
        else:
            logger.info("LLMLingua Refiner disabled (baseline mode)")

    def _init_compressor(self):
        """初始化 LLMLingua 压缩器"""
        self.compressor = LLMLinguaCompressor(
            model_name=self.cfg.get(
                "model_name",
                "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            ),
            use_llmlingua2=self.cfg.get("use_llmlingua2", True),
            device=self.cfg.get("device", "cuda"),
            target_token=self.cfg.get("target_token", -1),
            context_budget=self.cfg.get("context_budget", "+100"),
            use_sentence_level_filter=self.cfg.get("use_sentence_level_filter", False),
            use_context_level_filter=self.cfg.get("use_context_level_filter", True),
            use_token_level_filter=self.cfg.get("use_token_level_filter", True),
            force_tokens=self.cfg.get("force_tokens", ["\n", ".", "?", "!", ","]),
            drop_consecutive=self.cfg.get("drop_consecutive", False),
        )

        logger.info(
            f"LLMLingua Compressor initialized: "
            f"model={self.cfg.get('model_name', 'llmlingua-2-bert')}, "
            f"llmlingua2={self.cfg.get('use_llmlingua2', True)}"
        )

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
            result_data["refine_time"] = 0.0
            return result_data

        if not self.enabled:
            # Baseline mode: use original retrieval_results as refining_results
            result_data = data.copy()
            docs_text = self._extract_texts(retrieval_results)
            result_data["refining_results"] = docs_text
            result_data["compressed_context"] = "\n\n".join(docs_text)
            logger.info("LLMLingua Refiner disabled - passing through original documents")
            return result_data

        # Extract text from retrieval results
        docs_text = self._extract_texts(retrieval_results)

        # Construct original context
        original_context = "\n\n".join(docs_text)

        # Log input statistics
        logger.info(f"LLMLingua: Processing {len(docs_text)} documents, query: '{query[:50]}...'")

        try:
            # Get compression configuration
            budget = self.cfg.get("budget", 2048)
            compression_ratio = self.cfg.get("compression_ratio", None)

            # Compress with timing
            start_time = time.time()
            compress_result = self.compressor.compress(
                context=original_context,
                question=query,
                budget=budget,
                compression_ratio=compression_ratio,
            )
            refine_time = time.time() - start_time

            # Extract results
            compressed_text = compress_result.compressed_context
            original_tokens = compress_result.original_tokens
            compressed_tokens = compress_result.compressed_tokens
            compression_rate = compress_result.compression_rate

            # Log compression results
            logger.info(
                f"LLMLingua Compression: {original_tokens} -> {compressed_tokens} tokens "
                f"({compression_rate:.1%}), time={refine_time:.2f}s"
            )
            logger.debug(f"Compressed text preview: {compressed_text[:200]}...")

            # Build result - use refining_results for promptor
            result_data = data.copy()
            result_data["refining_results"] = [compressed_text]  # Promptor expects list
            result_data["compressed_context"] = compressed_text
            result_data["original_tokens"] = original_tokens
            result_data["compressed_tokens"] = compressed_tokens
            result_data["compression_rate"] = compression_rate
            result_data["refine_time"] = refine_time

            return result_data

        except Exception as e:
            logger.error(f"LLMLingua compression failed: {e}", exc_info=True)
            # Fallback: use original documents as refining_results
            result_data = data.copy()
            result_data["refining_results"] = docs_text
            result_data["compressed_context"] = original_context
            result_data["original_tokens"] = 0
            result_data["compressed_tokens"] = 0
            result_data["compression_rate"] = 1.0
            result_data["refine_time"] = 0.0
            logger.warning("Fallback to original documents due to compression error")
            return result_data

    def _extract_texts(self, retrieval_results: list) -> list[str]:
        """从检索结果中提取文本

        Args:
            retrieval_results: 检索结果列表

        Returns:
            文本列表
        """
        docs_text = []
        for result in retrieval_results:
            if isinstance(result, dict):
                # Try different possible keys
                if "contents" in result:
                    text = result["contents"]
                elif "text" in result:
                    title = result.get("title", "")
                    text_content = result["text"]
                    text = f"{title}\n{text_content}" if title else text_content
                else:
                    text = str(result)
            else:
                text = str(result)
            docs_text.append(text)
        return docs_text
