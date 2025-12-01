"""
Adaptive Compressor Operator for SAGE Pipeline
===============================================

将AdaptiveCompressor压缩算法封装为SAGE MapOperator，用于RAG pipeline。
支持Query感知的多粒度上下文压缩。
"""

import logging
import time

from sage.common.core.functions import MapFunction as MapOperator

from .compressor import AdaptiveCompressor

logger = logging.getLogger(__name__)


class AdaptiveRefinerOperator(MapOperator):
    """Adaptive Refiner Operator

    在RAG pipeline中使用AdaptiveCompressor算法压缩检索到的上下文。

    特性:
        - Query感知: 根据query类型调整压缩策略
        - 多粒度级联: 段落→句子→短语 级联压缩
        - MMR多样性: 避免冗余，最大化信息覆盖
        - 动态预算: 复杂query自动增加预算

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
            "query_type": str,                # Query类型
            "granularity_used": str,          # 使用的粒度
        }
    """

    def __init__(self, config: dict, ctx=None):
        super().__init__(config=config, ctx=ctx)
        self.cfg = config
        self.enabled = config.get("enabled", True)

        if self.enabled:
            self._init_compressor()
            logger.info("AdaptiveRefiner Operator initialized")
        else:
            logger.info("AdaptiveRefiner disabled (baseline mode)")

    def _init_compressor(self):
        """初始化Adaptive压缩器"""
        self.compressor = AdaptiveCompressor(
            tokenizer=None,  # 自动加载
            encoder=None,  # 自动加载
            use_query_classifier=self.cfg.get("use_query_classifier", True),
            use_ner=self.cfg.get("use_ner", False),
            similarity_threshold=self.cfg.get("similarity_threshold", 0.85),
            diversity_weight=self.cfg.get("diversity_weight", 0.3),
            min_budget=self.cfg.get("min_budget", 256),
            max_budget_multiplier=self.cfg.get("max_budget_multiplier", 2.0),
        )

        logger.info(
            f"AdaptiveCompressor initialized: "
            f"query_classifier={self.cfg.get('use_query_classifier', True)}, "
            f"diversity_weight={self.cfg.get('diversity_weight', 0.3)}"
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
            result_data["refine_time"] = 0.0
            result_data["query_type"] = "unknown"
            result_data["granularity_used"] = "none"
            return result_data

        if not self.enabled:
            # Baseline mode: use original retrieval_results as refining_results
            result_data = data.copy()
            docs_text = self._extract_texts(retrieval_results)
            result_data["refining_results"] = docs_text
            result_data["compressed_context"] = "\n\n".join(docs_text)
            logger.info("AdaptiveRefiner disabled - passing through original documents")
            return result_data

        # Extract text from retrieval results
        docs_text = self._extract_texts(retrieval_results)

        # Construct original context
        original_context = "\n\n".join(docs_text)

        # Log input statistics
        logger.info(
            f"AdaptiveRefiner: Processing {len(docs_text)} documents, query: '{query[:50]}...'"
        )

        try:
            # Get budget configuration
            budget = self.cfg.get("budget", 2048)
            force_granularity = self.cfg.get("force_granularity", None)

            # Compress with timing
            start_time = time.time()
            compress_result = self.compressor.compress(
                context=original_context,
                question=query,
                budget=budget,
                force_granularity=force_granularity,
            )
            refine_time = time.time() - start_time

            # Extract results
            compressed_text = compress_result.compressed_context
            original_tokens = compress_result.original_tokens
            compressed_tokens = compress_result.compressed_tokens
            compression_rate = compress_result.compression_rate
            query_type = compress_result.query_type
            granularity_used = compress_result.granularity_used

            # Log compression results
            logger.info(
                f"AdaptiveRefiner Compression: {original_tokens} -> {compressed_tokens} tokens "
                f"({compression_rate:.1%}), query_type={query_type}, "
                f"granularity={granularity_used}, time={refine_time:.2f}s"
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
            result_data["query_type"] = query_type
            result_data["granularity_used"] = granularity_used

            return result_data

        except Exception as e:
            logger.error(f"AdaptiveRefiner compression failed: {e}", exc_info=True)
            # Fallback: use original documents as refining_results
            result_data = data.copy()
            result_data["refining_results"] = docs_text
            result_data["compressed_context"] = original_context
            result_data["original_tokens"] = 0
            result_data["compressed_tokens"] = 0
            result_data["compression_rate"] = 1.0
            result_data["refine_time"] = 0.0
            result_data["query_type"] = "error"
            result_data["granularity_used"] = "fallback"
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
