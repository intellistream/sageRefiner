"""
LongLLMLingua Compression Operator for SAGE Pipeline
=====================================================

Wraps LongLLMLingua compressor as a SAGE MapOperator for RAG pipelines.
Provides question-aware context compression optimized for long documents.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sage.common.core.functions import MapFunction as MapOperator

from .compressor import DEFAULT_LONG_LLMLINGUA_CONFIG, LongLLMLinguaCompressor

if TYPE_CHECKING:
    pass


class LongLLMLinguaOperator(MapOperator):
    """LongLLMLingua Refiner Operator for SAGE Pipeline

    Applies LongLLMLingua compression algorithm in RAG pipelines to reduce
    context length while preserving question-relevant information.

    NOTE: By default, this operator uses paper-baseline settings for LongLLMLingua
    (question-aware ranking + contrastive perplexity with condition_compare=True).
    Override specific parameters in config for ablation studies.

    Input Format:
        {
            "query": str,                              # User query/question
            "retrieval_results": List[str | dict],    # Retrieved documents
        }

        Note: retrieval_results can be either:
            - List of plain strings (document texts), or
            - List of dicts containing text under keys like "text", "contents",
              "content", or "passage"

    Output Format:
        {
            "query": str,                    # Original query (preserved)
            "retrieval_results": List[...],  # Original documents (preserved)
            "refining_results": List[str],   # Compressed context for downstream
            "compressed_context": str,       # Compressed text
            "original_tokens": int,          # Token count before compression
            "compressed_tokens": int,        # Token count after compression
            "compression_rate": float,       # Compression rate (0-1)
            "compression_ratio": str,        # Human-readable ratio (e.g., "2.5x")
            "compression_failed": bool,      # True if fallback to original docs
        }

    Configuration (all optional, defaults to paper baseline):
        enabled: bool - Enable/disable compression (default: True)
        model_name: str - Model for compression (default: Llama-2-7b)
        device: str - Device for inference (default: cuda)
        rate: float - Target compression rate (default: 0.55, paper baseline)
        target_token: int - Target token count (overrides rate if > 0)
        condition_in_question: str - Question conditioning mode (default: "after")
        reorder_context: str - Context reordering strategy (default: "sort")
        dynamic_context_compression_ratio: float - Dynamic compression ratio (default: 0.3)
        use_context_level_filter: bool - Enable document-level filtering (default: True)
        use_sentence_level_filter: bool - Enable sentence-level filtering (default: False)
        use_token_level_filter: bool - Enable token-level compression (default: True)
        context_budget: str - Token budget expression (default: "+100")
        condition_compare: bool - Enable contrastive perplexity (default: True, paper baseline)
        iterative_size: int - Tokens per iteration (default: 200)
    """

    def __init__(self, config: dict[str, Any], ctx: Any = None):
        """Initialize LongLLMLingua Operator

        Args:
            config: Operator configuration dictionary
            ctx: SAGE execution context (optional)
        """
        super().__init__(config=config, ctx=ctx)
        self.logger = logging.getLogger(__name__)
        self.cfg = config
        self.enabled = config.get("enabled", True)
        self._compressor = None

        if self.enabled:
            self._init_compressor()
            self.logger.info("LongLLMLingua Compression Operator initialized")
        else:
            self.logger.info("LongLLMLingua Compression disabled (baseline mode)")

    def _init_compressor(self) -> None:
        """Initialize the LongLLMLingua compressor"""
        model_name = self.cfg.get("model_name") or self.cfg.get("model_path")
        device = self.cfg.get("device", "cuda")
        model_config = self.cfg.get("model_config", {})

        self._compressor = LongLLMLinguaCompressor(
            model_name=model_name,
            device=device,
            model_config=model_config,
        )

        self.logger.debug(f"LongLLMLingua compressor created: model={model_name}, device={device}")

    @property
    def compressor(self) -> LongLLMLinguaCompressor:
        """Get the compressor instance"""
        if self._compressor is None:
            raise RuntimeError("Compressor not initialized. Check if enabled=True in config.")
        return self._compressor

    def execute(self, data: dict[str, Any]) -> dict[str, Any]:
        """Execute compression on input data

        Args:
            data: Input dictionary with query and retrieval_results

        Returns:
            Output dictionary with compression results
        """
        if not isinstance(data, dict):
            self.logger.error(f"Unexpected input format: {type(data)}")
            return data

        query = data.get("query", "")
        retrieval_results = data.get("retrieval_results", [])

        # Handle empty retrieval results
        if not retrieval_results:
            self.logger.warning(f"No retrieval results for query: '{query[:50]}...'")
            result_data = data.copy()
            result_data["refining_results"] = []
            result_data["compressed_context"] = ""
            result_data["original_tokens"] = 0
            result_data["compressed_tokens"] = 0
            result_data["compression_rate"] = 1.0
            result_data["compression_failed"] = False
            return result_data

        # Baseline mode: pass through without compression
        if not self.enabled:
            docs_text = self._extract_document_texts(retrieval_results)
            result_data = data.copy()
            result_data["refining_results"] = docs_text
            result_data["compression_failed"] = False
            self.logger.debug("LongLLMLingua disabled - passing through original documents")
            return result_data

        # Extract document texts
        docs_text = self._extract_document_texts(retrieval_results)

        self.logger.debug(
            f"LongLLMLingua: Processing {len(docs_text)} documents, query: '{query[:50]}...'"
        )

        try:
            # Get compression parameters from config, falling back to paper baseline defaults
            cfg = DEFAULT_LONG_LLMLINGUA_CONFIG

            rate = self.cfg.get("rate", cfg["rate"])
            target_token = self.cfg.get("target_token")  # None is fine; overrides rate if set
            condition_in_question = self.cfg.get(
                "condition_in_question", cfg["condition_in_question"]
            )
            reorder_context = self.cfg.get("reorder_context", cfg["reorder_context"])
            dynamic_ratio = self.cfg.get(
                "dynamic_context_compression_ratio", cfg["dynamic_context_compression_ratio"]
            )
            use_context_filter = self.cfg.get(
                "use_context_level_filter", cfg["use_context_level_filter"]
            )
            use_sentence_filter = self.cfg.get(
                "use_sentence_level_filter", cfg["use_sentence_level_filter"]
            )
            use_token_filter = self.cfg.get("use_token_level_filter", cfg["use_token_level_filter"])
            context_budget = self.cfg.get("context_budget", cfg["context_budget"])
            condition_compare = self.cfg.get("condition_compare", cfg["condition_compare"])
            iterative_size = self.cfg.get("iterative_size", cfg["iterative_size"])

            # Perform compression with paper baseline defaults
            compress_result = self.compressor.compress(
                context=docs_text,
                question=query,
                rate=rate,
                target_token=target_token,
                condition_in_question=condition_in_question,
                reorder_context=reorder_context,
                dynamic_context_compression_ratio=dynamic_ratio,
                use_context_level_filter=use_context_filter,
                use_sentence_level_filter=use_sentence_filter,
                use_token_level_filter=use_token_filter,
                context_budget=context_budget,
                condition_compare=condition_compare,
                iterative_size=iterative_size,
            )

            compressed_text = compress_result["compressed_prompt"]
            original_tokens = compress_result["origin_tokens"]
            compressed_tokens = compress_result["compressed_tokens"]

            # Calculate compression rate
            compression_rate = compressed_tokens / original_tokens if original_tokens > 0 else 1.0

            self.logger.info(
                f"LongLLMLingua Compression: {original_tokens} -> {compressed_tokens} tokens "
                f"({compression_rate:.1%})"
            )

            # Build result
            result_data = data.copy()
            result_data["refining_results"] = [compressed_text]  # Promptor expects list
            result_data["compressed_context"] = compressed_text
            result_data["original_tokens"] = original_tokens
            result_data["compressed_tokens"] = compressed_tokens
            result_data["compression_rate"] = compression_rate
            result_data["compression_ratio"] = compress_result.get("ratio", "N/A")
            result_data["compression_failed"] = False

            return result_data

        except Exception as e:
            self.logger.error(f"LongLLMLingua Compression failed: {e}", exc_info=True)
            # Fallback: use original documents with realistic token statistics
            original_text = "\n\n".join(docs_text)

            # Try to compute actual token count if compressor is available
            try:
                original_tokens = self.compressor.get_token_length(original_text)
            except Exception:
                # Rough estimate: ~4 chars per token
                original_tokens = len(original_text) // 4

            result_data = data.copy()
            result_data["refining_results"] = docs_text
            result_data["compressed_context"] = original_text
            result_data["original_tokens"] = original_tokens
            result_data["compressed_tokens"] = original_tokens  # No compression
            result_data["compression_rate"] = 1.0
            result_data["compression_failed"] = True
            self.logger.warning("Fallback to original documents due to compression error")
            return result_data

    def _extract_document_texts(self, retrieval_results: list) -> list[str]:
        """Extract text content from retrieval results

        Args:
            retrieval_results: List of documents. Each item can be either:
                - A plain string (the document text), or
                - A dict containing text under common keys like "text",
                  "contents", "content", or "passage"

        Returns:
            List of document texts as strings
        """
        docs_text = []
        for result in retrieval_results:
            if isinstance(result, dict):
                # Handle dict format: try common keys
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
