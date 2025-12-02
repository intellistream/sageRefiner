"""
LLMLingua-2 Operator for SAGE Pipeline
======================================

Wraps the LLMLingua-2 compression algorithm as a SAGE MapOperator for RAG pipelines.
Uses BERT-based token classification for fast and accurate prompt compression.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

from sage.common.core.functions import MapFunction as MapOperator

from .compressor import LLMLingua2Compressor

logger = logging.getLogger(__name__)


class LLMLingua2Operator(MapOperator):
    """LLMLingua-2 Operator for SAGE Pipeline.

    This operator wraps the LLMLingua-2 compression algorithm for use in
    SAGE RAG pipelines. It uses BERT-based token classification for fast
    and accurate prompt compression.

    Input Format:
        {
            "query": str,
            "retrieval_results": List[dict],  # Retrieved documents
        }

    Output Format:
        {
            "query": str,
            "retrieval_results": List[dict],  # Original documents (preserved)
            "refining_results": List[str],    # Compressed document list
            "compressed_context": str,         # Full compressed context
            "original_tokens": int,
            "compressed_tokens": int,
            "compression_rate": float,
        }

    Configuration Options:
        enabled (bool): Whether compression is enabled. Default True.
        model_name (str): LLMLingua-2 model name. Default is the BERT-based model.
        device (str): Device for inference ("cuda", "cpu"). Default "cuda".
        rate (float): Compression rate (0.0-1.0). Default 0.5.
        target_token (int): Target token count. -1 means use rate.
        use_context_level_filter (bool): Use context-level filtering. Default True.
        use_token_level_filter (bool): Use token-level filtering. Default True.
        force_tokens (List[str]): Tokens to preserve. Default ["\\n", ".", "?"].
        force_reserve_digit (bool): Preserve digit-containing tokens. Default False.
        drop_consecutive (bool): Drop consecutive force tokens. Default False.

    Example:
        >>> config = {
        ...     "enabled": True,
        ...     "model_name": "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        ...     "device": "cuda",
        ...     "rate": 0.5,
        ... }
        >>> operator = LLMLingua2Operator(config)
        >>> result = operator.execute({
        ...     "query": "What is machine learning?",
        ...     "retrieval_results": [{"text": "Machine learning is..."}],
        ... })
    """

    def __init__(self, config: dict[str, Any], ctx: Any = None):
        """Initialize the LLMLingua-2 operator.

        Args:
            config: Configuration dictionary.
            ctx: Optional context object.
        """
        super().__init__(config=config, ctx=ctx)
        self.cfg = config
        self.enabled = config.get("enabled", True)

        if self.enabled:
            self._init_compressor()
            logger.info("LLMLingua2 Operator initialized")
        else:
            self._compressor = None
            logger.info("LLMLingua2 disabled (baseline mode)")

    def _init_compressor(self) -> None:
        """Initialize the LLMLingua-2 compressor."""
        self._compressor = LLMLingua2Compressor(
            model_name=self.cfg.get("model_name"),
            device=self.cfg.get("device", "cuda"),
            max_batch_size=self.cfg.get("max_batch_size", 50),
            max_force_token=self.cfg.get("max_force_token", 100),
        )
        logger.info(f"LLMLingua2 Compressor initialized with model: {self._compressor.model_name}")

    def execute(self, data: dict[str, Any]) -> dict[str, Any]:
        """Execute compression on input data.

        Args:
            data: Dictionary containing query and retrieval_results.

        Returns:
            Dictionary with added compression results.
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
            # Baseline mode: pass through original documents
            result_data = data.copy()
            docs_text = self._extract_document_texts(retrieval_results)
            result_data["refining_results"] = docs_text
            result_data["compressed_context"] = "\n\n".join(docs_text)
            logger.info("LLMLingua2 disabled - passing through original documents")
            return result_data

        # Extract document texts
        docs_text = self._extract_document_texts(retrieval_results)

        # Log input statistics
        logger.info(f"LLMLingua2: Processing {len(docs_text)} documents, query: '{query[:50]}...'")

        try:
            # Get compression parameters from config
            rate = self.cfg.get("rate", 0.5)
            target_token = self.cfg.get("target_token", -1)
            use_context_level_filter = self.cfg.get("use_context_level_filter", True)
            use_token_level_filter = self.cfg.get("use_token_level_filter", True)
            force_tokens = self.cfg.get("force_tokens", ["\n", ".", "?", "!"])
            force_reserve_digit = self.cfg.get("force_reserve_digit", False)
            drop_consecutive = self.cfg.get("drop_consecutive", False)

            # Perform compression
            compress_result = self._compressor.compress(
                context=docs_text,
                rate=rate,
                target_token=target_token,
                use_context_level_filter=use_context_level_filter,
                use_token_level_filter=use_token_level_filter,
                force_tokens=force_tokens,
                force_reserve_digit=force_reserve_digit,
                drop_consecutive=drop_consecutive,
            )

            # Extract results
            compressed_prompt = compress_result["compressed_prompt"]
            compressed_prompt_list = compress_result.get(
                "compressed_prompt_list", [compressed_prompt]
            )
            origin_tokens = compress_result["origin_tokens"]
            compressed_tokens = compress_result["compressed_tokens"]

            # Calculate compression rate
            compression_rate = compressed_tokens / origin_tokens if origin_tokens > 0 else 1.0

            # Log compression results
            logger.info(
                f"LLMLingua2 Compression: {origin_tokens} -> {compressed_tokens} tokens "
                f"({compression_rate:.1%})"
            )
            logger.debug(f"Compressed text preview: {compressed_prompt[:200]}...")

            # Build result
            result_data = data.copy()
            result_data["refining_results"] = compressed_prompt_list
            result_data["compressed_context"] = compressed_prompt
            result_data["original_tokens"] = origin_tokens
            result_data["compressed_tokens"] = compressed_tokens
            result_data["compression_rate"] = compression_rate

            return result_data

        except Exception as e:
            logger.error(f"LLMLingua2 compression failed: {e}", exc_info=True)
            # Fallback: return original documents
            result_data = data.copy()
            result_data["refining_results"] = docs_text
            result_data["compressed_context"] = "\n\n".join(docs_text)
            result_data["original_tokens"] = 0
            result_data["compressed_tokens"] = 0
            result_data["compression_rate"] = 1.0
            return result_data

    def _extract_document_texts(self, retrieval_results: list[dict[str, Any] | str]) -> list[str]:
        """Extract text content from retrieval results.

        Args:
            retrieval_results: List of retrieval result dictionaries or strings.

        Returns:
            List of document text strings.
        """
        docs_text = []
        for result in retrieval_results:
            if isinstance(result, dict):
                # Try common text field names
                text = (
                    result.get("text")
                    or result.get("contents")
                    or result.get("content")
                    or result.get("passage")
                    or str(result)
                )
                # Optionally prepend title
                title = result.get("title", "")
                if title and not text.startswith(title):
                    text = f"{title}\n{text}"
            else:
                text = str(result)
            docs_text.append(text)
        return docs_text

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LLMLingua2Operator(enabled={self.enabled}, "
            f"model={self._compressor.model_name if self._compressor else 'N/A'})"
        )
