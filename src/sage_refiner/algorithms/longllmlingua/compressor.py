"""
LongLLMLingua Context Compressor
================================

Optimized prompt compression for long document scenarios using LLMLingua
with question-aware ranking.

Core Approach:
    1. Use question to guide context importance evaluation
    2. Apply perplexity-based ranking with condition_in_question="after"
    3. Dynamically adjust compression ratio across context segments
    4. Reorder context by relevance for better generation quality

Default Configuration (Paper Baseline):
    - rate: 0.55 (55% compression)
    - condition_in_question: "after" (question appended after context for PPL)
    - condition_compare: True (contrastive perplexity enabled)
    - reorder_context: "sort" (sort by relevance)
    - dynamic_context_compression_ratio: 0.3

Implementation Notes:
    - Wraps the original LLMLingua PromptCompressor with LongLLMLingua-specific settings
    - Supports both rate-based and target_token-based compression
    - Handles multi-document RAG scenarios with context-level filtering

References:
    LongLLMLingua Paper: https://arxiv.org/abs/2310.06839
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================================
# LongLLMLingua Default Configuration (Paper Baseline)
# ============================================================================
# Default configuration approximating the "paper baseline" setting for LongLLMLingua
# as used in Jiang et al. (2024, arXiv:2310.06839) and many follow-up works.
#
# Key settings for baseline reproduction:
#   - condition_compare=True: Enables contrastive perplexity (self-information comparison)
#   - condition_in_question="after": Question-aware PPL calculation
#   - reorder_context="sort": Sort context by relevance
#   - rate=0.55: Default compression rate when target_token is not specified
#
# For ablation studies, override specific parameters (e.g., condition_compare=False
# corresponds to "w/o question-aware fine-grained compression").

DEFAULT_LONG_LLMLINGUA_CONFIG: dict[str, Any] = {
    "rate": 0.55,  # Default compression rate (paper baseline)
    "condition_in_question": "after",  # Evaluate PPL with question appended
    "reorder_context": "sort",  # Sort context by importance
    "dynamic_context_compression_ratio": 0.3,  # Dynamic compression ratio
    "context_budget": "+100",  # Token budget expression
    "use_context_level_filter": True,  # Apply document-level filtering
    "use_sentence_level_filter": False,  # Sentence-level filtering (off by default)
    "use_token_level_filter": True,  # Apply token-level compression
    "condition_compare": True,  # Enable contrastive perplexity (paper baseline)
    "iterative_size": 200,  # Tokens per iteration
}


class LongLLMLinguaCompressor:
    """LongLLMLingua: Question-aware Prompt Compressor for Long Documents

    LongLLMLingua extends the original LLMLingua with optimizations for
    long context scenarios in RAG pipelines.

    NOTE: By default, this wrapper uses a configuration that matches the
    LongLLMLingua "paper baseline" settings (question-aware doc ranking +
    dynamic context compression + contrastive perplexity with condition_compare=True).
    Users can override individual parameters via config for ablation studies.

    Features:
        - Question-aware context ranking using perplexity evaluation
        - Dynamic compression ratio that adapts to content relevance
        - Context reordering to prioritize important segments
        - Support for both token-level and context-level filtering
        - Contrastive perplexity (condition_compare=True) for better quality

    Attributes:
        compressor: Underlying LLMLingua PromptCompressor instance
        model_name: Name of the compression model (e.g., Llama-2-7b)
        device: Device for model inference ('cuda', 'cpu', etc.)
        default_rate: Default compression rate (0.0 to 1.0)

    Example:
        >>> compressor = LongLLMLinguaCompressor(
        ...     model_name="NousResearch/Llama-2-7b-hf",
        ...     device="cuda"
        ... )
        >>> result = compressor.compress(
        ...     context=["Document 1...", "Document 2..."],
        ...     question="What is the main topic?",
        ...     rate=0.55  # Paper baseline rate
        ... )
        >>> print(result["compressed_prompt"])
    """

    # Default model for LongLLMLingua compression
    DEFAULT_MODEL = "NousResearch/Llama-2-7b-hf"

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cuda",
        model_config: dict[str, Any] | None = None,
        open_api_config: dict[str, Any] | None = None,
    ):
        """Initialize LongLLMLingua Compressor

        Args:
            model_name: HuggingFace model name/path for compression.
                Defaults to "NousResearch/Llama-2-7b-hf".
                Other options: "meta-llama/Llama-2-7b-hf", "microsoft/phi-2"
            device: Device for model inference.
                Options: "cuda", "cuda:0", "cpu", "auto"
            model_config: Additional configuration for model loading.
                e.g., {"trust_remote_code": True, "torch_dtype": "auto"}
            open_api_config: Configuration for OpenAI-compatible API fallback.
                e.g., {"api_key": "...", "api_base": "..."}
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self.model_config = model_config or {}
        self.open_api_config = open_api_config or {}

        # Lazy initialization - compressor created on first use
        self._compressor = None
        self._initialized = False

        logger.info(
            f"LongLLMLinguaCompressor created with model={self.model_name}, device={self.device}"
        )

    def _ensure_initialized(self) -> None:
        """Lazy initialization of the underlying PromptCompressor"""
        if self._initialized:
            return

        # Import from installed llmlingua package
        try:
            from llmlingua import PromptCompressor
        except ImportError as e:
            raise ImportError(
                "LLMLingua package not found. Please install it:\n"
                "  pip install llmlingua\n"
                "For more information, see: https://github.com/microsoft/LLMLingua"
            ) from e

        logger.info(f"Loading LongLLMLingua model: {self.model_name}")

        self._compressor = PromptCompressor(
            model_name=self.model_name,
            device_map=self.device,
            model_config=self.model_config,
            open_api_config=self.open_api_config,
            use_llmlingua2=False,  # LongLLMLingua uses LLM-based compression
        )

        # Defensive check: ensure compressor was created successfully
        if self._compressor is None:
            raise RuntimeError(
                "PromptCompressor initialization failed unexpectedly. "
                "Check model availability and device settings."
            )

        self._initialized = True
        logger.info("LongLLMLingua model loaded successfully")

    @property
    def compressor(self):
        """Get the underlying PromptCompressor (lazy loaded)"""
        self._ensure_initialized()
        return self._compressor

    def compress(
        self,
        context: list[str] | str,
        question: str,
        instruction: str = "",
        rate: float | None = None,
        target_token: int | None = None,
        # LongLLMLingua-specific parameters
        condition_in_question: str | None = None,
        reorder_context: str | None = None,
        dynamic_context_compression_ratio: float | None = None,
        # Context-level filtering
        use_context_level_filter: bool | None = None,
        context_budget: str | None = None,
        force_context_ids: list[int] | None = None,
        force_context_number: int | None = None,
        # Sentence-level filtering
        use_sentence_level_filter: bool | None = None,
        keep_first_sentence: int = 0,
        keep_last_sentence: int = 0,
        keep_sentence_number: int = 0,
        high_priority_bonus: int = 100,
        token_budget_ratio: float = 1.4,
        # Token-level filtering
        use_token_level_filter: bool | None = None,
        iterative_size: int | None = None,
        keep_split: bool = False,
        # Advanced options
        condition_compare: bool | None = None,
        # Output options
        add_instruction: bool = False,
        concate_question: bool = True,
    ) -> dict[str, Any]:
        """Compress context using LongLLMLingua algorithm

        This method applies question-aware compression optimized for long documents.
        The key difference from standard LLMLingua is the use of question context
        to guide importance scoring.

        Args:
            context: List of context strings (documents) or single context string.
                For RAG, this is typically the retrieved documents.
            question: The question/query for question-aware compression.
                Required for LongLLMLingua's relevance scoring.
            instruction: Optional instruction text to prepend.
            rate: Target compression rate (0.0 to 1.0). If None, defaults to 0.55
                (55%), which matches the LongLLMLingua paper baseline setting.
            target_token: Target number of tokens after compression.
                If specified (> 0), overrides the rate parameter.

            condition_in_question: Where to place the question relative to the
                context when computing question-aware perplexity.
                - "after": Evaluate PPL with question appended (default, recommended)
                - "before": Evaluate PPL with question prepended
                - "none": No question conditioning (standard LLMLingua)
            reorder_context: Context reordering strategy after ranking.
                - "sort": Sort by importance (default, recommended)
                - "original": Keep original order
                - "two_stage": Split-and-interleave ordering
            dynamic_context_compression_ratio: Ratio for dynamic compression.
                Higher values = more aggressive compression on less relevant parts.
                Range: 0.0 to 1.0, default 0.3.

            use_context_level_filter: Apply document-level filtering.
            context_budget: Token budget expression for context filtering.
            force_context_ids: List of context indices to always include.
            force_context_number: Number of contexts to force include.

            use_sentence_level_filter: Apply sentence-level filtering.
            keep_first_sentence: Number of first sentences to always keep.
            keep_last_sentence: Number of last sentences to always keep.
            keep_sentence_number: Total sentences to keep per document.
            high_priority_bonus: PPL bonus for priority sentences.
            token_budget_ratio: Budget ratio for sentence filtering.

            use_token_level_filter: Apply token-level compression.
            iterative_size: Tokens to process per iteration.
            keep_split: Preserve original separators.

            condition_compare: Whether to use contrastive perplexity for
                question-aware fine-grained compression. If None, defaults to
                True as in the LongLLMLingua paper baseline.

            add_instruction: Add instruction to compressed output.
            concate_question: Append question to compressed output.

        Returns:
            Dictionary containing:
                - compressed_prompt (str): The compressed prompt text
                - origin_tokens (int): Original token count
                - compressed_tokens (int): Compressed token count
                - ratio (str): Compression ratio (e.g., "2.5x")
                - rate (str): Compression rate percentage (e.g., "40.0%")
                - saving (str): Estimated cost savings

        Raises:
            ValueError: If question is empty (required for LongLLMLingua)
            ImportError: If LLMLingua dependencies are not available
        """
        # =====================================================================
        # Guard: Handle empty context
        # =====================================================================
        if not context or (isinstance(context, list) and len(context) == 0):
            logger.warning("Empty context passed to LongLLMLinguaCompressor.compress")
            return {
                "compressed_prompt": "",
                "origin_tokens": 0,
                "compressed_tokens": 0,
                "ratio": "1.0x",
                "rate": "0.0%",
                "saving": ", Saving $0.0 in GPT-4.",
            }

        # Handle single empty string
        if isinstance(context, str) and not context.strip():
            logger.warning("Empty string context passed to LongLLMLinguaCompressor.compress")
            return {
                "compressed_prompt": "",
                "origin_tokens": 0,
                "compressed_tokens": 0,
                "ratio": "1.0x",
                "rate": "0.0%",
                "saving": ", Saving $0.0 in GPT-4.",
            }

        # =====================================================================
        # Guard: Validate question (required for LongLLMLingua)
        # =====================================================================
        if not question or not question.strip():
            raise ValueError(
                "LongLLMLingua is a question-aware compression algorithm and requires "
                "a non-empty question parameter for relevance scoring. "
                "Please provide the user query or question text."
            )

        # Ensure context is a list
        if isinstance(context, str):
            context = [context]

        # =====================================================================
        # Apply defaults from centralized config
        # =====================================================================
        defaults = DEFAULT_LONG_LLMLINGUA_CONFIG

        # Use provided values or fall back to defaults
        rate = rate if rate is not None else defaults["rate"]
        condition_in_question = (
            condition_in_question
            if condition_in_question is not None
            else defaults["condition_in_question"]
        )
        reorder_context = (
            reorder_context if reorder_context is not None else defaults["reorder_context"]
        )
        dynamic_context_compression_ratio = (
            dynamic_context_compression_ratio
            if dynamic_context_compression_ratio is not None
            else defaults["dynamic_context_compression_ratio"]
        )
        use_context_level_filter = (
            use_context_level_filter
            if use_context_level_filter is not None
            else defaults["use_context_level_filter"]
        )
        use_token_level_filter = (
            use_token_level_filter
            if use_token_level_filter is not None
            else defaults["use_token_level_filter"]
        )
        use_sentence_level_filter = (
            use_sentence_level_filter
            if use_sentence_level_filter is not None
            else defaults["use_sentence_level_filter"]
        )
        context_budget = (
            context_budget if context_budget is not None else defaults["context_budget"]
        )
        condition_compare = (
            condition_compare if condition_compare is not None else defaults["condition_compare"]
        )
        iterative_size = (
            iterative_size if iterative_size is not None else defaults["iterative_size"]
        )
        target_token = target_token if target_token is not None else -1

        # Validate rate
        if rate <= 0 or rate > 1.0:
            logger.warning(f"Rate {rate} out of range, clamping to [0.01, 1.0]")
            rate = max(0.01, min(1.0, rate))

        logger.debug(
            f"LongLLMLingua compression: {len(context)} documents, "
            f"rate={rate}, target_token={target_token}"
        )
        logger.debug(f"Question preview: {question[:100]}...")

        # Call underlying compressor with LongLLMLingua-specific settings
        result = self.compressor.compress_prompt(
            context=context,
            instruction=instruction,
            question=question,
            rate=rate,
            target_token=target_token,
            iterative_size=iterative_size,
            force_context_ids=force_context_ids,
            force_context_number=force_context_number,
            use_sentence_level_filter=use_sentence_level_filter,
            use_context_level_filter=use_context_level_filter,
            use_token_level_filter=use_token_level_filter,
            keep_split=keep_split,
            keep_first_sentence=keep_first_sentence,
            keep_last_sentence=keep_last_sentence,
            keep_sentence_number=keep_sentence_number,
            high_priority_bonus=high_priority_bonus,
            context_budget=context_budget,
            token_budget_ratio=token_budget_ratio,
            # LongLLMLingua-specific: question-aware ranking
            condition_in_question=condition_in_question,
            reorder_context=reorder_context,
            dynamic_context_compression_ratio=dynamic_context_compression_ratio,
            condition_compare=condition_compare,
            add_instruction=add_instruction,
            # Key setting: use longllmlingua ranking method
            rank_method="longllmlingua",
            concate_question=concate_question,
        )

        logger.info(
            f"LongLLMLingua compression: {result['origin_tokens']} -> {result['compressed_tokens']} "
            f"tokens ({result['rate']})"
        )

        return dict(result)

    def compress_for_rag(
        self,
        documents: list[str],
        query: str,
        rate: float | None = None,
        target_token: int | None = None,
        keep_top_k_docs: int | None = None,
    ) -> dict[str, Any]:
        """Convenience method for RAG pipeline compression

        Optimized wrapper for typical RAG use cases with sensible defaults.

        Args:
            documents: Retrieved documents to compress
            query: User query for relevance scoring
            rate: Compression rate (default from DEFAULT_LONG_LLMLINGUA_CONFIG)
            target_token: Target token count (overrides rate if > 0)
            keep_top_k_docs: Force keep top K most relevant documents

        Returns:
            Compression result dictionary
        """
        return self.compress(
            context=documents,
            question=query,
            rate=rate,
            target_token=target_token,
            # RAG-optimized settings (use defaults from config)
            force_context_number=keep_top_k_docs,
            concate_question=False,  # Query usually added by downstream prompter
        )

    def get_token_length(self, text: str, use_oai_tokenizer: bool = False) -> int:
        """Get token length of text

        Args:
            text: Text to tokenize
            use_oai_tokenizer: Use OpenAI tokenizer (for cost estimation)

        Returns:
            Number of tokens
        """
        result: int = self.compressor.get_token_length(text, use_oai_tokenizer=use_oai_tokenizer)
        return result

    def __repr__(self) -> str:
        return (
            f"LongLLMLinguaCompressor(model={self.model_name}, "
            f"device={self.device}, initialized={self._initialized})"
        )
