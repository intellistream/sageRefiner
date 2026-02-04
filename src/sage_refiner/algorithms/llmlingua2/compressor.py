"""
LLMLingua-2 Compressor
======================

Core implementation of the LLMLingua-2 compression algorithm.

LLMLingua-2 uses a fine-tuned BERT model for token classification to perform
fast and accurate prompt compression. Unlike LLM-based methods, it doesn't
require running an LLM for compression, making it significantly faster.

Supported Models:
    - microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank
    - microsoft/llmlingua-2-xlm-roberta-large-meetingbank

Key Features:
    - Token-level compression via classification
    - Context-level filtering support
    - Configurable compression rate
    - Force token preservation
    - Multilingual support
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)


class LLMLingua2Compressor:
    """
    LLMLingua-2: BERT-based Fast Prompt Compression.

    This compressor uses a fine-tuned BERT model for token classification
    to compress prompts efficiently. It's much faster than LLM-based methods
    while maintaining high compression quality.

    Features:
        - Fast compression (no LLM inference needed)
        - Multilingual support via XLM-RoBERTa or mBERT
        - Token-level precise compression
        - Optional context-level filtering for coarse-to-fine compression

    Example:
        >>> compressor = LLMLingua2Compressor()
        >>> result = compressor.compress(
        ...     context=["This is the first document.", "This is the second document."],
        ...     rate=0.5,
        ... )
        >>> print(result["compressed_prompt"])

    Args:
        model_name: The name or path of the LLMLingua-2 model.
            Default is "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank".
        device: Device to load the model on ("cuda", "cpu", "cuda:0", etc.).
            Default is "cuda".
        max_batch_size: Maximum batch size for inference. Default is 50.
        max_force_token: Maximum number of force tokens. Default is 100.
    """

    # Default model for LLMLingua-2
    DEFAULT_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"

    # Alternative models
    AVAILABLE_MODELS = [
        "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    ]

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cuda",
        max_batch_size: int = 50,
        max_force_token: int = 100,
    ):
        """Initialize LLMLingua-2 compressor.

        Args:
            model_name: Model name or path. If None, uses DEFAULT_MODEL.
            device: Device for model inference.
            max_batch_size: Maximum batch size for token classification.
            max_force_token: Maximum number of tokens to force preserve.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self.max_batch_size = max_batch_size
        self.max_force_token = max_force_token

        # Lazy loading - compressor will be initialized on first use
        self._compressor = None
        self._initialized = False

        logger.info(f"LLMLingua2Compressor created with model={self.model_name}, device={device}")

    def _ensure_initialized(self) -> None:
        """Ensure the underlying PromptCompressor is initialized."""
        if self._initialized:
            return

        try:
            # Import PromptCompressor from the llmlingua PyPI package
            from llmlingua import PromptCompressor
        except ImportError as e:
            raise ImportError(
                "Failed to import PromptCompressor from llmlingua package. "
                "Please install it with: pip install llmlingua "
                f"Original error: {e}"
            ) from e

        # Initialize the underlying PromptCompressor with use_llmlingua2=True
        logger.info(f"Loading LLMLingua-2 model: {self.model_name}")
        self._compressor = PromptCompressor(
            model_name=self.model_name,
            device_map=self.device,
            use_llmlingua2=True,  # Key flag for LLMLingua-2
            llmlingua2_config={
                "max_batch_size": self.max_batch_size,
                "max_force_token": self.max_force_token,
            },
        )
        self._initialized = True
        logger.info("LLMLingua-2 model loaded successfully")

    def compress(
        self,
        context: list[str] | str,
        rate: float = 0.5,
        target_token: int = -1,
        use_context_level_filter: bool = True,
        use_token_level_filter: bool = True,
        target_context: int = -1,
        context_level_rate: float = 1.0,
        context_level_target_token: int = -1,
        force_context_ids: list[int] | None = None,
        return_word_label: bool = False,
        word_sep: str = "\t\t|\t\t",
        label_sep: str = " ",
        token_to_word: str = "mean",
        force_tokens: list[str] | None = None,
        force_reserve_digit: bool = False,
        drop_consecutive: bool = False,
        chunk_end_tokens: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compress context using LLMLingua-2.

        This method performs prompt compression using BERT-based token classification.
        It supports both context-level and token-level filtering for coarse-to-fine
        compression.

        Args:
            context: List of context strings or a single string to compress.
            rate: Compression rate target (0.0 to 1.0). Lower means more compression.
                Default is 0.5 (compress to 50% of original).
            target_token: Maximum number of tokens in output. -1 means use rate.
            use_context_level_filter: Whether to apply context-level filtering first.
                Useful for multi-document scenarios.
            use_token_level_filter: Whether to apply token-level filtering.
            target_context: Maximum number of context chunks to keep. -1 means no limit.
            context_level_rate: Rate for context-level filtering. Default is 1.0.
            context_level_target_token: Target tokens for context-level filter.
            force_context_ids: List of context indices to always keep.
            return_word_label: Whether to return word-level labels.
            word_sep: Separator for words in label output.
            label_sep: Separator between word and label.
            token_to_word: How to aggregate token probs to word probs ("mean", "first").
            force_tokens: List of tokens to always preserve (e.g., [".", "?", "\\n"]).
            force_reserve_digit: Whether to preserve tokens containing digits.
            drop_consecutive: Whether to drop consecutive forced tokens.
            chunk_end_tokens: Tokens that mark chunk boundaries. Default is [".", "\\n"].

        Returns:
            Dictionary containing:
                - "compressed_prompt": The compressed prompt string.
                - "compressed_prompt_list": List of compressed context strings.
                - "origin_tokens": Original token count.
                - "compressed_tokens": Compressed token count.
                - "ratio": Compression ratio (e.g., "2.5x").
                - "rate": Compression rate (e.g., "40.0%").
                - "saving": Estimated cost saving for GPT-4.
                - "fn_labeled_original_prompt": Word labels (if return_word_label=True).

        Example:
            >>> compressor = LLMLingua2Compressor()
            >>> result = compressor.compress(
            ...     context=["Document 1 content here.", "Document 2 content here."],
            ...     rate=0.5,
            ...     force_tokens=[".", "?", "\\n"],
            ... )
            >>> print(f"Compressed: {result['compressed_tokens']} tokens")
            >>> print(f"Ratio: {result['ratio']}")
        """
        self._ensure_initialized()
        assert self._compressor is not None

        # Handle single string input
        if isinstance(context, str):
            context = [context]

        # Set defaults for optional parameters
        if force_tokens is None:
            force_tokens = []
        if force_context_ids is None:
            force_context_ids = []
        if chunk_end_tokens is None:
            chunk_end_tokens = [".", "\n"]

        # Call the underlying compress_prompt method
        # LLMLingua-2 uses compress_prompt_llmlingua2 internally when use_llmlingua2=True
        result = self._compressor.compress_prompt(
            context=context,
            rate=rate,
            target_token=target_token,
            use_context_level_filter=use_context_level_filter,
            use_token_level_filter=use_token_level_filter,
            target_context=target_context,
            context_level_rate=context_level_rate,
            context_level_target_token=context_level_target_token,
            force_context_ids=force_context_ids,
            return_word_label=return_word_label,
            word_sep=word_sep,
            label_sep=label_sep,
            token_to_word=token_to_word,
            force_tokens=force_tokens,
            force_reserve_digit=force_reserve_digit,
            drop_consecutive=drop_consecutive,
            chunk_end_tokens=chunk_end_tokens,
        )

        logger.debug(
            f"Compression result: {result['origin_tokens']} -> {result['compressed_tokens']} "
            f"tokens ({result['rate']})"
        )

        return dict(result)

    def compress_with_question(
        self,
        context: list[str] | str,
        question: str,
        rate: float = 0.5,
        target_token: int = -1,
        use_context_level_filter: bool = True,
        force_tokens: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compress context with a question for RAG scenarios.

        This is a convenience method for RAG pipelines where you have
        a question and retrieved documents. Note that LLMLingua-2 doesn't
        use the question for compression (unlike LongLLMLingua), but this
        method provides a consistent interface.

        Args:
            context: Retrieved documents to compress.
            question: The user's question (for logging/tracking purposes).
            rate: Compression rate target.
            target_token: Target token count.
            use_context_level_filter: Whether to use context-level filtering.
            force_tokens: Tokens to preserve.

        Returns:
            Compression result dictionary.
        """
        logger.info(
            f"Compressing {len(context) if isinstance(context, list) else 1} "
            f"documents for question: '{question[:50]}...'"
        )

        return self.compress(
            context=context,
            rate=rate,
            target_token=target_token,
            use_context_level_filter=use_context_level_filter,
            use_token_level_filter=True,
            force_tokens=force_tokens or ["\n", ".", "?", "!"],
        )

    def get_token_length(self, text: str, use_oai_tokenizer: bool = False) -> int:
        """Get the token length of a text.

        Args:
            text: Text to count tokens for.
            use_oai_tokenizer: Whether to use OpenAI tokenizer for counting.

        Returns:
            Number of tokens.
        """
        self._ensure_initialized()
        assert self._compressor is not None
        result: int = self._compressor.get_token_length(text, use_oai_tokenizer=use_oai_tokenizer)
        return result

    @property
    def tokenizer(self):
        """Get the underlying tokenizer."""
        self._ensure_initialized()
        return self._compressor.tokenizer

    @property
    def model(self):
        """Get the underlying model."""
        self._ensure_initialized()
        return self._compressor.model

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LLMLingua2Compressor(model_name='{self.model_name}', "
            f"device='{self.device}', initialized={self._initialized})"
        )
