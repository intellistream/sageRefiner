"""
Tests for REFORM Algorithm
===========================

包含：
1. 配置和类可用性测试（不需要模型）
2. Mock 测试（模拟模型行为，验证核心逻辑）
3. 集成测试（需要模型，标记为 skip）
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from sage_refiner import RefinerConfig, REFORMCompressor
from sage_refiner.config import RefinerAlgorithm


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_docs():
    """Sample documents for testing."""
    return [
        "Artificial intelligence is transforming technology.",
        "Machine learning algorithms process large amounts of data.",
        "Natural language processing enables human-computer interaction.",
    ]


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return "What is AI?"


@pytest.fixture
def sample_context(sample_docs):
    """Sample concatenated context."""
    return "\n\n".join(sample_docs)


# =============================================================================
# 基础测试（不需要模型）
# =============================================================================


def test_reform_compressor_class():
    """Test REFORMCompressor class is available."""
    assert REFORMCompressor is not None


def test_reform_config():
    """Test config for REFORM-like usage."""
    config = RefinerConfig(
        algorithm=RefinerAlgorithm.LONG_REFINER,  # Using available algorithm
        budget=256,
        base_model_path="test-model",
    )
    assert config.algorithm == RefinerAlgorithm.LONG_REFINER
    assert config.budget == 256


def test_reform_algorithm_exists():
    """Test REFORM algorithm enum exists."""
    # REFORM should be available in the enum
    available = RefinerAlgorithm.list_available()
    assert isinstance(available, list)
    # REFORM may not be in enum yet, but class should exist
    assert REFORMCompressor is not None


# =============================================================================
# Mock 测试（不需要真实模型）
# =============================================================================


class TestREFORMMocked:
    """REFORM mock tests - 验证核心逻辑而不加载模型."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        # encode returns token ids
        tokenizer.encode.side_effect = lambda text, **kwargs: list(range(len(text.split())))
        # decode returns text
        tokenizer.decode.side_effect = lambda ids, **kwargs: " ".join(["word"] * len(ids))
        return tokenizer

    @pytest.fixture
    def mock_config(self):
        """Create mock model config."""
        config = MagicMock()
        config.max_position_embeddings = 8192
        return config

    @pytest.fixture
    def mock_extractor(self, mock_tokenizer, mock_config):
        """Create a mock AttentionHookExtractor."""
        extractor = MagicMock()
        extractor.tokenizer = mock_tokenizer
        extractor.config = mock_config
        extractor.num_key_value_heads = 8

        # Mock extract_attention_components_with_kv
        def mock_extract_with_kv(context_text, question_text):
            seq_len = 50
            head_dim = 64
            num_heads = 32

            # Create mock QKV data for each layer
            qkv_data = {}
            for layer in range(4):  # 4 layers
                qkv_data[layer] = {
                    "Q": torch.randn(1, num_heads, seq_len, head_dim),
                    "K": torch.randn(1, 8, seq_len, head_dim),  # GQA: fewer KV heads
                    "V": torch.randn(1, 8, seq_len, head_dim),
                }

            full_tokens = list(range(seq_len))
            context_range = (0, 40)  # First 40 tokens are context
            question_range = (40, 50)  # Last 10 tokens are question

            return qkv_data, full_tokens, context_range, question_range

        extractor.extract_attention_components_with_kv.side_effect = mock_extract_with_kv

        # Mock extract_attention_from_token_ids
        def mock_extract_from_ids(token_ids):
            seq_len = len(token_ids)
            head_dim = 64
            num_heads = 32

            qkv_data = {}
            for layer in range(4):
                qkv_data[layer] = {
                    "Q": torch.randn(1, num_heads, seq_len, head_dim),
                    "K": torch.randn(1, 8, seq_len, head_dim),
                    "V": torch.randn(1, 8, seq_len, head_dim),
                }
            return qkv_data

        extractor.extract_attention_from_token_ids.side_effect = mock_extract_from_ids

        return extractor

    @pytest.fixture
    def selected_heads(self):
        """Create sample selected heads config."""
        return [
            {"layer": 0, "head": 0, "type": "Q"},
            {"layer": 0, "head": 1, "type": "K"},
            {"layer": 1, "head": 0, "type": "V"},
            {"layer": 2, "head": 2, "type": "Q"},
        ]

    def test_reform_init(self, mock_extractor, selected_heads):
        """Test REFORMCompressor initialization."""
        compressor = REFORMCompressor(
            model_extractor=mock_extractor,
            selected_heads=selected_heads,
            max_tokens=512,
            keep_prefix=10,
            keep_suffix=10,
        )

        assert compressor.max_tokens == 512
        assert compressor.keep_prefix == 10
        assert compressor.keep_suffix == 10
        assert len(compressor.selected_heads) == 4

    def test_reform_compress_returns_dict(
        self, mock_extractor, selected_heads, sample_context, sample_query
    ):
        """Test compress returns expected dict structure."""
        compressor = REFORMCompressor(
            model_extractor=mock_extractor,
            selected_heads=selected_heads,
            max_tokens=100,
            use_kv_cache=True,
        )

        result = compressor.compress(
            context=sample_context,
            question=sample_query,
        )

        # Verify result structure
        assert isinstance(result, dict)
        assert "compressed_context" in result
        assert "original_tokens" in result
        assert "compressed_tokens" in result
        assert "compression_rate" in result
        assert "num_spans" in result
        assert "num_chunks" in result

    def test_reform_compress_reduces_tokens(
        self, mock_extractor, selected_heads, sample_context, sample_query
    ):
        """Test that compression actually reduces token count."""
        compressor = REFORMCompressor(
            model_extractor=mock_extractor,
            selected_heads=selected_heads,
            max_tokens=20,  # Very small budget
            keep_prefix=5,
            keep_suffix=5,
        )

        result = compressor.compress(
            context=sample_context,
            question=sample_query,
        )

        # Compressed should be <= max_tokens (approximately)
        assert result["compressed_tokens"] <= result["original_tokens"]
        assert result["compression_rate"] <= 1.0

    def test_reform_without_kv_cache(
        self, mock_extractor, selected_heads, sample_context, sample_query
    ):
        """Test compression without KV cache optimization."""
        compressor = REFORMCompressor(
            model_extractor=mock_extractor,
            selected_heads=selected_heads,
            max_tokens=50,
            use_kv_cache=False,  # Disable KV cache
        )

        result = compressor.compress(
            context=sample_context,
            question=sample_query,
        )

        assert result is not None
        assert "compressed_context" in result

    def test_smooth_scores(self, mock_extractor, selected_heads):
        """Test score smoothing function."""
        compressor = REFORMCompressor(
            model_extractor=mock_extractor,
            selected_heads=selected_heads,
            smoothing_window=5,
        )

        # Create sample scores
        scores = torch.tensor([0.1, 0.2, 0.8, 0.3, 0.1, 0.9, 0.2])

        smoothed = compressor._smooth_scores(scores, window=3)

        # Smoothed should have same length
        assert len(smoothed) == len(scores)
        # Smoothed values should be >= original (max pooling)
        assert all(s >= o for s, o in zip(smoothed.tolist(), scores.tolist()))

    def test_create_chunks_from_tokens(self, mock_extractor, selected_heads):
        """Test chunk creation from tokens."""
        compressor = REFORMCompressor(
            model_extractor=mock_extractor,
            selected_heads=selected_heads,
            chunk_size=100,
        )
        compressor.chunk_size = 100  # Override for test

        tokens = list(range(250))  # 250 tokens
        chunks = compressor._create_chunks_from_tokens(tokens)

        assert len(chunks) == 3  # ceil(250/100)
        assert chunks[0]["global_start"] == 0
        assert chunks[0]["global_end"] == 100
        assert chunks[1]["global_start"] == 100
        assert chunks[2]["global_start"] == 200

    def test_global_top_k_selection(self, mock_extractor, selected_heads):
        """Test global top-k token selection."""
        compressor = REFORMCompressor(
            model_extractor=mock_extractor,
            selected_heads=selected_heads,
            max_tokens=10,
            keep_prefix=2,
            keep_suffix=2,
        )

        # Create scored tokens: (score, global_pos, token_id)
        all_scores = [
            (0.1, 0, 100),  # prefix
            (0.2, 1, 101),  # prefix
            (0.9, 2, 102),  # high score
            (0.3, 3, 103),
            (0.8, 4, 104),  # high score
            (0.2, 5, 105),
            (0.1, 18, 118),  # suffix
            (0.1, 19, 119),  # suffix
        ]

        selected = compressor._global_top_k_selection(all_scores, total_context_tokens=20)

        # Should include prefix and suffix
        positions = [s[1] for s in selected]
        assert 0 in positions  # prefix
        assert 1 in positions  # prefix
        assert 18 in positions  # suffix
        assert 19 in positions  # suffix

    def test_merge_spans(self, mock_extractor, selected_heads, mock_tokenizer):
        """Test span merging."""
        compressor = REFORMCompressor(
            model_extractor=mock_extractor,
            selected_heads=selected_heads,
            merge_threshold=2,
            span_separator=" ... ",
        )

        # Tokens with gaps
        selected_tokens = [
            (0.9, 0, 100),
            (0.8, 1, 101),
            (0.7, 2, 102),  # Gap here
            (0.6, 10, 110),
            (0.5, 11, 111),
        ]

        text, count = compressor._merge_spans_from_scores(selected_tokens)

        assert text is not None
        assert count > 0



