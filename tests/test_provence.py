"""
Tests for Provence Algorithm
=============================
"""

from unittest.mock import MagicMock, patch

import pytest

from sage_refiner import ProvenceCompressor, RefinerConfig
from sage_refiner.config import RefinerAlgorithm


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


def test_provence_compressor_class():
    """Test ProvenceCompressor class is available."""
    assert ProvenceCompressor is not None


def test_provence_config():
    """Test config for Provence-like usage."""
    config = RefinerConfig(
        algorithm=RefinerAlgorithm.LONG_REFINER,
        budget=256,
        base_model_path="test-model",
    )
    assert config.algorithm == RefinerAlgorithm.LONG_REFINER
    assert config.budget == 256


def test_provence_algorithm_exists():
    """Test that PROVENCE could be added to RefinerAlgorithm."""
    # Provence is not in enum yet, but we can test the class exists
    assert hasattr(ProvenceCompressor, "compress")
    assert hasattr(ProvenceCompressor, "batch_compress")


class TestProvenceMocked:
    """Test Provence with mocked model."""

    @patch("sage_refiner.algorithms.provence.compressor.AutoModel")
    @patch("sage_refiner.algorithms.provence.compressor.AutoTokenizer")
    def test_provence_init(self, mock_auto_tokenizer, mock_auto_model):
        """Test Provence initialization with mocked model."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda x: list(range(len(x.split())))
        # Need to mock .to() chain: from_pretrained().to() returns mock_model
        mock_auto_model.from_pretrained.return_value.to.return_value = mock_model
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        compressor = ProvenceCompressor(
            model_name="test-model",
            threshold=0.1,
            device="cpu",
        )

        assert compressor.threshold == 0.1
        assert compressor.device.type == "cpu"
        mock_auto_model.from_pretrained.assert_called_once()

    @patch("sage_refiner.algorithms.provence.compressor.AutoModel")
    @patch("sage_refiner.algorithms.provence.compressor.AutoTokenizer")
    def test_provence_compress_returns_dict(self, mock_auto_tokenizer, mock_auto_model):
        """Test that compress returns expected dict structure."""
        mock_model = MagicMock()
        mock_model.process.return_value = {
            "pruned_context": [
                ["AI transforms tech.", "ML processes data."],
            ],
            "reranking_score": [[0.9, 0.7]],
        }
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda x: list(range(len(x.split())))
        # Need to mock .to() chain
        mock_auto_model.from_pretrained.return_value.to.return_value = mock_model
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        compressor = ProvenceCompressor(model_name="test-model", device="cpu")

        result = compressor.compress(
            context="Doc 1 content.\n\nDoc 2 content.",
            question="What is this?",
        )

        assert isinstance(result, dict)
        assert "compressed_context" in result
        assert "original_tokens" in result
        assert "compressed_tokens" in result
        assert "compression_rate" in result
        assert "pruned_docs" in result

    @patch("sage_refiner.algorithms.provence.compressor.AutoModel")
    @patch("sage_refiner.algorithms.provence.compressor.AutoTokenizer")
    def test_provence_batch_compress(self, mock_auto_tokenizer, mock_auto_model):
        """Test batch compression."""
        mock_model = MagicMock()
        mock_model.process.return_value = {
            "pruned_context": [
                ["Pruned doc 1."],
                ["Pruned doc 2."],
            ],
            "reranking_score": [[0.9], [0.8]],
        }
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda x: list(range(len(x.split())))
        # Need to mock .to() chain
        mock_auto_model.from_pretrained.return_value.to.return_value = mock_model
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        compressor = ProvenceCompressor(model_name="test-model", device="cpu")

        results = compressor.batch_compress(
            question_list=["Question 1?", "Question 2?"],
            context_list=[["Doc 1."], ["Doc 2."]],
        )

        assert len(results) == 2
        assert all(isinstance(r, dict) for r in results)
        assert all("compressed_context" in r for r in results)

    @patch("sage_refiner.algorithms.provence.compressor.AutoModel")
    @patch("sage_refiner.algorithms.provence.compressor.AutoTokenizer")
    def test_provence_with_reorder(self, mock_auto_tokenizer, mock_auto_model):
        """Test compression with reordering enabled."""
        mock_model = MagicMock()
        mock_model.process.return_value = {
            "pruned_context": [
                ["Low score doc.", "High score doc.", "Medium score doc."],
            ],
            "reranking_score": [[0.3, 0.9, 0.6]],
        }
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda x: list(range(len(x.split())))
        # Need to mock .to() chain
        mock_auto_model.from_pretrained.return_value.to.return_value = mock_model
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        compressor = ProvenceCompressor(
            model_name="test-model",
            device="cpu",
            reorder=True,
            top_k=2,
        )

        result = compressor.compress(
            context="Doc 1.\n\nDoc 2.\n\nDoc 3.",
            question="Test question?",
        )

        # With reorder=True and top_k=2, should only have 2 docs
        assert len(result["pruned_docs"]) <= 2


@pytest.mark.skip(reason="Requires model download and resources")
def test_provence_initialization():
    """Test Provence initialization with real model."""
    compressor = ProvenceCompressor(
        model_name="naver/provence-reranker-debertav3-v1",
        threshold=0.1,
        device="cpu",
    )
    assert compressor is not None


@pytest.mark.skip(reason="Requires model download and resources")
def test_provence_compress(sample_query, sample_docs):
    """Test Provence compression with real model."""
    compressor = ProvenceCompressor(
        model_name="naver/provence-reranker-debertav3-v1",
        threshold=0.1,
        device="cpu",
    )

    result = compressor.compress(
        context="\n\n".join(sample_docs),
        question=sample_query,
    )

    assert result is not None
    assert "compressed_context" in result
    assert "compression_rate" in result
