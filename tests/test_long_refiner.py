"""
Tests for LongRefiner Algorithm
================================
"""

import pytest

from sageRefiner import LongRefinerCompressor, RefinerConfig
from sageRefiner.config import RefinerAlgorithm


@pytest.fixture
def sample_docs():
    """Sample documents for testing."""
    return [
        "The quick brown fox jumps over the lazy dog. This is a simple test sentence.",
        "Machine learning is a subset of artificial intelligence focused on data-driven learning.",
        "Python is a high-level programming language known for its simplicity and readability.",
    ]


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return "What is machine learning?"


def test_config_creation():
    """Test RefinerConfig creation."""
    config = RefinerConfig(algorithm=RefinerAlgorithm.LONG_REFINER, budget=512)
    assert config.algorithm == RefinerAlgorithm.LONG_REFINER
    assert config.budget == 512


def test_config_to_dict():
    """Test config serialization."""
    config = RefinerConfig(
        algorithm=RefinerAlgorithm.LONG_REFINER,
        budget=256,
        base_model_path="test-model",
    )
    config_dict = config.to_dict()
    assert config_dict["algorithm"] == "long_refiner"
    assert config_dict["budget"] == 256
    assert config_dict["base_model_path"] == "test-model"


@pytest.mark.skip(reason="Requires model download and GPU/CPU resources")
def test_long_refiner_initialization():
    """Test LongRefiner initialization."""
    config = RefinerConfig(
        algorithm=RefinerAlgorithm.LONG_REFINER,
        budget=256,
        base_model_path="Qwen/Qwen2.5-0.5B-Instruct",
    )
    refiner = LongRefinerCompressor(config.to_dict())
    assert refiner is not None


@pytest.mark.skip(reason="Requires model download and GPU/CPU resources")
def test_long_refiner_compress(sample_query, sample_docs):
    """Test LongRefiner compression."""
    config = RefinerConfig(
        algorithm=RefinerAlgorithm.LONG_REFINER,
        budget=128,
        base_model_path="Qwen/Qwen2.5-0.5B-Instruct",
    )
    refiner = LongRefinerCompressor(config.to_dict())

    result = refiner.compress(
        question=sample_query,
        document_list=[{"contents": doc} for doc in sample_docs],
        budget=128,
    )

    assert result is not None
    assert "compressed_context" in result


def test_config_from_dict():
    """Test config creation from dict."""
    config_dict = {
        "algorithm": "long_refiner",
        "budget": 512,
        "compression_ratio": 0.5,
    }
    config = RefinerConfig.from_dict(config_dict)
    assert config.algorithm == RefinerAlgorithm.LONG_REFINER
    assert config.budget == 512
    assert config.compression_ratio == 0.5


def test_algorithm_enum():
    """Test RefinerAlgorithm enum."""
    assert RefinerAlgorithm.LONG_REFINER.value == "long_refiner"
    assert RefinerAlgorithm.SIMPLE.value == "simple"
    assert "long_refiner" in RefinerAlgorithm.list_available()
