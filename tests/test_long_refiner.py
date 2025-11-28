"""
Tests for LongRefiner Algorithm
================================
"""

import pytest

from sageRefiner import LongRefiner, RefinerConfig


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
    config = RefinerConfig(algorithm="long_refiner", budget=512)
    assert config.algorithm.value == "long_refiner"
    assert config.budget == 512


def test_config_to_dict():
    """Test config serialization."""
    config = RefinerConfig(
        algorithm="long_refiner",
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
        algorithm="long_refiner",
        budget=256,
        base_model_path="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu",
    )
    refiner = LongRefiner(config.to_dict())
    refiner.initialize()
    assert refiner.is_initialized
    refiner.shutdown()


@pytest.mark.skip(reason="Requires model download and GPU/CPU resources")
def test_long_refiner_refine(sample_query, sample_docs):
    """Test LongRefiner compression."""
    config = RefinerConfig(
        algorithm="long_refiner",
        budget=128,
        base_model_path="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu",
    )
    refiner = LongRefiner(config.to_dict())
    refiner.initialize()

    result = refiner.refine(sample_query, sample_docs, budget=128)

    assert result is not None
    assert result.refined_content is not None
    assert isinstance(result.refined_content, list)
    assert result.metrics.original_tokens > 0
    assert result.metrics.refined_tokens > 0
    assert result.metrics.refined_tokens <= result.metrics.original_tokens

    refiner.shutdown()


def test_invalid_algorithm():
    """Test invalid algorithm raises error."""
    with pytest.raises(ValueError):
        RefinerConfig(algorithm="invalid_algorithm")


def test_invalid_budget():
    """Test invalid budget raises error."""
    with pytest.raises(ValueError):
        RefinerConfig(budget=-100)


def test_config_from_dict():
    """Test config creation from dict."""
    config_dict = {"algorithm": "long_refiner", "budget": 512, "compress_ratio": 0.5}
    config = RefinerConfig.from_dict(config_dict)
    assert config.algorithm.value == "long_refiner"
    assert config.budget == 512
    assert config.compress_ratio == 0.5
