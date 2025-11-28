"""
Tests for Reform Algorithm
===========================
"""

import pytest

from sageRefiner import RefinerConfig
from sageRefiner.algorithms.reform.compressor import ReformCompressor


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


def test_reform_config():
    """Test Reform configuration."""
    config = RefinerConfig(
        algorithm="reform",
        budget=256,
        base_model_path="test-model",
    )
    assert config.algorithm.value == "reform"
    assert config.budget == 256


@pytest.mark.skip(reason="Requires model download and resources")
def test_reform_initialization():
    """Test Reform initialization."""
    config = RefinerConfig(
        algorithm="reform",
        budget=256,
        base_model_path="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu",
    )
    refiner = ReformCompressor(config.to_dict())
    refiner.initialize()
    assert refiner.is_initialized
    refiner.shutdown()


@pytest.mark.skip(reason="Requires model download and resources")
def test_reform_refine(sample_query, sample_docs):
    """Test Reform compression."""
    config = RefinerConfig(
        algorithm="reform",
        budget=128,
        base_model_path="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu",
    )
    refiner = ReformCompressor(config.to_dict())
    refiner.initialize()

    result = refiner.refine(sample_query, sample_docs, budget=128)

    assert result is not None
    assert result.refined_content is not None
    assert isinstance(result.refined_content, list)

    refiner.shutdown()
