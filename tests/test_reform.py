"""
Tests for REFORM Algorithm
===========================
"""

import pytest

from sageRefiner import RefinerConfig, REFORMCompressor
from sageRefiner.config import RefinerAlgorithm


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


@pytest.mark.skip(reason="Requires model download and resources")
def test_reform_initialization():
    """Test REFORM initialization."""
    # REFORMCompressor requires model_extractor and selected_heads
    pass


@pytest.mark.skip(reason="Requires model download and resources")
def test_reform_compress(sample_query, sample_docs):
    """Test REFORM compression."""
    # REFORMCompressor.compress() requires model to be loaded
    pass
