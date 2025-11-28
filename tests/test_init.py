"""Test package initialization."""

from sageRefiner import (
    LongRefiner,
    RefinerAlgorithm,
    RefinerConfig,
    ReformCompressor,
    __version__,
)


def test_imports():
    """Test that all main imports are available."""
    assert RefinerConfig is not None
    assert RefinerAlgorithm is not None
    assert LongRefiner is not None
    assert ReformCompressor is not None
    assert __version__ is not None


def test_version():
    """Test version string format."""
    assert isinstance(__version__, str)
    assert len(__version__.split(".")) == 3  # Major.Minor.Patch
