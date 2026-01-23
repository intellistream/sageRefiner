"""Test package initialization."""

from sage_refiner import (
    LongRefiner,
    LongRefinerCompressor,
    ProvenceCompressor,
    RefinerAlgorithm,
    RefinerConfig,
    __version__,
)


def test_imports():
    """Test that all main imports are available."""
    assert RefinerConfig is not None
    assert RefinerAlgorithm is not None
    assert LongRefinerCompressor is not None
    assert ProvenceCompressor is not None
    assert __version__ is not None


def test_aliases():
    """Test that aliases work correctly."""
    # LongRefiner is alias for LongRefinerCompressor
    assert LongRefiner is LongRefinerCompressor


def test_version():
    """Test version string format."""
    assert isinstance(__version__, str)
    assert len(__version__.split(".")) == 3  # Major.Minor.Patch
