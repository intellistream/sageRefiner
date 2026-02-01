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
    # Support both semantic versioning (Major.Minor.Patch) and 4-part versions (Major.Minor.Patch.Build)
    parts = __version__.split(".")
    assert len(parts) in (3, 4), f"Version should have 3 or 4 parts, got {len(parts)}"
    assert all(part.isdigit() for part in parts), "All version parts should be numeric"
