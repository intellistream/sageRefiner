# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""
Unit tests for LongLLMLingua integration.

Tests cover:
- LongLLMLinguaCompressor import and structure
- LongLLMLinguaRefinerOperator import and structure
- Default configuration validation
- Mock compression functionality
"""

from __future__ import annotations

import pytest


class TestLongLLMLinguaImports:
    """Tests for LongLLMLingua module imports."""

    def test_import_from_algorithms(self) -> None:
        """Test importing from algorithms submodule."""
        from sage_refiner.algorithms.longllmlingua import (
            DEFAULT_LONG_LLMLINGUA_CONFIG,
            LongLLMLinguaCompressor,
            LongLLMLinguaRefinerOperator,
        )

        assert LongLLMLinguaCompressor is not None
        assert LongLLMLinguaRefinerOperator is not None
        assert DEFAULT_LONG_LLMLINGUA_CONFIG is not None

    def test_import_from_main_module(self) -> None:
        """Test importing from main sage_refiner module."""
        from sage_refiner import (
            DEFAULT_LONG_LLMLINGUA_CONFIG,
            LongLLMLinguaCompressor,
            LongLLMLinguaRefinerOperator,
        )

        assert LongLLMLinguaCompressor is not None
        assert LongLLMLinguaRefinerOperator is not None
        assert DEFAULT_LONG_LLMLINGUA_CONFIG is not None


class TestLongLLMLinguaConfig:
    """Tests for LongLLMLingua default configuration."""

    def test_default_config_values(self) -> None:
        """Test that default config matches paper baseline."""
        from sage_refiner.algorithms.longllmlingua import (
            DEFAULT_LONG_LLMLINGUA_CONFIG,
        )

        # Paper baseline: rate=0.55, condition_compare=True
        assert DEFAULT_LONG_LLMLINGUA_CONFIG["rate"] == 0.55
        assert DEFAULT_LONG_LLMLINGUA_CONFIG["condition_compare"] is True
        assert DEFAULT_LONG_LLMLINGUA_CONFIG["condition_in_question"] == "after"
        assert DEFAULT_LONG_LLMLINGUA_CONFIG["reorder_context"] == "sort"

    def test_default_config_has_required_keys(self) -> None:
        """Test that default config has all required keys."""
        from sage_refiner.algorithms.longllmlingua import (
            DEFAULT_LONG_LLMLINGUA_CONFIG,
        )

        required_keys = [
            "rate",
            "condition_in_question",
            "reorder_context",
            "dynamic_context_compression_ratio",
            "condition_compare",
        ]

        for key in required_keys:
            assert key in DEFAULT_LONG_LLMLINGUA_CONFIG, f"Missing key: {key}"


class TestLongLLMLinguaCompressor:
    """Tests for LongLLMLinguaCompressor class."""

    def test_compressor_class_exists(self) -> None:
        """Test that LongLLMLinguaCompressor class exists and has expected attributes."""
        from sage_refiner.algorithms.longllmlingua import (
            LongLLMLinguaCompressor,
        )

        # Check class has compress method
        assert hasattr(LongLLMLinguaCompressor, "compress")

    def test_compressor_default_model(self) -> None:
        """Test that DEFAULT_MODEL is set correctly."""
        from sage_refiner.algorithms.longllmlingua import (
            LongLLMLinguaCompressor,
        )

        assert LongLLMLinguaCompressor.DEFAULT_MODEL == "NousResearch/Llama-2-7b-hf"

    def test_compressor_initialization_no_model_load(self) -> None:
        """Test compressor initialization without loading model (lazy init)."""
        from sage_refiner.algorithms.longllmlingua import (
            LongLLMLinguaCompressor,
        )

        # Create compressor - should not load model yet (lazy initialization)
        compressor = LongLLMLinguaCompressor(
            model_name="test-model",
            device="cpu",
        )

        # Verify lazy initialization - model should not be loaded yet
        assert compressor._initialized is False
        assert compressor._compressor is None
        assert compressor.model_name == "test-model"
        assert compressor.device == "cpu"


class TestLongLLMLinguaRefinerOperator:
    """Tests for LongLLMLinguaRefinerOperator class."""

    def test_operator_class_exists(self) -> None:
        """Test that LongLLMLinguaRefinerOperator class exists."""
        from sage_refiner.algorithms.longllmlingua import (
            LongLLMLinguaRefinerOperator,
        )

        assert LongLLMLinguaRefinerOperator is not None

    def test_operator_has_execute_method(self) -> None:
        """Test that LongLLMLinguaRefinerOperator has execute method (SAGE MapOperator interface)."""
        from sage_refiner.algorithms.longllmlingua import (
            LongLLMLinguaRefinerOperator,
        )

        # Check it has the execute method (SAGE MapOperator interface)
        assert hasattr(LongLLMLinguaRefinerOperator, "execute")


class TestLongLLMLinguaInRefinerAlgorithm:
    """Tests for LongLLMLingua in RefinerAlgorithm enum."""

    def test_longllmlingua_in_enum(self) -> None:
        """Test that LONGLLMLINGUA is in RefinerAlgorithm enum."""
        from benchmarks.experiments import RefinerAlgorithm

        assert hasattr(RefinerAlgorithm, "LONGLLMLINGUA")
        assert RefinerAlgorithm.LONGLLMLINGUA.value == "longllmlingua"

    def test_longllmlingua_in_available(self) -> None:
        """Test that longllmlingua is in available algorithms."""
        from benchmarks.experiments import RefinerAlgorithm

        available = RefinerAlgorithm.available()
        assert "longllmlingua" in available


class TestLongLLMLinguaPipelineStructure:
    """Tests for LongLLMLingua pipeline file structure."""

    def test_pipeline_file_exists(self) -> None:
        """Test that pipeline file exists."""
        # Use relative import to test module can be found (not necessarily imported without deps)
        try:
            import importlib.util

            spec = importlib.util.find_spec(
                "benchmarks.implementations.pipelines.longllmlingua_rag"
            )
            assert spec is not None
        except ModuleNotFoundError:
            pytest.skip("Pipeline module not found in path")

    def test_config_file_exists(self) -> None:
        """Test that config file exists."""
        from pathlib import Path

        config_path = (
            Path(__file__).parent.parent
            / "benchmarks"
            / "config"
            / "config_longllmlingua.yaml"
        )

        # Check relative to benchmarks directory
        assert config_path.exists(), f"Config file not found at {config_path}"
