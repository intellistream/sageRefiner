from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sage_refiner.algorithms.llmlingua2.operator import LLMLingua2RefinerOperator
from sage_refiner.algorithms.longllmlingua.operator import LongLLMLinguaRefinerOperator
from sage_refiner.algorithms.LongRefiner.operator import LongRefinerOperator
from sage_refiner.algorithms.provence.operator import ProvenceRefinerOperator
from sage_refiner.algorithms.recomp_abst.operator import RECOMPAbstractiveRefinerOperator
from sage_refiner.algorithms.recomp_extr.operator import RECOMPExtractiveRefinerOperator
from sage_refiner.algorithms.reform.operator import REFORMRefinerOperator


def _sample_input() -> dict:
    return {
        "query": "what is ai?",
        "retrieval_results": ["doc one", "doc two"],
    }


def test_recomp_abstractive_operator_fails_fast_on_compression_error() -> None:
    operator = RECOMPAbstractiveRefinerOperator.__new__(RECOMPAbstractiveRefinerOperator)
    operator.enabled = True
    operator.compressor = MagicMock()
    operator.compressor.compress.side_effect = ValueError("model error")

    with pytest.raises(RuntimeError, match="RECOMP Abstractive compression failed"):
        operator.execute(_sample_input())


def test_recomp_extractive_operator_fails_fast_on_compression_error() -> None:
    operator = RECOMPExtractiveRefinerOperator.__new__(RECOMPExtractiveRefinerOperator)
    operator.enabled = True
    operator.compressor = MagicMock()
    operator.compressor.compress.side_effect = ValueError("model error")

    with pytest.raises(RuntimeError, match="RECOMP Extractive compression failed"):
        operator.execute(_sample_input())


def test_reform_operator_fails_fast_on_compression_error() -> None:
    operator = REFORMRefinerOperator.__new__(REFORMRefinerOperator)
    operator.enabled = True
    operator._logger = logging.getLogger("test.reform.operator")
    operator.compressor = MagicMock()
    operator.compressor.compress.side_effect = ValueError("model error")

    with pytest.raises(RuntimeError, match="REFORM compression failed"):
        operator.execute(_sample_input())


def test_longllmlingua_operator_fails_fast_on_compression_error() -> None:
    operator = LongLLMLinguaRefinerOperator.__new__(LongLLMLinguaRefinerOperator)
    operator.enabled = True
    operator._logger = logging.getLogger("test.longllmlingua.operator")
    operator.cfg = {}
    operator._compressor = MagicMock()
    operator._compressor.compress.side_effect = ValueError("model error")

    with pytest.raises(RuntimeError, match="LongLLMLingua compression failed"):
        operator.execute(_sample_input())


def test_provence_operator_fails_fast_on_compression_error() -> None:
    operator = ProvenceRefinerOperator.__new__(ProvenceRefinerOperator)
    operator.enabled = True
    operator.compressor = MagicMock()
    operator.compressor.batch_compress.side_effect = ValueError("model error")

    with pytest.raises(RuntimeError, match="Provence compression failed"):
        operator.execute(_sample_input())


def test_llmlingua2_operator_fails_fast_when_compressor_missing() -> None:
    operator = LLMLingua2RefinerOperator.__new__(LLMLingua2RefinerOperator)
    operator.enabled = True
    operator.cfg = {}
    operator._compressor = None

    with pytest.raises(RuntimeError, match="LLMLingua2 compressor is not initialized"):
        operator.execute(_sample_input())


def test_llmlingua2_operator_fails_fast_on_compression_error() -> None:
    operator = LLMLingua2RefinerOperator.__new__(LLMLingua2RefinerOperator)
    operator.enabled = True
    operator.cfg = {}
    operator._compressor = MagicMock()
    operator._compressor.compress.side_effect = ValueError("model error")

    with pytest.raises(RuntimeError, match="LLMLingua2 compression failed"):
        operator.execute(_sample_input())


def test_longrefiner_operator_fails_fast_on_compression_error() -> None:
    operator = LongRefinerOperator.__new__(LongRefinerOperator)
    operator.enabled = True
    operator.cfg = {}
    operator.compressor = MagicMock()
    operator.compressor.compress.side_effect = ValueError("model error")

    with pytest.raises(RuntimeError, match="LongRefiner compression failed"):
        operator.execute(_sample_input())
