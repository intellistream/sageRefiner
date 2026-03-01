"""Regression tests for issue #17 core compression/extraction paths.

Coverage goals:
- LongLLMLingua compression boundary behavior (empty input, missing question).
- LongLLMLingua compression argument contract passed to llmlingua backend.
- RECOMP extractive core selection flow and empty-context boundary behavior.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from sage_refiner.algorithms.longllmlingua.compressor import (
    LongLLMLinguaCompressor,
)
from sage_refiner.algorithms.provence.compressor import ProvenceCompressor
from sage_refiner.algorithms.recomp_abst.compressor import RECOMPAbstractiveCompressor
from sage_refiner.algorithms.recomp_extr.compressor import RECOMPExtractiveCompressor


class _SimpleTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(text.split())))


def test_longllmlingua_compress_empty_context_returns_zero_stats() -> None:
    compressor = LongLLMLinguaCompressor(model_name="dummy", device="cpu")

    result = compressor.compress(context="", question="what is this?")

    assert result["compressed_prompt"] == ""
    assert result["origin_tokens"] == 0
    assert result["compressed_tokens"] == 0


def test_longllmlingua_compress_requires_non_empty_question() -> None:
    compressor = LongLLMLinguaCompressor(model_name="dummy", device="cpu")

    with pytest.raises(ValueError, match="requires a non-empty question"):
        compressor.compress(context=["doc-1"], question="")


def test_longllmlingua_compress_passes_expected_backend_contract() -> None:
    compressor = LongLLMLinguaCompressor(model_name="dummy", device="cpu")
    backend = MagicMock()
    backend.compress_prompt.return_value = {
        "compressed_prompt": "compressed",
        "origin_tokens": 100,
        "compressed_tokens": 55,
        "ratio": "1.8x",
        "rate": "55%",
        "saving": "n/a",
    }
    compressor._compressor = backend
    compressor._initialized = True

    result = compressor.compress(
        context=["doc-1", "doc-2"],
        question="query",
        rate=3.0,
        condition_compare=None,
    )

    assert result["compressed_prompt"] == "compressed"
    kwargs = backend.compress_prompt.call_args.kwargs
    assert kwargs["rank_method"] == "longllmlingua"
    assert kwargs["condition_compare"] is True
    assert kwargs["target_token"] == -1
    assert kwargs["rate"] == 1.0


def test_longllmlingua_compress_for_rag_uses_query_contract() -> None:
    compressor = LongLLMLinguaCompressor(model_name="dummy", device="cpu")
    backend = MagicMock()
    backend.compress_prompt.return_value = {
        "compressed_prompt": "compressed",
        "origin_tokens": 10,
        "compressed_tokens": 6,
        "ratio": "1.6x",
        "rate": "60%",
        "saving": "n/a",
    }
    compressor._compressor = backend
    compressor._initialized = True

    compressor.compress_for_rag(
        documents=["doc-a", "doc-b"],
        query="what is ai?",
        keep_top_k_docs=1,
    )

    kwargs = backend.compress_prompt.call_args.kwargs
    assert kwargs["force_context_number"] == 1
    assert kwargs["concate_question"] is False


def _build_recomp_extractive(top_k: int = 2, score_threshold: float = 0.0) -> RECOMPExtractiveCompressor:
    compressor = RECOMPExtractiveCompressor.__new__(RECOMPExtractiveCompressor)
    compressor.top_k = top_k
    compressor.score_threshold = score_threshold
    compressor.tokenizer = _SimpleTokenizer()
    return compressor


def test_recomp_extractive_select_sentences_uses_top_k_path_when_threshold_filters_all() -> None:
    compressor = _build_recomp_extractive(top_k=2, score_threshold=0.95)
    sentences = ["s1", "s2", "s3"]
    scores = torch.tensor([0.1, 0.8, 0.7])

    selected_sentences, selected_indices = compressor.select_sentences(sentences, scores)

    assert selected_indices == [1, 2]
    assert selected_sentences == ["s2", "s3"]


def test_recomp_extractive_compress_empty_context_returns_identity_stats() -> None:
    compressor = _build_recomp_extractive()

    result = compressor.compress(context="   ", question="q")

    assert result["compressed_context"] == ""
    assert result["original_tokens"] == 0
    assert result["compressed_tokens"] == 0
    assert result["num_selected_sentences"] == 0


def test_recomp_extractive_compress_runs_scoring_selection_pipeline() -> None:
    compressor = _build_recomp_extractive(top_k=2, score_threshold=0.0)
    compressor.score_sentences = MagicMock(
        return_value=(
            ["sentence one", "sentence two", "sentence three"],
            torch.tensor([0.1, 0.9, 0.7]),
        )
    )

    result = compressor.compress(context="sentence one sentence two sentence three", question="q")

    assert result["selected_indices"] == [1, 2]
    assert result["num_selected_sentences"] == 2
    assert result["total_sentences"] == 3
    assert result["compressed_context"] == "sentence two sentence three"


def test_recomp_abstractive_compress_fails_fast_when_generation_raises() -> None:
    compressor = RECOMPAbstractiveCompressor.__new__(RECOMPAbstractiveCompressor)
    compressor._count_tokens = MagicMock(return_value=10)
    compressor.generate_summary = MagicMock(side_effect=ValueError("generation error"))

    with pytest.raises(RuntimeError, match="summary generation failed"):
        compressor.compress(context="ctx", question="q")


def test_recomp_abstractive_compress_fails_fast_on_empty_summary() -> None:
    compressor = RECOMPAbstractiveCompressor.__new__(RECOMPAbstractiveCompressor)
    compressor._count_tokens = MagicMock(return_value=10)
    compressor.generate_summary = MagicMock(return_value="   ")

    with pytest.raises(ValueError, match="generated empty summary"):
        compressor.compress(context="ctx", question="q")


def test_provence_batch_compress_fails_fast_when_model_process_raises() -> None:
    compressor = ProvenceCompressor.__new__(ProvenceCompressor)
    compressor.model = MagicMock()
    compressor.model.process.side_effect = ValueError("process error")
    compressor.threshold = 0.1
    compressor.batch_size = 8
    compressor.always_select_title = True
    compressor.enable_warnings = True
    compressor.reorder = False
    compressor.top_k = 5
    compressor.tokenizer = _SimpleTokenizer()

    with pytest.raises(RuntimeError, match="Provence model processing failed"):
        compressor.batch_compress(question_list=["q"], context_list=[["doc1", "doc2"]])
