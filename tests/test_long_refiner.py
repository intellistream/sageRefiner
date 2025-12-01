"""
Tests for LongRefiner Algorithm
================================

包含：
1. 配置类测试（不需要模型）
2. Mock 测试（模拟模型行为，验证逻辑）
3. 集成测试（需要模型，标记为 skip）
"""

from unittest.mock import MagicMock, patch

import pytest

from sage_refiner import LongRefinerCompressor, RefinerConfig
from sage_refiner.config import RefinerAlgorithm


# =============================================================================
# Fixtures
# =============================================================================


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


@pytest.fixture
def sample_doc_dicts(sample_docs):
    """Sample documents as dict format."""
    return [{"contents": doc} for doc in sample_docs]


# =============================================================================
# 配置类测试（不需要模型）
# =============================================================================


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


def test_config_default_values():
    """Test config default values."""
    config = RefinerConfig()
    assert config.budget == 2048
    assert config.compression_ratio is None
    assert config.base_model_path == "Qwen/Qwen2.5-3B-Instruct"


def test_config_validation():
    """Test config accepts valid values (no validation raises)."""
    # Note: RefinerConfig is a dataclass and doesn't validate values
    # These configs should be created without error
    config1 = RefinerConfig(budget=100)
    assert config1.budget == 100

    config2 = RefinerConfig(compression_ratio=0.5)
    assert config2.compression_ratio == 0.5


def test_algorithm_enum():
    """Test RefinerAlgorithm enum."""
    assert RefinerAlgorithm.LONG_REFINER.value == "long_refiner"
    assert RefinerAlgorithm.SIMPLE.value == "simple"
    assert "long_refiner" in RefinerAlgorithm.list_available()


def test_algorithm_enum_from_string():
    """Test creating algorithm enum from string."""
    algo = RefinerAlgorithm("long_refiner")
    assert algo == RefinerAlgorithm.LONG_REFINER

    algo2 = RefinerAlgorithm("simple")
    assert algo2 == RefinerAlgorithm.SIMPLE


# =============================================================================
# Mock 测试（模拟模型行为）
# =============================================================================


class TestLongRefinerMocked:
    """LongRefiner mock tests - 不需要真实模型."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(100))  # 模拟 100 个 tokens
        tokenizer.decode.return_value = "Compressed text output"
        tokenizer.__call__ = MagicMock(return_value={"input_ids": list(range(100))})
        return tokenizer

    @pytest.fixture
    def mock_model(self):
        """Create a mock vLLM model."""
        model = MagicMock()

        # 模拟 generate 输出
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock()]
        mock_output.outputs[0].text = '{"selected_titles": ["section1"]}'
        mock_output.outputs[0].logprobs = {
            1: {0: MagicMock(logprob=-0.5), 1: MagicMock(logprob=-1.0)}
        }
        model.generate.return_value = [mock_output]

        return model

    @patch("sage_refiner.algorithms.LongRefiner.compressor.AutoTokenizer")
    @patch("vllm.LLM", autospec=True)
    @patch(
        "sage_refiner.algorithms.LongRefiner.compressor.AutoModelForSequenceClassification"
    )
    def test_compress_returns_expected_structure(
        self,
        mock_auto_model,
        mock_llm_class,
        mock_auto_tokenizer,
        sample_query,
        sample_doc_dicts,
        mock_tokenizer,
        mock_model,
    ):
        """Test that compress returns expected dict structure."""
        # Setup mocks
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_llm_class.return_value = mock_model

        mock_score_model = MagicMock()
        mock_score_model.cuda.return_value = mock_score_model
        mock_score_model.eval.return_value = mock_score_model
        mock_score_model.half.return_value = mock_score_model
        mock_auto_model.from_pretrained.return_value = mock_score_model

        # Create compressor with mocked dependencies
        with patch.object(LongRefinerCompressor, "_load_trained_model"):
            with patch.object(LongRefinerCompressor, "_load_score_model"):
                compressor = LongRefinerCompressor.__new__(LongRefinerCompressor)
                compressor.model = mock_model
                compressor.tokenizer = mock_tokenizer
                compressor.score_model = mock_score_model
                compressor.score_tokenizer = mock_tokenizer
                compressor.local_score_func = MagicMock(
                    return_value=[0.5] * 10
                )  # Mock scores
                compressor.step_to_config = {
                    "query_analysis": {
                        "prompt_template": MagicMock(get_prompt=lambda **k: "prompt"),
                        "sampling_params": MagicMock(),
                        "lora_request": MagicMock(),
                    },
                    "doc_structuring": {
                        "prompt_template": MagicMock(get_prompt=lambda **k: "prompt"),
                        "sampling_params": MagicMock(),
                        "lora_request": MagicMock(),
                    },
                    "global_selection": {
                        "prompt_template": MagicMock(get_prompt=lambda **k: "prompt"),
                        "sampling_params": MagicMock(),
                        "lora_request": MagicMock(),
                    },
                }

                # Mock internal methods
                compressor.run_query_analysis = MagicMock(
                    return_value=[{"Local": 0.6, "Global": 0.4}]
                )
                compressor.run_doc_structuring = MagicMock(
                    return_value=[
                        [
                            {
                                "title": "Doc1",
                                "abstract": ["Test abstract"],
                                "sections": {},
                            }
                        ]
                    ]
                )
                compressor.run_all_search = MagicMock(
                    return_value=[["Compressed content here"]]
                )

                # Test compress
                result = compressor.compress(
                    question=sample_query,
                    document_list=sample_doc_dicts,
                    budget=128,
                )

                # Verify result structure
                assert result is not None
                assert isinstance(result, dict)
                assert "compressed_context" in result
                assert "original_tokens" in result
                assert "compressed_tokens" in result
                assert "compression_rate" in result

    def test_batch_compress_multiple_queries(self, mock_tokenizer, mock_model):
        """Test batch_compress with multiple queries."""
        with patch.object(LongRefinerCompressor, "_load_trained_model"):
            with patch.object(LongRefinerCompressor, "_load_score_model"):
                compressor = LongRefinerCompressor.__new__(LongRefinerCompressor)
                compressor.model = mock_model
                compressor.tokenizer = mock_tokenizer

                # Mock all internal methods
                compressor.run_query_analysis = MagicMock(
                    return_value=[
                        {"Local": 0.6, "Global": 0.4},
                        {"Local": 0.7, "Global": 0.3},
                    ]
                )
                compressor.run_doc_structuring = MagicMock(
                    return_value=[
                        [{"title": "D1", "abstract": ["a1"], "sections": {}}],
                        [{"title": "D2", "abstract": ["a2"], "sections": {}}],
                    ]
                )
                compressor.run_all_search = MagicMock(
                    return_value=[["Content 1"], ["Content 2"]]
                )

                # Test
                questions = ["Q1?", "Q2?"]
                docs = [
                    [{"contents": "Doc1 content"}],
                    [{"contents": "Doc2 content"}],
                ]

                results = compressor.batch_compress(
                    question_list=questions,
                    document_list=docs,
                    budget=256,
                )

                assert len(results) == 2
                for r in results:
                    assert "compressed_context" in r


# =============================================================================
# 集成测试（需要真实模型，默认跳过）
# =============================================================================


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
