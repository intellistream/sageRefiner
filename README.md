# sageRefiner

**Intelligent Context Compression Library for LLM Systems**

sageRefiner provides state-of-the-art context compression algorithms to reduce token usage while
maintaining semantic quality for Large Language Model applications.

## Features

- **8 Compression Algorithms**

  - **LongRefiner**: LLM-based selective compression with importance scoring
  - **REFORM**: Attention-based compression with KV cache optimization
  - **Provence**: Sentence-level pruning using DeBERTa reranker
  - **LLMLingua2**: Fast BERT-based token classification
  - **LongLLMLingua**: Question-aware perplexity-based compression
  - **RECOMP-Abstractive**: T5-based summarization
  - **RECOMP-Extractive**: BERT-based sentence selection
  - **EHPC**: Evaluator Heads based efficient compression

- **High Compression Ratios**: 2-10x compression while preserving key information

- **Flexible Configuration**: YAML/dict-based configuration

- **Production Ready**: Battle-tested in the SAGE framework

## Installation

```bash
# Basic installation
pip install isage-refiner

# With vLLM support (for LongRefiner)
pip install isage-refiner[vllm]

# Development mode
git clone https://github.com/intellistream/sageRefiner.git
cd sageRefiner
pip install -e .
```

## Quick Start

```python
from sage_refiner import LLMLingua2Compressor

# Initialize compressor
compressor = LLMLingua2Compressor()

# Compress context
result = compressor.compress(
    context="Your long document text here...",
    question="What is the main topic?",
    target_token=500,
)

print(f"Compression rate: {result['compression_rate']:.2%}")
print(f"Compressed: {result['compressed_context']}")
```

## Algorithms

| Algorithm         | Model             | Best For                          |
| ----------------- | ----------------- | --------------------------------- |
| **LongRefiner**   | Qwen/Llama + LoRA | High-quality semantic compression |
| **REFORM**        | Any Llama/Qwen    | Fast attention-based selection    |
| **Provence**      | DeBERTa           | Document-level filtering          |
| **LLMLingua2**    | BERT              | Speed-critical applications       |
| **LongLLMLingua** | GPT-2/Llama       | Long document scenarios           |
| **RECOMP-Abst**   | T5                | Summarization-style compression   |
| **RECOMP-Extr**   | BERT              | Sentence extraction               |
| **EHPC**          | Llama-8B          | Evaluator heads selection         |

## Configuration

```python
from sage_refiner import RefinerConfig

config = RefinerConfig(
    algorithm="llmlingua2",
    target_token=500,
    force_tokens=["important", "keyword"],
)
```

## Examples

```bash
# Basic compression
python examples/basic_compression.py

# Compare algorithms
python examples/algorithm_comparison.py
```

## Requirements

- Python 3.11+
- PyTorch 2.0+
- Transformers 4.43+

## Benchmarking

The `benchmarks` module provides a comprehensive evaluation framework for all compression
algorithms:

```bash
# Quick comparison of multiple algorithms
pip install isage-refiner[benchmark]
sage-refiner-bench compare \
    --algorithms baseline,longrefiner,reform,provence \
    --samples 100

# Detailed evaluation with budget sweep
sage-refiner-bench sweep \
    --algorithm longrefiner \
    --budgets 512,1024,2048,4096
```

For detailed benchmarking documentation, see [benchmarks/README.md](benchmarks/README.md) and
[benchmarks/STRUCTURE.md](benchmarks/STRUCTURE.md).

## Citation

```bibtex
@software{sageRefiner2025,
  title = {sageRefiner: Context Compression for LLM},
  author = {SAGE Team},
  year = {2025},
  url = {https://github.com/intellistream/sageRefiner}
}
```

## License

Apache License 2.0

## Links

- **SAGE Framework**: https://github.com/intellistream/SAGE
- **Issues**: https://github.com/intellistream/sageRefiner/issues

## Documentation & Development

- **[Development Guide](docs/README.md)** - Setup, workflow, and guidelines
- **[Pre-commit Hooks](docs/PRE_COMMIT.md)** - Setup and usage guide
- **[CI/CD Pipeline](docs/HOOKS_SETUP.md)** - Automated checks details

For quick setup:

```bash
bash utils/installation/quickstart.sh    # Full installation
bash utils/hooks/setup-hooks.sh          # Setup pre-commit hooks
```
