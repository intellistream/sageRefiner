# sageRefiner

**Intelligent Context Compression Algorithms for RAG Systems**

sageRefiner is a standalone Python library providing state-of-the-art context compression algorithms to reduce token usage while maintaining semantic quality in RAG (Retrieval-Augmented Generation) systems.

## Features

- **Multiple Compression Algorithms**
  - **LongRefiner**: Advanced selective compression using LLM-based importance scoring
  - **REFORM**: Efficient attention-based compression with KV cache optimization
  - **Provence**: Sentence-level context pruning using DeBERTa-based scoring
  
- **High Compression Ratios**: Achieve 2-10x compression while preserving key information
- **Flexible Configuration**: Easy-to-use YAML/dict-based configuration
- **Production Ready**: Battle-tested in the SAGE framework

## Installation

```bash
# From PyPI (coming soon)
pip install sage-refiner

# From source
pip install git+https://github.com/intellistream/sageRefiner.git

# Development mode
git clone https://github.com/intellistream/sageRefiner.git
cd sageRefiner
pip install -e .
```

## Quick Start

```python
from sage_refiner import LongRefinerCompressor, RefinerConfig

# Configure the refiner
config = RefinerConfig(
    algorithm="long_refiner",
    budget=2048,  # Target token count
    base_model_path="Qwen/Qwen2.5-3B-Instruct",
)

# Initialize compressor
compressor = LongRefinerCompressor(
    base_model_path=config.base_model_path,
    max_model_len=25000,
    gpu_memory_utilization=0.5,
)

# Compress documents
query = "What are the benefits of exercise?"
documents = [
    {"contents": "Exercise improves cardiovascular health..."},
    {"contents": "Regular physical activity boosts mental wellbeing..."},
    # ... more documents
]

result = compressor.compress(
    question=query,
    document_list=documents,
    budget=2048,
)

print(f"Original tokens: {result['original_tokens']}")
print(f"Compressed tokens: {result['compressed_tokens']}")
print(f"Compression ratio: {result['compression_rate']:.2f}")
print(f"\nCompressed content:\n{result['compressed_context']}")
```

## Algorithms

### LongRefiner

Based on selective compression with LLM-guided importance scoring. Best for:
- High-quality compression with minimal information loss
- Scenarios where semantic coherence is critical
- Budget-constrained LLM applications

**Key Parameters:**
- `budget`: Target token count
- `base_model_path`: HuggingFace model for compression
- `compression_ratio`: Compression aggressiveness (0.0-1.0)

### REFORM

Efficient attention-based compression using attention head analysis. Best for:
- Fast compression with lower compute requirements
- Batch processing scenarios
- When exact wording preservation is less critical

**Key Parameters:**
- `max_tokens`: Maximum tokens to keep
- `selected_heads`: Attention heads for scoring
- `use_kv_cache`: Enable KV cache optimization

### Provence

Sentence-level context pruning using DeBERTa-based relevance scoring. Best for:
- Document-level pruning in RAG pipelines
- When you need to filter out irrelevant documents
- Scenarios with many retrieved documents

**Key Parameters:**
- `threshold`: Relevance threshold (0-1) for filtering
- `reorder`: Whether to reorder by relevance score
- `top_k`: Number of top documents to keep

## Configuration

```python
config = RefinerConfig(
    algorithm="long_refiner",  # or "reform"
    budget=2048,
    base_model_path="Qwen/Qwen2.5-3B-Instruct",
    
    # LongRefiner specific
    compression_ratio=0.5,
    device="cuda",
)
```

## Architecture

sageRefiner is designed as a standalone library that can be integrated into any Python application:

```
Your Application
      ↓
sageRefiner (this library)
      ↓
[LongRefiner | Reform] → Compressed Context
      ↓
Your LLM Pipeline
```

## Integration with SAGE

This library is part of the [SAGE framework](https://github.com/intellistream/SAGE) ecosystem. For seamless integration with SAGE pipelines, use the `RefinerAdapter` in `sage-middleware`:

```python
# In SAGE environment
from sage.middleware.components.sage_refiner import RefinerAdapter

env.from_batch(...)
   .map(ChromaRetriever, retriever_config)
   .map(RefinerAdapter, refiner_config)  # Add compression step
   .map(QAPromptor, promptor_config)
   .sink(...)
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.30+

## Examples

See the [examples/](examples/) directory for complete examples:
- `basic_compression.py`: Simple compression workflow
- `algorithm_comparison.py`: Compare different algorithms
- `batch_processing.py`: Process multiple queries efficiently

## Performance

Benchmark on common RAG datasets (RTX 3090):

| Algorithm    | Compression Ratio | Latency (avg) | Quality Score |
|--------------|-------------------|---------------|---------------|
| LongRefiner  | 3.2x              | 0.8s          | 0.92          |
| Reform       | 2.5x              | 0.3s          | 0.87          |

## Citation

If you use sageRefiner in your research, please cite:

```bibtex
@software{sageRefiner2025,
  title = {sageRefiner: Intelligent Context Compression for RAG},
  author = {SAGE Team},
  year = {2025},
  url = {https://github.com/intellistream/sageRefiner}
}
```

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](https://github.com/intellistream/SAGE/blob/main/CONTRIBUTING.md) for guidelines.

## Links

- **Documentation**: https://sage-docs.example.com (coming soon)
- **SAGE Framework**: https://github.com/intellistream/SAGE
- **Issues**: https://github.com/intellistream/sageRefiner/issues
