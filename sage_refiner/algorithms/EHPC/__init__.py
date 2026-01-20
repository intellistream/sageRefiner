"""
EHPC (Efficient Prompt Compression with Evaluator Heads)
========================================================

⚠️ EMI (External Model Initialization) 模式实现

EHPC 是一种基于 Evaluator Heads 的高效 Prompt 压缩算法。

EMI 模式说明:
    - **Refiner Model**: 专门用于压缩/token selection（可以是较小的模型）
    - **Generator Model**: 专门用于生成答案（可以是更大的模型）
    - **分离优势**: Refiner 和 Generator 可以是不同的模型，各取所长

核心思想:
    1. 使用预设的 Evaluator Heads（已通过原论文实验确定）对 prompt tokens 打分
    2. 选择 Top-K 重要的 tokens 来压缩 prompt
    3. 将压缩后的文本送给 **Generator 模型**生成答案

关键特性:
    - Evaluator Heads: 预设的最佳 attention heads（来自原论文实验）
    - Head-restricted Selection: 只使用指定 heads 的注意力分数来选择 tokens
    - Window Preservation: 最后 window_size 个 tokens 永远不压缩，保留近期上下文
    - EMI 模式: Refiner ≠ Generator，支持不同规模的模型组合

算法流程 (EMI):
    1. Compression (用 Refiner): 运行 Refiner 到指定层，触发 token selection
    2. Index Extraction: 从 attention layer 提取被选中的 token indices
    3. Text Reconstruction: 重建压缩后的文本
    4. Generation (用 Generator): 将压缩文本送给 Generator 生成答案

示例 (EMI 模式):
    >>> from sage_refiner.algorithms.EHPC import EHPCCompressor, EHPCConfig, load_ehpc_model
    >>>
    >>> # Step 1: 加载 Refiner 模型
    >>> model, tokenizer = load_ehpc_model("meta-llama/Llama-3.1-8B-Instruct")
    >>> compressor = EHPCCompressor(model, tokenizer)
    >>>
    >>> # Step 2: 压缩（使用预设配置）
    >>> config = EHPCConfig(topk=2048)  # 使用 Llama-3.1-8B 的默认 layer/heads
    >>> result = compressor.compress_text(long_context, question="What is...?", config=config)
    >>> compressed_text = result["compressed_context"]
    >>>
    >>> # Step 3: 用 Generator 生成 (可以是不同的模型)
    >>> generator = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")
    >>> generator_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")
    >>> inputs = generator_tokenizer(compressed_text, return_tensors="pt")
    >>> outputs = generator.generate(**inputs)

预设配置 (来自原论文实验):
    - Llama-3.1-8B: layer=14, heads=[24, 3, 18, 7, 29, 2, 9, 1]
    - CodeLlama-7B: layer=31, heads=[24, 3, 18, 7, 29, 2, 9, 1]
    - Mistral-Nemo: layer=19, heads=[18, 13, 21, 8]
    - Phi-3.5-mini: layer=19, heads=[18, 13, 21, 8]

Exports:
    - EHPCCompressor: Refiner 压缩器类（EMI 模式）
    - EHPCConfig: 配置类
    - EHPCOperator: SAGE Pipeline 操作符（可选，需要 SAGE）
    - load_ehpc_model: 模型加载函数
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .compressor import EHPCCompressor
from .config import (
    CODELLAMA_7B_CONFIG,
    LLAMA_31_8B_CONFIG,
    MISTRAL_NEMO_CONFIG,
    MODEL_CONFIGS,
    PHI3_MINI_CONFIG,
    EHPCConfig,
    get_config_for_model,
)
from .models.gemfilter.model_loader import load_ehpc_model

__all__ = [
    "EHPCCompressor",
    "EHPCConfig",
    "load_ehpc_model",
    "get_config_for_model",
    "MODEL_CONFIGS",
    "LLAMA_31_8B_CONFIG",
    "CODELLAMA_7B_CONFIG",
    "MISTRAL_NEMO_CONFIG",
    "PHI3_MINI_CONFIG",
]

# Type hint for optional import
if TYPE_CHECKING:
    from .operator import EHPCRefinerOperator as _EHPCRefinerOperator

# Operator 需要 SAGE 依赖，可选导出
EHPCRefinerOperator: type[_EHPCRefinerOperator] | None
try:
    from .operator import EHPCRefinerOperator

    __all__.append("EHPCRefinerOperator")
except ImportError:
    EHPCRefinerOperator = None
