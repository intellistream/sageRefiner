"""
EHPC Configuration
==================

定义 EHPC 压缩算法的配置参数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class EHPCConfig:
    """EHPC 压缩配置.

    Args:
        layer: 在哪一层执行 token selection (从0开始)。
            通常选择模型中间层或靠后的层，不同模型最优层不同:
            - Llama-3.1-8B: layer=14 或 layer=31
            - Mistral-Nemo: layer=19
            - Phi-3.5-mini: layer=19
        heads: Evaluator heads 列表。
            通过 needle probing 找到的最重要的 attention heads。
            如果为空列表，则使用所有 heads（退化为 baseline 行为）。
        topk: 压缩后保留的总 token 数。
            历史部分保留 topk - window_size 个 tokens，
            最后 window_size 个 tokens 永远保留。
        window_size: 最后保留不压缩的 token 数。
            这些 tokens 对应最近的上下文，不参与压缩选择。
            通常设为 16-64。
        pool: 池化类型，用于平滑 attention scores。
            - 'avg_pool': 平均池化，更平滑
            - 'max_pool': 最大池化，更尖锐
            - None: 不使用池化
        kernel_size: 池化核大小。
            较大的 kernel_size 会使选择更平滑，减少噪声。
    """

    layer: int = 14
    heads: list[int] = field(default_factory=lambda: [24, 3, 18, 7, 29, 2, 9, 1])
    topk: int = 2048
    window_size: int = 32
    pool: Literal["avg_pool", "max_pool"] | None = "avg_pool"
    kernel_size: int = 4

    def to_dict(self) -> dict:
        """转换为字典格式."""
        return {
            "layer": self.layer,
            "heads": self.heads,
            "topk": self.topk,
            "window_size": self.window_size,
            "pool": self.pool,
            "kernel_size": self.kernel_size,
        }

    @classmethod
    def from_dict(cls, config: dict) -> EHPCConfig:
        """从字典创建配置."""
        return cls(
            layer=config.get("layer", 14),
            heads=config.get("heads", [24, 3, 18, 7, 29, 2, 9, 1]),
            topk=config.get("topk", 2048),
            window_size=config.get("window_size", 32),
            pool=config.get("pool", "avg_pool"),
            kernel_size=config.get("kernel_size", 4),
        )

    def __post_init__(self):
        """参数验证."""
        if self.layer < 0:
            raise ValueError(f"layer must be >= 0, got {self.layer}")
        if self.topk <= 0:
            raise ValueError(f"topk must be > 0, got {self.topk}")
        if self.window_size < 0:
            raise ValueError(f"window_size must be >= 0, got {self.window_size}")
        if self.window_size >= self.topk:
            raise ValueError(f"window_size ({self.window_size}) must be < topk ({self.topk})")
        if self.kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, got {self.kernel_size}")
        if self.pool not in ("avg_pool", "max_pool", None):
            raise ValueError(f"pool must be 'avg_pool', 'max_pool', or None, got {self.pool}")


# ============================================================
# 预定义配置：不同模型的推荐参数
# 这些参数来自原始 EHPC 论文/代码中的 needle probe 实验结果
# ============================================================

# Llama 系列配置
LLAMA_31_8B_CONFIG = EHPCConfig(
    layer=14,  # 原代码中 select_layer_idx=14 for llama
    heads=[24, 3, 18, 7, 29, 2, 9, 1],  # 通过 needle probe 得到的最佳 heads
    topk=2048,
    window_size=32,
    pool="avg_pool",
    kernel_size=4,
)

# CodeLlama 配置 (与 Llama 类似，但 layer=31)
CODELLAMA_7B_CONFIG = EHPCConfig(
    layer=31,  # 原代码默认 select_layer_idx=31
    heads=[24, 3, 18, 7, 29, 2, 9, 1],
    topk=2048,
    window_size=32,
    pool="avg_pool",
    kernel_size=4,
)

# Mistral 系列配置
MISTRAL_NEMO_CONFIG = EHPCConfig(
    layer=19,  # 原代码 select_layer_idx=19 for mistral (out of 40 layers)
    heads=[18, 13, 21, 8],  # Mistral 的推荐 heads (需要通过 probe 验证)
    topk=2048,
    window_size=32,
    pool="avg_pool",
    kernel_size=4,
)

# Phi-3 系列配置
PHI3_MINI_CONFIG = EHPCConfig(
    layer=19,  # 原代码 select_layer_idx=19 for phi3 (out of 32 layers)
    heads=[18, 13, 21, 8],  # 需要通过 probe 验证
    topk=2048,
    window_size=32,
    pool="avg_pool",
    kernel_size=4,
)


# 模型名称到配置的映射
MODEL_CONFIGS: dict[str, EHPCConfig] = {
    # Llama 系列
    "meta-llama/Llama-3.1-8B-Instruct": LLAMA_31_8B_CONFIG,
    "meta-llama/Meta-Llama-3.1-8B-Instruct": LLAMA_31_8B_CONFIG,
    "llama-3.1-8b-instruct": LLAMA_31_8B_CONFIG,
    "llama3": LLAMA_31_8B_CONFIG,
    # CodeLlama
    "codellama-7b-hf": CODELLAMA_7B_CONFIG,
    "codellama": CODELLAMA_7B_CONFIG,
    # Mistral 系列
    "mistralai/Mistral-Nemo-Instruct-2407": MISTRAL_NEMO_CONFIG,
    "mistral-nemo-instruct-2407": MISTRAL_NEMO_CONFIG,
    "mistral": MISTRAL_NEMO_CONFIG,
    # Phi-3 系列
    "microsoft/Phi-3.5-mini-instruct": PHI3_MINI_CONFIG,
    "phi-3.5-mini-instruct": PHI3_MINI_CONFIG,
    "phi3": PHI3_MINI_CONFIG,
    "phi": PHI3_MINI_CONFIG,
}


def get_config_for_model(model_name: str, **overrides) -> EHPCConfig:
    """根据模型名称获取推荐的 EHPC 配置.

    会自动匹配模型名称（支持部分匹配），并返回对应的预定义配置。
    可以通过 overrides 参数覆盖默认值。

    Args:
        model_name: 模型名称或路径
        **overrides: 要覆盖的参数，如 topk=1024, window_size=64

    Returns:
        EHPCConfig 对象

    Example:
        >>> config = get_config_for_model("meta-llama/Llama-3.1-8B-Instruct")
        >>> config = get_config_for_model("mistral", topk=1024)
    """
    model_name_lower = model_name.lower()

    # 精确匹配
    if model_name in MODEL_CONFIGS:
        base_config = MODEL_CONFIGS[model_name]
    else:
        # 部分匹配
        matched_config = None
        for key, config in MODEL_CONFIGS.items():
            if key.lower() in model_name_lower or model_name_lower in key.lower():
                matched_config = config
                break

        if matched_config is None:
            # 默认使用 Llama 配置
            import logging

            logging.warning(
                f"No predefined config for model '{model_name}', "
                f"using default Llama config. Consider running needle probe."
            )
            matched_config = LLAMA_31_8B_CONFIG

        base_config = matched_config

    # 应用覆盖
    if overrides:
        config_dict = base_config.to_dict()
        config_dict.update(overrides)
        return EHPCConfig.from_dict(config_dict)

    return base_config
