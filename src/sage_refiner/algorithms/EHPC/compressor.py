"""
EHPC Refiner (Compressor)
==========================

EHPC (Efficient Prompt Compression with Evaluator Heads) 核心 Refiner 实现。

⚠️ 注意：这是 EMI (External Model Initialization) 模式的实现
    - Refiner Model: 用于压缩/token selection 的模型（可以是小模型）
    - Generator Model: 用于生成答案的模型（可以是不同的大模型）
    - 选头：在 Refiner 模型上进行，而非 Generator

实现了基于 Evaluator Heads 的 prompt 压缩算法:
    1. 用指定的 evaluator heads 在某一层对 prompt tokens 打分
    2. 选择 Top-K 重要的 tokens
    3. 保留最后 window_size 个 tokens 不压缩

压缩流程 (EMI Mode):
    Input: text (原始 prompt 文本), config (EHPC 配置)
    Output: compressed_text (压缩后的文本，可用于任何 generator)

    1. set_select_mode(refiner_model, True) - 启用选择模式
    2. set_config(refiner_model, config) - 注入配置到每层 attention
    3. reduce_layer(refiner_model, config.layer) - 只运行到指定层
    4. refiner_model(input_ids) - 触发 attention 内部的 token selection
    5. get_layer_context() - 从 attention.indices 提取选中的 tokens
    6. recover_layer + set_select_mode(False) - 恢复正常模式
    7. 返回压缩后的文本（而非 token ids，以支持不同的 generator）

参考实现: AttentionCompressor/my_utils/my_generation.py
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

from .config import EHPCConfig

logger = logging.getLogger(__name__)


class EHPCCompressor:
    """EHPC Refiner (用于压缩/token selection).

    ⚠️ EMI 模式说明:
        - 这个 Refiner 模型专门用于压缩，选出重要的 tokens
        - 压缩结果是文本，可以送给任何 Generator 模型生成答案
        - Refiner 和 Generator 可以是不同的模型（如 8B refiner + 70B generator）

    使用 Evaluator Heads 进行高效 prompt 压缩。

    Example (EMI Mode):
        >>> # Refiner 模型 (用于压缩)
        >>> refiner = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        >>> compressor = EHPCCompressor(refiner, tokenizer)
        >>>
        >>> # 压缩
        >>> config = EHPCConfig(layer=14, heads=[24, 3, 18, 7], topk=2048)
        >>> result = compressor.compress(text="Long context...", config=config)
        >>> compressed_text = result["compressed_text"]
        >>>
        >>> # 用 Generator 生成 (可以是不同的模型)
        >>> generator = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")
        >>> generator_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")
        >>> inputs = generator_tokenizer(compressed_text, return_tensors="pt")
        >>> outputs = generator.generate(**inputs)

    Args:
        model: Refiner 模型（已替换为 SelectAttention）。
        tokenizer: Refiner 模型对应的 tokenizer。
        device: 推理设备。
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda",
    ):
        """初始化 EHPC Refiner.

        Args:
            model: Refiner 预训练模型（需要已替换为 SelectAttention）。
            tokenizer: Refiner 模型对应的 tokenizer。
            device: 推理设备。
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._original_layers = None

        logger.info("EHPCCompressor (Refiner) initialized in EMI mode")

    # ========== 模型控制函数 ==========

    def set_select_mode(self, mode: bool) -> None:
        """设置所有 attention 层的选择模式.

        Args:
            mode: True 启用选择模式，False 正常模式。
        """
        decoder_layers = self.model.model.layers
        for layer in decoder_layers:
            layer.self_attn.select_mode = mode

    def set_config(self, config: EHPCConfig | dict) -> None:
        """将 EHPC 配置注入到所有 attention 层.

        Args:
            config: EHPC 配置对象或字典。
        """
        config_dict = config.to_dict() if isinstance(config, EHPCConfig) else config

        decoder_layers = self.model.model.layers
        for layer in decoder_layers:
            layer.self_attn.config = config_dict

    def set_topk(self, topk: int) -> None:
        """设置 topk 参数.

        Args:
            topk: 保留的 token 数量。
        """
        decoder_layers = self.model.model.layers
        for layer in decoder_layers:
            layer.self_attn.topk = topk

    def reduce_layer(self, layer_idx: int) -> torch.nn.ModuleList:
        """临时减少模型层数，只运行到指定层.

        用于获取指定层的 attention indices，无需运行完整模型。

        Args:
            layer_idx: 目标层索引（包含）。

        Returns:
            原始层列表（用于恢复）。
        """
        original_layers = self.model.model.layers
        self.model.model.layers = self.model.model.layers[: layer_idx + 1]
        self._original_layers = original_layers
        return original_layers

    def recover_layer(self, original_layers: torch.nn.ModuleList | None = None) -> None:
        """恢复模型的所有层.

        Args:
            original_layers: 原始层列表。如果为 None，使用保存的原始层。
        """
        if original_layers is not None:
            self.model.model.layers = original_layers
        elif self._original_layers is not None:
            self.model.model.layers = self._original_layers
            self._original_layers = None

    # ========== 核心压缩函数 ==========

    def get_layer_context(
        self,
        input_ids: torch.Tensor,
        layer_idx: int,
        print_context: bool = False,
    ) -> torch.Tensor:
        """从指定层提取选中的 token indices 并重建 input_ids.

        这是 EHPC 压缩的核心步骤：
        1. 从 attention 层获取 indices（被选中的历史 token 位置）
        2. 排序后从原始 input_ids gather 对应 tokens
        3. 拼接最后 window_size 个 tokens

        Args:
            input_ids: 原始 input_ids (1D tensor)。
            layer_idx: 从哪一层获取 indices。
            print_context: 是否打印压缩后的文本。

        Returns:
            压缩后的 input_ids (2D tensor: [1, compressed_len])。
        """
        # 获取配置
        config = self.model.model.layers[0].self_attn.config
        window_size = config.get("window_size", 16) if config else 16

        # 获取选中的 indices
        decoder_layer = self.model.model.layers[layer_idx]
        idx = decoder_layer.self_attn.indecies[0, 0, :]  # [k - window_size]

        # 排序 indices（保持原始顺序）
        values, _ = torch.sort(idx)
        values = values.to(self.device)

        # Gather 选中的 tokens
        if input_ids.dim() == 2:
            input_ids = input_ids[0]  # [seq_len]
        select_input_ids = input_ids.gather(0, values)

        # 拼接最后 window_size 个 tokens（不压缩部分）
        new_input_ids = torch.cat([select_input_ids, input_ids[-window_size:]])

        if print_context:
            logger.info(f"Compressed context: {self.tokenizer.decode(new_input_ids)}")

        return new_input_ids.unsqueeze(0)

    @torch.no_grad()
    def compress(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        config: EHPCConfig | dict | None = None,
        return_time: bool = False,
    ) -> dict[str, Any]:
        """执行 EHPC 压缩.

        完整的压缩流程：
        1. 设置选择模式并注入配置
        2. 只运行到指定层，触发 token selection
        3. 提取 indices 并重建压缩后的 input_ids
        4. 恢复模型状态

        Args:
            input_ids: 原始 input_ids [batch_size, seq_len] 或 [seq_len]。
            attention_mask: 注意力掩码（可选）。
            config: EHPC 配置。如果为 None，使用默认配置。
            return_time: 是否返回时间统计。

        Returns:
            压缩结果字典:
            {
                "compressed_input_ids": Tensor,  # 压缩后的 input_ids
                "original_length": int,           # 原始长度
                "compressed_length": int,         # 压缩后长度
                "compression_ratio": float,       # 压缩比
                "time_dict": dict (optional),     # 时间统计
            }
        """
        if config is None:
            config = EHPCConfig()

        if isinstance(config, dict):
            config = EHPCConfig.from_dict(config)

        config_dict = config.to_dict()

        # 确保 input_ids 是 2D
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        original_length = input_ids.shape[1]

        # 如果输入长度小于 topk，无需压缩
        if original_length <= config.topk:
            logger.info(f"Input length {original_length} <= topk {config.topk}, skip compression")
            return {
                "compressed_input_ids": input_ids,
                "original_length": original_length,
                "compressed_length": original_length,
                "compression_ratio": 1.0,
            }

        t1 = time.time()

        # Step 1: 设置选择模式并注入配置
        self.set_select_mode(True)
        self.set_config(config_dict)

        # Step 2: 只运行到指定层
        original_layers = self.reduce_layer(config.layer)

        # Step 3: 触发 token selection (prefill)
        # 这会触发 attention.forward() 中的 find_context()
        _ = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            use_cache=False,
        )

        t2 = time.time()

        # Step 4: 提取 indices 并重建 input_ids
        new_input_ids = self.get_layer_context(
            input_ids[0],
            config.layer,
            print_context=False,
        )

        t3 = time.time()

        # Step 5: 恢复模型状态
        self.recover_layer(original_layers)
        self.set_select_mode(False)

        compressed_length = new_input_ids.shape[1]

        result = {
            "compressed_input_ids": new_input_ids,
            "original_length": original_length,
            "compressed_length": compressed_length,
            "compression_ratio": compressed_length / original_length,
        }

        if return_time:
            result["time_dict"] = {
                "selection_time": t2 - t1,
                "reconstruction_time": t3 - t2,
                "total_compression_time": t3 - t1,
            }

        logger.info(
            f"EHPC compression: {original_length} -> {compressed_length} "
            f"(ratio: {result['compression_ratio']:.2%})"
        )

        return result

    def compress_text(
        self,
        context: str | list[str],
        question: str = "",
        config: EHPCConfig | dict | None = None,
        return_time: bool = False,
    ) -> dict[str, Any]:
        """使用文本输入/输出的压缩接口 (EMI 模式).

        适用于 External Model Initialization (EMI) 模式：
        - Refiner 模型：用于压缩 (本模型)
        - Generator 模型：用于生成 (可以是不同的模型)

        Args:
            context: 上下文文本或文本列表。
            question: 问题文本（可选）。
            config: EHPC 配置。
            return_time: 是否返回时间统计。

        Returns:
            压缩结果字典:
            {
                "compressed_context": str,       # 压缩后的文本
                "original_tokens": int,          # 原始 token 数
                "compressed_tokens": int,        # 压缩后 token 数
                "compression_rate": float,       # 压缩率
                "time_dict": dict (optional),    # 时间统计
            }

        Example (EMI mode):
            >>> # Refiner: 小模型用于压缩
            >>> refiner_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
            >>> refiner_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
            >>> refiner_model = load_ehpc_model(refiner_model)
            >>> compressor = EHPCCompressor(refiner_model, refiner_tokenizer)
            >>>
            >>> # 压缩上下文
            >>> result = compressor.compress_text(
            ...     context=["Doc1...", "Doc2...", "Doc3..."],
            ...     question="What is the answer?",
            ...     config=config,
            ... )
            >>>
            >>> # Generator: 大模型用于生成 (可以是不同的模型)
            >>> generator = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-70B")
            >>> generator_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B")
            >>>
            >>> # 用压缩后的文本构建 prompt 并生成
            >>> prompt = f"Context: {result['compressed_context']}\\n\\nQuestion: {question}\\nAnswer:"
            >>> inputs = generator_tokenizer(prompt, return_tensors="pt")
            >>> output = generator.generate(**inputs)
        """
        # 构建输入文本
        context_text = "\n\n".join(context) if isinstance(context, list) else context
        full_text = f"{context_text}\n\nQuestion: {question}" if question else context_text

        # Tokenize
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)

        # 执行压缩
        compress_result = self.compress(
            input_ids=input_ids,
            attention_mask=attention_mask,
            config=config,
            return_time=return_time,
        )

        # 解码压缩后的 tokens
        compressed_input_ids = compress_result["compressed_input_ids"]
        compressed_text = self.tokenizer.decode(compressed_input_ids[0], skip_special_tokens=True)

        result = {
            "compressed_context": compressed_text,
            "original_tokens": compress_result["original_length"],
            "compressed_tokens": compress_result["compressed_length"],
            "compression_rate": compress_result["compression_ratio"],
        }

        if return_time and "time_dict" in compress_result:
            result["time_dict"] = compress_result["time_dict"]

        return result
