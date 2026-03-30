"""
EHPC Needle Probe
=================

用于找到最佳 Evaluator Heads 的 Needle-in-a-Haystack 探测工具。

通过在不同深度插入 needle，观察每个 attention head 对 needle 区域的注意力，
从而找到能准确定位关键信息的 Evaluator Heads。

使用方法:
    >>> from sage_refiner.algorithms.EHPC.probe import NeedleProbe
    >>> probe = NeedleProbe(model, tokenizer)
    >>> best_layer, best_heads = probe.find_evaluator_heads(
    ...     context_length=16000,
    ...     depths=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    ... )
"""

from __future__ import annotations

import glob
import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


class NeedleProbe:
    """Needle-in-a-Haystack Probe for finding Evaluator Heads.

    通过 needle probing 找到对关键信息最敏感的 attention heads。

    Args:
        model: 已替换为 SelectAttention 的模型
        tokenizer: 对应的 tokenizer
        device: 推理设备
    """

    DEFAULT_NEEDLE = (
        "\nThe best thing to do in San Francisco is eat a sandwich "
        "and sit in Dolores Park on a sunny day.\n"
    )

    DEFAULT_QUESTION = "What is the best thing to do in San Francisco?"

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def load_context(
        self,
        fpath: str = "*.txt",
        ctx_len: int = 16000,
    ) -> str:
        """加载背景文本 (haystack).

        Args:
            fpath: 文本文件路径 pattern (支持 glob)
            ctx_len: 目标 token 长度

        Returns:
            截断后的背景文本
        """
        context = ""
        files = glob.glob(fpath)

        if not files:
            # 使用默认占位文本
            logger.warning(f"No files found for {fpath}, using placeholder text")
            context = "This is a placeholder text. " * 10000
        else:
            for file in sorted(files):
                with open(file, encoding="utf-8", errors="ignore") as f:
                    context += f.read()

        # 截断到目标长度
        tokens = self.tokenizer.encode(context, add_special_tokens=False)
        if len(tokens) > ctx_len:
            tokens = tokens[:ctx_len]
            context = self.tokenizer.decode(tokens)

        return context

    def insert_needle(
        self,
        context: str,
        needle: str,
        depth: float = 0.5,
    ) -> str:
        """在指定深度插入 needle.

        Args:
            context: 背景文本
            needle: 要插入的 needle
            depth: 插入深度 (0.0 = 开头, 1.0 = 结尾)

        Returns:
            插入 needle 后的文本
        """
        # 找到插入位置
        tokens = self.tokenizer.encode(context, add_special_tokens=False)
        insert_pos = int(len(tokens) * depth)

        # 在句子边界插入
        # 简化处理：直接在 token 位置插入
        context_before = self.tokenizer.decode(tokens[:insert_pos])
        context_after = self.tokenizer.decode(tokens[insert_pos:])

        return context_before + needle + context_after

    def set_select_mode(self, mode: bool) -> None:
        """设置模型的选择模式."""
        decoder_layers = self.model.model.layers
        for layer in decoder_layers:
            layer.self_attn.select_mode = mode

    def set_probe_context(self, probe_context: tuple[int, int] | None) -> None:
        """设置 probe 上下文范围."""
        decoder_layers = self.model.model.layers
        for layer in decoder_layers:
            layer.self_attn.probe_context = probe_context

    def reduce_layer(self, layer_idx: int) -> torch.nn.ModuleList:
        """临时减少模型层数."""
        original_layers = self.model.model.layers
        self.model.model.layers = self.model.model.layers[: layer_idx + 1]
        return original_layers

    def recover_layer(self, original_layers: torch.nn.ModuleList) -> None:
        """恢复模型层数."""
        self.model.model.layers = original_layers

    def get_attention_scores(self, layer_idx: int) -> list[torch.Tensor]:
        """获取所有层的 attention scores (对 needle 区域)."""
        scores = []
        num_layers = len(self.model.model.layers)

        for i in range(min(layer_idx + 1, num_layers)):
            layer = self.model.model.layers[i]
            if hasattr(layer.self_attn, "attention_socres"):
                score = layer.self_attn.attention_socres
                if score is not None:
                    scores.append(score)

        return scores

    @torch.no_grad()
    def probe_single_depth(
        self,
        context: str,
        needle: str,
        depth: float,
        question: str,
        select_layer_idx: int = 31,
    ) -> torch.Tensor:
        """在单个深度执行 probe.

        Args:
            context: 背景文本
            needle: needle 文本
            depth: 插入深度
            question: 问题
            select_layer_idx: 执行 probe 的层

        Returns:
            attention scores tensor [num_layers, num_heads]
        """
        # 插入 needle
        context_with_needle = self.insert_needle(context, needle, depth)

        # 构建 prompt
        prompt = f"\n<|im_start|> This is a very long story book: <book> {context_with_needle} </book>.\n"
        prompt += f"Based on the content of the book, Question: {question}\nAnswer:"

        # 找到 needle 的 token 位置
        needle_idx = prompt.find(needle)
        prefix = prompt[:needle_idx]
        begin_token_id = len(self.tokenizer(prefix).input_ids)
        needle_token_len = len(self.tokenizer(needle).input_ids) - 2

        # Tokenize
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # 设置 probe context
        probe_context = (begin_token_id, needle_token_len)
        self.set_select_mode(True)
        self.set_probe_context(probe_context)

        # 只运行到指定层
        original_layers = self.reduce_layer(select_layer_idx)

        # 运行模型
        _ = self.model(input_ids, attention_mask=attention_mask)

        # 收集 attention scores
        scores = self.get_attention_scores(select_layer_idx)

        # 恢复
        self.recover_layer(original_layers)
        self.set_select_mode(False)
        self.set_probe_context(None)

        if scores:
            return torch.stack(scores, dim=0)  # [num_layers, num_heads]
        return torch.zeros(1)

    def find_evaluator_heads(
        self,
        context_path: str = "*.txt",
        context_length: int = 16000,
        needle: str | None = None,
        question: str | None = None,
        depths: list[float] | None = None,
        select_layer_idx: int = 31,
        num_heads_to_select: int = 8,
    ) -> tuple[int, list[int]]:
        """找到最佳的 Evaluator Heads.

        通过在不同深度插入 needle 并观察 attention scores，
        找到能准确定位关键信息的层和 heads。

        Args:
            context_path: 背景文本文件路径
            context_length: 上下文 token 长度
            needle: needle 文本 (默认使用 DEFAULT_NEEDLE)
            question: 问题 (默认使用 DEFAULT_QUESTION)
            depths: 要测试的深度列表
            select_layer_idx: 测试的层索引
            num_heads_to_select: 选择的 head 数量

        Returns:
            Tuple of (best_layer_idx, best_head_indices)
        """
        if needle is None:
            needle = self.DEFAULT_NEEDLE
        if question is None:
            question = self.DEFAULT_QUESTION
        if depths is None:
            depths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # 加载背景文本
        context = self.load_context(context_path, context_length)

        # 在各个深度 probe
        all_scores = []
        for depth in depths:
            logger.info(f"Probing at depth {depth:.1f}")
            scores = self.probe_single_depth(context, needle, depth, question, select_layer_idx)
            all_scores.append(scores)

        # 平均所有深度的 scores
        avg_scores = torch.stack(all_scores, dim=0).mean(0)  # [num_layers, num_heads]

        # 找到最佳层
        layer_scores = avg_scores.sum(dim=1)  # [num_layers]
        best_layer = torch.argmax(layer_scores).item()

        # 找到最佳 heads
        head_scores = avg_scores[best_layer]  # [num_heads]
        _, best_heads = torch.topk(head_scores, min(num_heads_to_select, len(head_scores)))
        best_heads = best_heads.tolist()

        logger.info(f"Best layer: {best_layer}")
        logger.info(f"Best heads: {best_heads}")

        # 计算覆盖率
        total_score = head_scores.sum()
        selected_score = head_scores[best_heads].sum()
        coverage = selected_score / total_score if total_score > 0 else 0
        logger.info(f"Score coverage: {coverage:.2%}")

        return best_layer, best_heads

    def visualize_scores(
        self,
        scores: torch.Tensor,
        save_path: str = "attention_heatmap.png",
    ) -> None:
        """可视化 attention scores.

        Args:
            scores: [num_layers, num_heads] 的 attention scores
            save_path: 保存路径
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            plt.figure(figsize=(scores.shape[1], scores.shape[0]))
            sns.heatmap(
                scores.cpu().numpy(),
                annot=True,
                fmt=".2f",
                cmap="viridis",
            )
            plt.xlabel("Head")
            plt.ylabel("Layer")
            plt.title("Attention Scores Heatmap")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved heatmap to {save_path}")
        except ImportError:
            logger.warning("matplotlib/seaborn not available, skipping visualization")
