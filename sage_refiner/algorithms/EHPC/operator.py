"""
EHPC Operator for SAGE Pipeline (EMI Mode)
==========================================

将 EHPC 压缩算法封装为 SAGE MapOperator，用于 RAG pipeline。
支持 External Model Initialization (EMI) 模式：refiner ≠ generator。

特点:
    - Refiner 模型：专门用于压缩（本 Operator 管理）
    - Generator 模型：专门用于生成（由 SAGE pipeline 管理）
    - 基于 Evaluator Heads 的高效压缩
    - 输出压缩后的文本，可用于任何 Generator

使用方法 (EMI):
    >>> config = {
    ...     "enabled": True,
    ...     "refiner_model": "meta-llama/Llama-3.1-8B-Instruct",  # Refiner
    ...     "layer": 14,
    ...     "heads": [24, 3, 18, 7, 29, 2, 9, 1],
    ...     "topk": 2048,
    ... }
    >>> operator = EHPCRefinerOperator(config)
    >>> # Refiner 压缩上下文
    >>> result = operator.execute({
    ...     "query": "What is machine learning?",
    ...     "retrieval_results": [{"text": "Machine learning is..."}],
    ... })
    >>> # result["compressed_context"] 可用于任何 Generator 模型
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from sage.common.core.functions import MapFunction as MapOperator

from .compressor import EHPCCompressor
from .config import EHPCConfig

logger = logging.getLogger(__name__)


class EHPCRefinerOperator(MapOperator):
    """EHPC Refiner Operator for SAGE Pipeline (EMI Mode).

    将 EHPC 压缩算法封装为 SAGE pipeline 的操作符。
    支持 EMI 模式：refiner ≠ generator。

    Input Format:
        {
            "query": str,
            "retrieval_results": List[dict],  # 检索到的文档
        }

    Output Format:
        {
            "query": str,
            "retrieval_results": List[dict],  # 原始文档（保留）
            "refining_results": List[str],    # 压缩后的文档列表
            "compressed_context": str,         # 完整压缩后的上下文（文本）
            "original_tokens": int,
            "compressed_tokens": int,
            "compression_rate": float,
        }

    Configuration Options:
        enabled (bool): 是否启用压缩。默认 True。
        refiner_model (str): Refiner 模型名称（用于压缩）。
            默认 "meta-llama/Llama-3.1-8B-Instruct"。
        device (str): 推理设备。默认 "cuda"。
        layer (int): 执行 selection 的层索引。
        heads (List[int]): Evaluator heads 列表。
        topk (int): 保留的 token 数量。
        window_size (int): 窗口大小。
        pool (str): 池化类型 ('avg_pool', 'max_pool')。
        kernel_size (int): 池化核大小。
        auto_config (bool): 是否根据模型自动选择 layer/heads 配置。默认 True。
    """

    def __init__(self, config: dict[str, Any], ctx: Any = None):
        """初始化 EHPC Operator.

        Args:
            config: 配置字典
            ctx: 可选的上下文对象
        """
        super().__init__(config=config, ctx=ctx)
        self.cfg = config
        self.enabled = config.get("enabled", True)

        self._compressor: EHPCCompressor | None = None
        self._model = None
        self._tokenizer = None
        self._refiner_model_name: str = config.get(
            "refiner_model",
            config.get("model_name", "meta-llama/Llama-3.1-8B-Instruct"),
        )

        if self.enabled:
            self._init_compressor()
            logger.info("EHPC Operator initialized")
        else:
            logger.info("EHPC disabled (baseline mode)")

    def _init_compressor(self) -> None:
        """初始化 EHPC 压缩器（Refiner 模型）."""
        from .models.gemfilter import load_ehpc_model

        device = self.cfg.get("device", "cuda")

        # 加载 Refiner 模型（用于压缩）
        self._model, self._tokenizer = load_ehpc_model(
            self._refiner_model_name,
            flash_attention_2=self.cfg.get("flash_attention_2", True),
        )

        # 创建压缩器
        self._compressor = EHPCCompressor(
            model=self._model,
            tokenizer=self._tokenizer,
            device=device,
        )

        logger.info(f"EHPC Refiner initialized with model: {self._refiner_model_name}")

    def _get_ehpc_config(self) -> EHPCConfig:
        """从配置创建 EHPCConfig 对象.

        如果启用 auto_config，会根据模型名称自动选择合适的 layer 和 heads。
        手动指定的参数会覆盖自动配置。
        """
        from .config import get_config_for_model

        auto_config = self.cfg.get("auto_config", True)

        if auto_config:
            # 收集用户覆盖的参数
            overrides = {}
            if "layer" in self.cfg:
                overrides["layer"] = self.cfg["layer"]
            if "heads" in self.cfg:
                overrides["heads"] = self.cfg["heads"]
            if "topk" in self.cfg:
                overrides["topk"] = self.cfg["topk"]
            if "window_size" in self.cfg:
                overrides["window_size"] = self.cfg["window_size"]
            if "pool" in self.cfg:
                overrides["pool"] = self.cfg["pool"]
            if "kernel_size" in self.cfg:
                overrides["kernel_size"] = self.cfg["kernel_size"]

            # 根据模型自动获取配置，并应用覆盖
            config = get_config_for_model(self._refiner_model_name, **overrides)
            logger.info(
                f"Auto-configured EHPC for {self._refiner_model_name}: "
                f"layer={config.layer}, heads={config.heads}"
            )
            return config
        # 完全手动配置
        return EHPCConfig(
            layer=self.cfg.get("layer", 14),
            heads=self.cfg.get("heads", [24, 3, 18, 7, 29, 2, 9, 1]),
            topk=self.cfg.get("topk", 2048),
            window_size=self.cfg.get("window_size", 32),
            pool=self.cfg.get("pool", "avg_pool"),
            kernel_size=self.cfg.get("kernel_size", 4),
        )

    def execute(self, data: dict[str, Any]) -> dict[str, Any]:
        """执行 EHPC 压缩（EMI 模式）.

        使用 Refiner 模型压缩上下文，输出压缩后的文本。
        压缩后的文本可用于任何 Generator 模型。

        Args:
            data: 输入数据，包含 query 和 retrieval_results

        Returns:
            添加了压缩结果的数据（包含压缩后的文本）
        """
        if not self.enabled:
            return self._bypass(data)

        query = data.get("query", "")
        retrieval_results = data.get("retrieval_results", [])

        # 如果 retrieval_results 为空，尝试从 context 字段获取（支持 LongBench 等数据源）
        if not retrieval_results:
            context = data.get("context", "")
            if context:
                retrieval_results = [context] if isinstance(context, str) else list(context)

        # 提取文档文本
        documents = []
        for doc in retrieval_results:
            text = doc.get("text", doc.get("content", "")) if isinstance(doc, dict) else str(doc)
            documents.append(text)

        if not documents:
            logger.warning("No documents to compress")
            return self._bypass(data)

        # 获取 EHPC 配置
        ehpc_config = self._get_ehpc_config()

        # 执行压缩（使用 compress_text 接口 for EMI mode）
        try:
            result = self._compressor.compress_text(
                context=documents,
                question=query if self.cfg.get("include_query", False) else "",
                config=ehpc_config,
            )

            compressed_context = result["compressed_context"]
            original_tokens = result["original_tokens"]
            compressed_tokens = result["compressed_tokens"]
            compression_rate = result["compression_rate"]

            # 更新数据
            data["refining_results"] = [compressed_context]
            data["compressed_context"] = compressed_context
            data["original_tokens"] = original_tokens
            data["compressed_tokens"] = compressed_tokens
            data["compression_rate"] = compression_rate

            logger.info(
                f"EHPC compression (EMI): {original_tokens} -> {compressed_tokens} tokens "
                f"(rate: {compression_rate:.2%})"
            )

        except Exception as e:
            logger.error(f"EHPC compression failed: {e}")
            return self._bypass(data)

        return data

    def _bypass(self, data: dict[str, Any]) -> dict[str, Any]:
        """跳过压缩，直接返回原始数据.

        Args:
            data: 输入数据

        Returns:
            带原始文档的数据
        """
        retrieval_results = data.get("retrieval_results", [])

        documents = []
        for doc in retrieval_results:
            text = doc.get("text", doc.get("content", "")) if isinstance(doc, dict) else str(doc)
            documents.append(text)

        context = "\n\n".join(documents)

        data["refining_results"] = documents
        data["compressed_context"] = context
        data["original_tokens"] = len(context.split())
        data["compressed_tokens"] = len(context.split())
        data["compression_rate"] = 1.0

        return data

    def cleanup(self) -> None:
        """清理资源."""
        if self._model is not None:
            del self._model
            self._model = None

        if self._compressor is not None:
            del self._compressor
            self._compressor = None

        # 尝试释放 GPU 内存
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

        logger.info("EHPC Operator cleaned up")
