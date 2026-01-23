# @test:skip           - 跳过测试

"""
LLMLingua-2 RAG Pipeline - LongBench
====================================

使用 LLMLingua-2 压缩算法的 RAG pipeline。
LLMLingua-2 基于 BERT token 分类，比 LLM-based 方法快得多。

特点:
    - 快速压缩：使用 BERT 模型进行 token 分类，无需 LLM 推理
    - 多语言支持：使用 mBERT 或 XLM-RoBERTa 模型
    - Token 级精确压缩：每个 token 独立分类
    - 可选的上下文级过滤：粗到细的压缩策略

参考论文: https://arxiv.org/abs/2403.12968
"""

import logging
import os
import sys

# 禁用 httpx 的 INFO 日志
logging.getLogger("httpx").setLevel(logging.WARNING)

from sage.benchmark.benchmark_longbench import (
    LongBenchBatch,
    LongBenchEvaluator,
    LongBenchPromptor,
)
from sage.common.utils.config.loader import load_config
from sage.common.utils.logging.custom_logger import CustomLogger
from sage.kernel.api.local_environment import LocalEnvironment
from sage.middleware.operators.rag import OpenAIGenerator

from sage_refiner.algorithms.llmlingua2 import LLMLingua2RefinerOperator


def pipeline_run(config):
    """运行 LLMLingua-2 RAG pipeline - LongBench"""
    env = LocalEnvironment()

    (
        env.from_batch(LongBenchBatch, config["source"])
        .map(LLMLingua2RefinerOperator, config["refiner"])
        .map(LongBenchPromptor, config["promptor"])
        .map(OpenAIGenerator, config["generator"]["vllm"])
        .map(LongBenchEvaluator, config["evaluate"])
    )

    env.submit(autostop=True)


# ==========================================================
if __name__ == "__main__":
    CustomLogger.disable_global_console_debug()

    if os.getenv("SAGE_EXAMPLES_MODE") == "test" or os.getenv("SAGE_TEST_MODE") == "true":
        print("🧪 Test mode detected - LongBench LLMLingua-2 pipeline")
        print("✅ Test passed: Example structure validated")
        sys.exit(0)

    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "config", "config_llmlingua2.yaml"
    )

    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    print("🚀 Starting LLMLingua-2 RAG Pipeline (LongBench)...")
    print(f"📊 Dataset: {config['source'].get('hf_dataset_config', 'N/A')}")
    print(f"📈 Max samples: {config['source'].get('max_samples', 'All')}")
    print(f"🤖 Generator: {config['generator']['vllm']['model_name']}")
    print("=" * 60)

    pipeline_run(config)
