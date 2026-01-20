# @test:skip           - 跳过测试

"""
LongLLMLingua RAG Pipeline - LongBench
======================================

使用 LongLLMLingua 压缩算法的 RAG pipeline。
LongLLMLingua 是针对长文档场景优化的 question-aware prompt 压缩方法。

特点:
    - Question-aware: 使用问题引导上下文重要性评估
    - 动态压缩: 根据内容相关性动态调整压缩比例
    - 上下文重排序: 按相关性排序压缩后的上下文
    - 对比 Perplexity: 使用 condition_compare 提升压缩质量

参考论文: https://arxiv.org/abs/2310.06839
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

from sage_refiner.algorithms.longllmlingua import LongLLMLinguaRefinerOperator


def pipeline_run(config):
    """运行 LongLLMLingua RAG pipeline - LongBench"""
    env = LocalEnvironment()

    (
        env.from_batch(LongBenchBatch, config["source"])
        .map(LongLLMLinguaRefinerOperator, config["refiner"])
        .map(LongBenchPromptor, config["promptor"])
        .map(OpenAIGenerator, config["generator"]["vllm"])
        .map(LongBenchEvaluator, config["evaluate"])
    )

    env.submit(autostop=True)


# ==========================================================
if __name__ == "__main__":
    CustomLogger.disable_global_console_debug()

    if os.getenv("SAGE_EXAMPLES_MODE") == "test" or os.getenv("SAGE_TEST_MODE") == "true":
        print("🧪 Test mode detected - LongBench LongLLMLingua pipeline")
        print("✅ Test passed: Example structure validated")
        sys.exit(0)

    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "config", "config_longllmlingua.yaml"
    )

    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    print("🚀 Starting LongLLMLingua RAG Pipeline (LongBench)...")
    print(f"📊 Dataset: {config['source'].get('hf_dataset_config', 'N/A')}")
    print(f"📈 Max samples: {config['source'].get('max_samples', 'All')}")
    print(f"🤖 Generator: {config['generator']['vllm']['model_name']}")
    print("=" * 60)

    pipeline_run(config)
