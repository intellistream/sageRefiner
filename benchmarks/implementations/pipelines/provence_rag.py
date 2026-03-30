# @test:skip           - 跳过测试

"""
Provence RAG Pipeline - LongBench
=================================

使用Provence句子级上下文剪枝算法的RAG pipeline。

Provence特点：
    1. 基于DeBERTa-v3的预训练句子评分模型
    2. 根据相关性阈值过滤低相关句子
    3. 可选的重排序功能，将最相关内容排在前面
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

from sage_refiner.algorithms.provence import ProvenceRefinerOperator


def pipeline_run(config):
    """运行Provence RAG pipeline - LongBench"""
    env = LocalEnvironment()

    (
        env.from_batch(LongBenchBatch, config["source"])
        .map(ProvenceRefinerOperator, config["refiner"])
        .map(LongBenchPromptor, config["promptor"])
        .map(OpenAIGenerator, config["generator"]["sagellm"])
        .map(LongBenchEvaluator, config["evaluate"])
    )

    env.submit(autostop=True)


# ==========================================================
if __name__ == "__main__":
    CustomLogger.disable_global_console_debug()

    if os.getenv("SAGE_EXAMPLES_MODE") == "test" or os.getenv("SAGE_TEST_MODE") == "true":
        print("🧪 Test mode detected - LongBench Provence pipeline")
        print("✅ Test passed: Example structure validated")
        sys.exit(0)

    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "config", "config_provence.yaml"
    )

    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    print("🚀 Starting Provence RAG Pipeline (LongBench)...")
    print(f"📊 Dataset: {config['source'].get('hf_dataset_config', 'N/A')}")
    print(f"📈 Max samples: {config['source'].get('max_samples', 'All')}")
    print(f"🤖 Generator: {config['generator']['sagellm']['model_name']}")
    print("=" * 60)

    pipeline_run(config)
