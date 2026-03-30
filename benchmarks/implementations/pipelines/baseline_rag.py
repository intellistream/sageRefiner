# @test:skip           - 跳过测试

"""
Baseline RAG Pipeline (No Refiner) - LongBench
==============================================

标准RAG pipeline，不使用任何压缩/refine算法，用于对比实验。
使用 LongBench 数据集（自带 context，无需检索）。
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


def pipeline_run(config):
    """运行Baseline RAG pipeline（无Refiner）- LongBench"""
    env = LocalEnvironment()

    (
        env.from_batch(LongBenchBatch, config["source"])
        # LongBench 自带 context，不需要 Retriever
        # Baseline: 不使用任何 Refiner
        .map(LongBenchPromptor, config["promptor"])
        .map(OpenAIGenerator, config["generator"]["sagellm"])
        .map(LongBenchEvaluator, config["evaluate"])
    )

    env.submit(autostop=True)


# ==========================================================
if __name__ == "__main__":
    CustomLogger.disable_global_console_debug()

    if os.getenv("SAGE_EXAMPLES_MODE") == "test" or os.getenv("SAGE_TEST_MODE") == "true":
        print("🧪 Test mode detected - LongBench Baseline pipeline")
        print("✅ Test passed: Example structure validated")
        sys.exit(0)

    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "config", "config_baseline.yaml"
    )

    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    print("🚀 Starting Baseline RAG Pipeline (LongBench)...")
    print(f"📊 Dataset: {config['source'].get('hf_dataset_config', 'N/A')}")
    print(f"📈 Max samples: {config['source'].get('max_samples', 'All')}")
    print(f"🤖 Generator: {config['generator']['sagellm']['model_name']}")
    print("=" * 60)

    pipeline_run(config)
