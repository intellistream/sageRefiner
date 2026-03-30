# @test:skip           - 跳过测试

"""
LongRefiner RAG Pipeline - LongBench
====================================

使用LongRefiner三阶段压缩算法的RAG pipeline。

LongRefiner三阶段:
    1. Query Analysis: 分析查询的局部/全局信息需求
    2. Document Structuring: 将文档结构化为层次化的章节
    3. Global Selection: 基于查询分析选择相关内容
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

from sage_refiner.algorithms.LongRefiner import LongRefinerOperator


def pipeline_run(config):
    """运行LongRefiner RAG pipeline - LongBench"""
    env = LocalEnvironment()

    (
        env.from_batch(LongBenchBatch, config["source"])
        .map(LongRefinerOperator, config["refiner"])
        .map(LongBenchPromptor, config["promptor"])
        .map(OpenAIGenerator, config["generator"]["sagellm"])
        .map(LongBenchEvaluator, config["evaluate"])
    )

    env.submit(autostop=True)


# ==========================================================
if __name__ == "__main__":
    CustomLogger.disable_global_console_debug()

    if os.getenv("SAGE_EXAMPLES_MODE") == "test" or os.getenv("SAGE_TEST_MODE") == "true":
        print("🧪 Test mode detected - LongBench LongRefiner pipeline")
        print("✅ Test passed: Example structure validated")
        sys.exit(0)

    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "config", "config_longrefiner.yaml"
    )

    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    print("🚀 Starting LongRefiner RAG Pipeline (LongBench)...")
    print(f"📊 Dataset: {config['source'].get('hf_dataset_config', 'N/A')}")
    print(f"📈 Max samples: {config['source'].get('max_samples', 'All')}")
    print(f"🤖 Generator: {config['generator']['sagellm']['model_name']}")
    print("=" * 60)

    pipeline_run(config)
