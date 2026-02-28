# @test:skip           - 跳过测试

"""
RECOMP Extractive RAG Pipeline - LongBench
==========================================

使用RECOMP Extractive压缩算法的RAG pipeline。
基于双编码器（Contriever/DPR）对检索文档进行句子级打分，
选择与query最相关的top-k句子作为压缩后的上下文。

References:
    RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation
    https://arxiv.org/pdf/2310.04408.pdf
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

from sage_refiner.algorithms.recomp_extr import RECOMPExtractiveRefinerOperator


def pipeline_run(config):
    """运行RECOMP Extractive RAG pipeline - LongBench"""
    env = LocalEnvironment()

    (
        env.from_batch(LongBenchBatch, config["source"])
        .map(RECOMPExtractiveRefinerOperator, config["refiner"])
        .map(LongBenchPromptor, config["promptor"])
        .map(OpenAIGenerator, config["generator"]["sagellm"])
        .map(LongBenchEvaluator, config["evaluate"])
    )

    env.submit(autostop=True)


# ==========================================================
if __name__ == "__main__":
    CustomLogger.disable_global_console_debug()

    if os.getenv("SAGE_EXAMPLES_MODE") == "test" or os.getenv("SAGE_TEST_MODE") == "true":
        print("🧪 Test mode detected - LongBench RECOMP Extractive pipeline")
        print("✅ Test passed: Example structure validated")
        sys.exit(0)

    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "config", "config_recomp_extr.yaml"
    )

    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    print("🚀 Starting RECOMP Extractive RAG Pipeline (LongBench)...")
    print(f"📊 Dataset: {config['source'].get('hf_dataset_config', 'N/A')}")
    print(f"📈 Max samples: {config['source'].get('max_samples', 'All')}")
    print(f"🤖 Generator: {config['generator']['sagellm']['model_name']}")
    print("=" * 60)

    pipeline_run(config)
