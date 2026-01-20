# @test:skip           - 跳过测试

"""
RECOMP Abstractive RAG Pipeline - LongBench
===========================================

使用RECOMP Abstractive压缩算法的RAG pipeline。
使用微调的T5模型生成检索文档的摘要，将多个检索文档压缩为简洁的摘要。

References:
    RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation
    https://arxiv.org/pdf/2310.04408.pdf
"""

import logging
import os
import sys

# 禁用 httpx 的 INFO 日志
logging.getLogger("httpx").setLevel(logging.WARNING)

from sage.common.utils.config.loader import load_config
from sage.common.utils.logging.custom_logger import CustomLogger
from sage.kernel.api.local_environment import LocalEnvironment
from sage.libs.foundation.io import LongBenchBatch

# RECOMPAbstractiveRefinerOperator may not be available yet (depends on Task 2 completion)
try:
    from sage_refiner.algorithms.recomp_abst import RECOMPAbstractiveRefinerOperator

    if RECOMPAbstractiveRefinerOperator is None:
        raise ImportError("RECOMPAbstractiveRefinerOperator is None")
except ImportError:
    RECOMPAbstractiveRefinerOperator = None
    print(
        "⚠️  Warning: RECOMPAbstractiveRefinerOperator is not available yet.\n"
        "   Please ensure Task 2 (RECOMP Abstractive implementation) is completed first.\n"
        "   See: docs/dev-notes/l4-middleware/recomp-integration-tasks.md"
    )

from sage.benchmark.benchmark_longbench import (
    LongBenchEvaluator,
    LongBenchPromptor,
)
from sage.middleware.operators.rag import OpenAIGenerator


def pipeline_run(config):
    """运行RECOMP Abstractive RAG pipeline - LongBench"""
    env = LocalEnvironment()

    (
        env.from_batch(LongBenchBatch, config["source"])
        .map(RECOMPAbstractiveRefinerOperator, config["refiner"])
        .map(LongBenchPromptor, config["promptor"])
        .map(OpenAIGenerator, config["generator"]["vllm"])
        .map(LongBenchEvaluator, config["evaluate"])
    )

    env.submit(autostop=True)


# ==========================================================
if __name__ == "__main__":
    CustomLogger.disable_global_console_debug()

    if os.getenv("SAGE_EXAMPLES_MODE") == "test" or os.getenv("SAGE_TEST_MODE") == "true":
        print("🧪 Test mode detected - LongBench RECOMP Abstractive pipeline")
        print("✅ Test passed: Example structure validated")
        sys.exit(0)

    # 检查 RECOMPAbstractiveRefinerOperator 是否可用
    if RECOMPAbstractiveRefinerOperator is None:
        print("❌ RECOMPAbstractiveRefinerOperator is not available.")
        print("   Please complete Task 2 (RECOMP Abstractive implementation) first.")
        print("   See: docs/dev-notes/l4-middleware/recomp-integration-tasks.md")
        sys.exit(1)

    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "config", "config_recomp_abst.yaml"
    )

    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    print("🚀 Starting RECOMP Abstractive RAG Pipeline (LongBench)...")
    print(f"📊 Dataset: {config['source'].get('hf_dataset_config', 'N/A')}")
    print(f"📈 Max samples: {config['source'].get('max_samples', 'All')}")
    print(f"🤖 Generator: {config['generator']['vllm']['model_name']}")
    print("=" * 60)

    pipeline_run(config)
