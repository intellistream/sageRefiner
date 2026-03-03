"""
Base Experiment Framework for Refiner Benchmarking
==================================================

提供 Refiner 算法评测的基础类和数据模型。

设计原则：
1. 与 sage.middleware.operators.rag.evaluate 中的评测指标集成
2. 支持多种 Refiner 算法的统一评测
3. 支持配置驱动的实验管理
"""

from abc import ABC, abstractmethod
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class RefinerAlgorithm(str, Enum):
    """支持的 Refiner 算法"""

    BASELINE = "baseline"  # 无压缩基线
    LONGREFINER = "longrefiner"  # LongRefiner 三阶段压缩
    REFORM = "reform"  # REFORM 注意力头压缩
    PROVENCE = "provence"  # Provence 句子级剪枝
    SIMPLE = "simple"  # 简单截断
    RECOMP = "recomp"  # RECOMP (TODO)
    LLMLINGUA2 = "llmlingua2"  # LLMLingua-2: BERT token classification compression
    LONGLLMLINGUA = "longllmlingua"  # LongLLMLingua: Question-aware long context compression

    @classmethod
    def available(cls) -> list[str]:
        """返回已实现的算法列表"""
        return [
            cls.BASELINE.value,
            cls.LONGREFINER.value,
            cls.REFORM.value,
            cls.PROVENCE.value,
            cls.LLMLINGUA2.value,
            cls.LONGLLMLINGUA.value,
        ]


class DatasetType(str, Enum):
    """支持的数据集类型"""

    NQ = "nq"  # Natural Questions
    HOTPOTQA = "hotpotqa"
    TRIVIAQA = "triviaqa"
    SQUAD = "squad"
    ASQA = "asqa"
    CUSTOM = "custom"


# FlashRAG 支持的数据集列表
AVAILABLE_DATASETS = [
    "nq",  # Natural Questions
    "triviaqa",  # TriviaQA
    "hotpotqa",  # HotpotQA (multi-hop)
    "2wikimultihopqa",  # 2Wiki Multi-hop
    "musique",  # Musique (multi-hop)
    "asqa",  # ASQA (long-form)
    "popqa",  # PopQA
    "webq",  # WebQuestions
]


@dataclass
class RefinerExperimentConfig:
    """Refiner 评测实验配置"""

    # === 基础配置 ===
    name: str = "refiner_experiment"
    description: str = ""

    # === 数据配置 ===
    dataset: DatasetType = DatasetType.NQ  # 向后兼容，单数据集
    datasets: list[str] = field(default_factory=lambda: ["nq"])  # 多数据集支持
    dataset_config: str = "nq"  # HuggingFace dataset config
    split: str = "test"
    max_samples: int = 100

    # === 算法配置 ===
    algorithms: list[str] = field(default_factory=lambda: ["baseline", "longrefiner"])
    budget: int = 2048  # Token 预算

    # === 检索配置 ===
    retriever_type: str = "wiki18_faiss"
    top_k: int = 100
    index_path: str = ""
    documents_path: str = ""

    # === 生成配置 ===
    generator_model: str = "Llama-3.1-8B-Instruct"
    generator_base_url: str = "http://localhost:8889/v1"

    # === 输出配置 ===
    output_dir: str = "./.benchmarks/refiner"
    save_raw_results: bool = True
    generate_report: bool = True

    # === 运行配置 ===
    seed: int = 42
    verbose: bool = True
    timeout: int = 600  # 单个样本超时（秒）

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "description": self.description,
            "dataset": self.dataset.value if isinstance(self.dataset, Enum) else self.dataset,
            "datasets": self.datasets,
            "dataset_config": self.dataset_config,
            "split": self.split,
            "max_samples": self.max_samples,
            "algorithms": self.algorithms,
            "budget": self.budget,
            "retriever_type": self.retriever_type,
            "top_k": self.top_k,
            "index_path": self.index_path,
            "documents_path": self.documents_path,
            "generator_model": self.generator_model,
            "generator_base_url": self.generator_base_url,
            "output_dir": self.output_dir,
            "save_raw_results": self.save_raw_results,
            "generate_report": self.generate_report,
            "seed": self.seed,
            "verbose": self.verbose,
            "timeout": self.timeout,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RefinerExperimentConfig":
        """从字典创建配置"""
        # 处理枚举类型
        if "dataset" in data and isinstance(data["dataset"], str):
            with suppress(ValueError):
                data["dataset"] = DatasetType(data["dataset"])
        # 处理 datasets 字段
        if "datasets" not in data and "dataset" in data:
            # 向后兼容：如果没有 datasets，从 dataset 创建
            dataset_val = data["dataset"]
            if isinstance(dataset_val, DatasetType):
                data["datasets"] = [dataset_val.value]
            else:
                data["datasets"] = [dataset_val]
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str) -> "RefinerExperimentConfig":
        """从 YAML 文件加载配置"""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data.get("experiment", data))

    def save_yaml(self, path: str) -> None:
        """保存配置到 YAML 文件"""
        with open(path, "w") as f:
            yaml.dump({"experiment": self.to_dict()}, f, default_flow_style=False)

    def validate(self) -> list[str]:
        """验证配置，返回错误列表"""
        errors = []

        # 检查算法
        available = RefinerAlgorithm.available()
        for algo in self.algorithms:
            if algo not in available and algo not in [e.value for e in RefinerAlgorithm]:
                errors.append(f"Unknown algorithm: {algo}")

        # 检查数据集
        if self.max_samples <= 0:
            errors.append("max_samples must be positive")

        # 验证 datasets 列表
        if not self.datasets:
            errors.append("datasets cannot be empty")
        else:
            for ds in self.datasets:
                if ds not in AVAILABLE_DATASETS:
                    errors.append(f"Unknown dataset: {ds}. Available: {AVAILABLE_DATASETS}")

        # 检查 budget
        if self.budget <= 0:
            errors.append("budget must be positive")

        return errors

    def get_datasets(self) -> list[str]:
        """
        获取要运行的数据集列表

        Returns:
            数据集名称列表
        """
        return self.datasets


@dataclass
class AlgorithmMetrics:
    """单个算法的评测指标"""

    algorithm: str
    num_samples: int = 0

    # 质量指标
    avg_f1: float = 0.0
    avg_recall: float = 0.0
    avg_rouge_l: float = 0.0
    avg_accuracy: float = 0.0

    # 压缩指标
    avg_compression_rate: float = 0.0
    avg_original_tokens: float = 0.0
    avg_compressed_tokens: float = 0.0

    # 延迟指标（秒）
    avg_retrieve_time: float = 0.0
    avg_refine_time: float = 0.0
    avg_generate_time: float = 0.0
    avg_total_time: float = 0.0

    # 标准差
    std_f1: float = 0.0
    std_compression_rate: float = 0.0
    std_total_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "algorithm": self.algorithm,
            "num_samples": self.num_samples,
            "quality": {
                "avg_f1": self.avg_f1,
                "avg_recall": self.avg_recall,
                "avg_rouge_l": self.avg_rouge_l,
                "avg_accuracy": self.avg_accuracy,
                "std_f1": self.std_f1,
            },
            "compression": {
                "avg_compression_rate": self.avg_compression_rate,
                "avg_original_tokens": self.avg_original_tokens,
                "avg_compressed_tokens": self.avg_compressed_tokens,
                "std_compression_rate": self.std_compression_rate,
            },
            "latency": {
                "avg_retrieve_time": self.avg_retrieve_time,
                "avg_refine_time": self.avg_refine_time,
                "avg_generate_time": self.avg_generate_time,
                "avg_total_time": self.avg_total_time,
                "std_total_time": self.std_total_time,
            },
        }


@dataclass
class ExperimentResult:
    """实验结果"""

    experiment_id: str
    config: dict[str, Any]
    algorithm_metrics: dict[str, AlgorithmMetrics] = field(default_factory=dict)
    raw_results: list[dict[str, Any]] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0
    success: bool = True
    error: str = ""

    # 对比结果
    best_f1_algorithm: str = ""
    best_compression_algorithm: str = ""
    best_latency_algorithm: str = ""

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "experiment_id": self.experiment_id,
            "config": self.config,
            "algorithm_metrics": {
                name: metrics.to_dict() for name, metrics in self.algorithm_metrics.items()
            },
            "raw_results": self.raw_results if len(self.raw_results) <= 100 else [],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "success": self.success,
            "error": self.error,
            "summary": {
                "best_f1_algorithm": self.best_f1_algorithm,
                "best_compression_algorithm": self.best_compression_algorithm,
                "best_latency_algorithm": self.best_latency_algorithm,
            },
        }


class BaseRefinerExperiment(ABC):
    """
    Refiner 评测实验基类

    定义实验生命周期：
    1. prepare() - 准备数据和资源
    2. run() - 执行实验
    3. analyze() - 分析结果
    4. finalize() - 清理和保存

    子类必须实现 run() 方法。
    """

    def __init__(self, config: RefinerExperimentConfig):
        """
        初始化实验

        Args:
            config: 实验配置
        """
        self.config = config
        self.experiment_id = (
            f"{config.name}_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )
        self.output_dir = Path(config.output_dir) / self.experiment_id
        self.result: ExperimentResult | None = None

    def _log(self, message: str) -> None:
        """日志输出"""
        if self.config.verbose:
            print(message)

    def prepare(self) -> None:
        """
        准备实验

        - 验证配置
        - 创建输出目录
        - 加载数据源
        """
        # 验证配置
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Configuration errors: {errors}")

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 保存配置
        config_path = self.output_dir / "config.yaml"
        self.config.save_yaml(str(config_path))

        self._log(f"📁 Output directory: {self.output_dir}")
        self._log(f"📝 Config saved to: {config_path}")

    @abstractmethod
    def run(self) -> ExperimentResult:
        """
        运行实验

        Returns:
            ExperimentResult 包含所有算法的评测结果
        """

    def analyze(self, result: ExperimentResult) -> ExperimentResult:
        """
        分析实验结果，确定最佳算法

        Args:
            result: 原始实验结果

        Returns:
            添加了分析结论的实验结果
        """
        if not result.algorithm_metrics:
            return result

        # 找出最佳 F1
        best_f1 = max(
            result.algorithm_metrics.items(),
            key=lambda x: x[1].avg_f1,
        )
        result.best_f1_algorithm = best_f1[0]

        # 找出最佳压缩率（越高越好）
        best_compression = max(
            result.algorithm_metrics.items(),
            key=lambda x: x[1].avg_compression_rate,
        )
        result.best_compression_algorithm = best_compression[0]

        # 找出最低延迟
        best_latency = min(
            result.algorithm_metrics.items(),
            key=lambda x: x[1].avg_total_time if x[1].avg_total_time > 0 else float("inf"),
        )
        result.best_latency_algorithm = best_latency[0]

        return result

    def finalize(self, result: ExperimentResult) -> None:
        """
        完成实验

        - 保存结果
        - 生成报告
        - 清理资源
        """
        import json

        # 保存 JSON 结果
        result_path = self.output_dir / "results.json"
        with open(result_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        self._log(f"💾 Results saved to: {result_path}")

        # 生成 Markdown 报告
        if self.config.generate_report:
            report_path = self.output_dir / "report.md"
            self._generate_report(result, report_path)
            self._log(f"📊 Report saved to: {report_path}")

    def _generate_report(self, result: ExperimentResult, path: Path) -> None:
        """生成 Markdown 报告"""
        lines = [
            "# Refiner Benchmark Report",
            "",
            f"**Experiment ID**: {result.experiment_id}",
            f"**Duration**: {result.duration_seconds:.1f}s",
            f"**Samples**: {self.config.max_samples}",
            f"**Dataset**: {self.config.dataset.value}",
            "",
            "## Summary",
            "",
            "| Best For | Algorithm |",
            "|----------|-----------|",
            f"| F1 Score | {result.best_f1_algorithm} |",
            f"| Compression | {result.best_compression_algorithm} |",
            f"| Latency | {result.best_latency_algorithm} |",
            "",
            "## Algorithm Comparison",
            "",
            "| Algorithm | F1 | Compression | Latency (s) |",
            "|-----------|-----|-------------|-------------|",
        ]

        for name, metrics in result.algorithm_metrics.items():
            lines.append(
                f"| {name} | {metrics.avg_f1:.4f} | "
                f"{metrics.avg_compression_rate:.2f}x | {metrics.avg_total_time:.2f} |"
            )

        lines.extend(
            [
                "",
                "## Configuration",
                "",
                "```yaml",
            ]
        )

        # 添加配置
        import yaml

        lines.append(yaml.dump(result.config, default_flow_style=False))
        lines.append("```")

        with open(path, "w") as f:
            f.write("\n".join(lines))

    def run_full(self) -> ExperimentResult:
        """
        运行完整实验流程

        Returns:
            ExperimentResult
        """
        self._log(f"\n{'=' * 60}")
        self._log(f"🚀 Starting Refiner Benchmark: {self.config.name}")
        self._log(f"{'=' * 60}")

        try:
            # 1. 准备
            self._log("\n📦 Preparing experiment...")
            self.prepare()

            # 2. 运行
            self._log("\n▶️  Running experiment...")
            result = self.run()

            # 3. 分析
            self._log("\n🔍 Analyzing results...")
            result = self.analyze(result)

            # 4. 完成
            self._log("\n💾 Finalizing...")
            self.finalize(result)

            self._log(f"\n{'=' * 60}")
            self._log("✅ Experiment completed successfully")
            self._log(f"{'=' * 60}")

            return result

        except Exception as e:
            self._log(f"\n❌ Experiment failed: {e}")
            import traceback

            traceback.print_exc()

            return ExperimentResult(
                experiment_id=self.experiment_id,
                config=self.config.to_dict(),
                success=False,
                error=str(e),
            )
