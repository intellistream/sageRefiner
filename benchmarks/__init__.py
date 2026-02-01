"""
Benchmarks - Refiner 算法评测套件
==================================

Layer: L5 (Applications - Benchmarking)
Dependencies: sage.middleware (L4), sage.libs (L3)

提供多种上下文压缩算法的评测框架，包括：
- 评测 Pipeline (baseline, longrefiner, reform, provence, llmlingua2, longllmlingua)
- 注意力头分析工具 (MNR指标)
- 统一实验管理框架
- CLI 命令行工具

核心评测指标 (来自 sage.middleware.operators.rag.evaluate):
- F1Evaluate: Token级别F1分数
- TokenCountEvaluate: Token计数
- LatencyEvaluate: 延迟评测 (Retrieve/Refine/Generate)
- CompressionRateEvaluate: 压缩率
- RougeLEvaluate: ROUGE-L F1分数
- RecallEvaluate / BertRecallEvaluate: 召回率

已实现的算法:
- Baseline: 无压缩基线 (截断)
- LongRefiner: 三阶段压缩 (Query Analysis → Doc Structuring → Global Selection)
- REFORM: 基于注意力头的token级压缩，支持KV cache优化
- Provence: 句子级上下文剪枝 (DeBERTa-v3)
- LLMLingua-2: BERT-based 快速压缩
- LongLLMLingua: 问题感知的长文档压缩 (PPL-based)

CLI 使用:
    # 快速对比多算法
    sage-refiner-bench compare --algorithms baseline,longrefiner,reform --samples 100

    # 从配置文件运行实验
    sage-refiner-bench run --config experiment.yaml

    # Budget 扫描
    sage-refiner-bench sweep --algorithm longrefiner --budgets 512,1024,2048

Python 使用:
    from benchmarks.experiments import (
        RefinerExperimentRunner,
        RefinerExperimentConfig,
    )

    # 快速对比
    runner = RefinerExperimentRunner()
    result = runner.quick_compare(
        algorithms=["baseline", "longrefiner", "reform"],
        max_samples=100,
    )

目录结构:
    benchmarks/
    ├── analysis/              # 注意力头分析工具
    │   ├── head_analysis.py   # 注意力头重要性分析
    │   ├── statistical.py     # 统计显著性检验
    │   ├── visualization.py   # 可视化工具
    │   ├── latex_export.py    # LaTeX导出
    │   └── find_heads.py      # 头部选择工具
    ├── config/                # 配置文件
    │   └── config_*.yaml      # 各算法配置
    ├── experiments/           # 统一实验管理
    │   ├── base_experiment.py # 基础实验类
    │   ├── comparison_experiment.py # 对比实验
    │   ├── results_collector.py # 结果收集器
    │   └── runner.py          # 实验运行器
    ├── implementations/       # Pipeline实现
    │   └── pipelines/         # 各算法RAG pipeline
    ├── cli.py                 # CLI入口
    └── README.md              # 本模块文档
"""

__layer__ = "L5"

# Analysis tools
from benchmarks.analysis import (
    AttentionHookExtractor,
    HeadwiseEvaluator,
    MetricsAggregator,
    mean_normalized_rank,
    plot_mnr_curve,
)

# Experiment framework
from benchmarks.experiments import (
    AlgorithmMetrics,
    BaseRefinerExperiment,
    ComparisonExperiment,
    CompressionExperiment,
    DatasetType,
    ExperimentResult,
    LatencyExperiment,
    QualityExperiment,
    RefinerAlgorithm,
    RefinerExperimentConfig,
    RefinerExperimentRunner,
)

__all__ = [
    # Analysis
    "AttentionHookExtractor",
    "HeadwiseEvaluator",
    "MetricsAggregator",
    "mean_normalized_rank",
    "plot_mnr_curve",
    # Experiment framework - Enums
    "RefinerAlgorithm",
    "DatasetType",
    # Experiment framework - Config and results
    "RefinerExperimentConfig",
    "AlgorithmMetrics",
    "ExperimentResult",
    # Experiment framework - Experiment classes
    "BaseRefinerExperiment",
    "ComparisonExperiment",
    "QualityExperiment",
    "LatencyExperiment",
    "CompressionExperiment",
    # Experiment framework - Runner
    "RefinerExperimentRunner",
]
