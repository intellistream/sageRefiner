# Benchmarks 模块结构文档

## 概述

`benchmarks` 模块是 sageRefiner 的核心评测套件，用于评测各种上下文压缩算法在 RAG 场景下的性能。

## 目录结构

```
benchmarks/
├── __init__.py              # 模块导出
├── README.md                # 用户文档
├── STRUCTURE.md             # 本文档 - 结构说明
├── cli.py                   # CLI命令行入口
│
├── analysis/                # 注意力头分析工具层
│   ├── __init__.py          # 导出分析工具
│   ├── head_analysis.py     # 注意力头重要性分析器
│   ├── find_heads.py        # 最优头部选择算法
│   ├── statistical.py       # 统计显著性检验
│   ├── visualization.py     # 结果可视化 (图表/热力图)
│   └── latex_export.py      # LaTeX表格导出
│
├── config/                  # 配置管理层
│   ├── __init__.py          # 配置类和加载器
│   ├── config_baseline.yaml     # Baseline (截断) 配置
│   ├── config_longrefiner.yaml  # LongRefiner 配置
│   ├── config_reform.yaml       # REFORM 配置
│   ├── config_provence.yaml     # Provence 配置
│   ├── config_llmlingua2.yaml   # LLMLingua-2 配置
│   ├── config_longllmlingua.yaml# LongLLMLingua 配置
│   ├── config_recomp_extr.yaml  # RECOMP-Extractive 配置
│   ├── config_recomp_abst.yaml  # RECOMP-Abstractive 配置
│   └── head_analysis_config.yaml # 注意力头分析配置
│
├── experiments/             # 实验管理层
│   ├── __init__.py          # 导出实验框架
│   ├── base_experiment.py   # 基础实验类
│   │                        #   - BaseRefinerExperiment (单算法实验)
│   │                        #   - QualityExperiment (质量评测)
│   │                        #   - LatencyExperiment (延迟评测)
│   │                        #   - CompressionExperiment (压缩率评测)
│   │
│   ├── comparison_experiment.py # 对比实验
│   │                        #   - ComparisonExperiment (多算法对比)
│   │
│   ├── results_collector.py # 结果收集和处理
│   │                        #   - ExperimentResult (结果数据类)
│   │                        #   - AlgorithmMetrics (算法指标)
│   │                        #   - ResultsCollector (结果收集器)
│   │
│   └── runner.py            # 实验运行器
│                            #   - RefinerExperimentRunner (执行框架)
│                            #   - RefinerExperimentConfig (实验配置)
│                            #   - RefinerAlgorithm (算法枚举)
│                            #   - DatasetType (数据集类型)
│
└── implementations/         # RAG Pipeline实现层
    ├── __init__.py          # 导出Pipeline实现
    │
    └── pipelines/           # 各算法的RAG Pipeline
        ├── __init__.py      # 导出所有pipelines
        ├── baseline_rag.py     # Baseline (无压缩) Pipeline
        ├── longrefiner_rag.py  # LongRefiner Pipeline
        ├── reform_rag.py       # REFORM Pipeline
        ├── provence_rag.py     # Provence Pipeline
        ├── llmlingua2_rag.py   # LLMLingua-2 Pipeline
        ├── longllmlingua_rag.py# LongLLMLingua Pipeline
        ├── recomp_extr_rag.py  # RECOMP-Extractive Pipeline
        └── ehpc_rag.py         # EHPC (Enhanced Hierarchical Passage Compression) Pipeline
```

## 核心概念

### 1. 评测指标

所有评测都基于 `sage.middleware.operators.rag.evaluate` 中定义的标准指标：

| 指标 | 类 | 说明 |
|------|-----|------|
| **F1** | `F1Evaluate` | Token级别F1分数 |
| **Token Count** | `TokenCountEvaluate` | 压缩后的Token数 |
| **Latency** | `LatencyEvaluate` | Retrieve/Refine/Generate各阶段延迟 |
| **Compression Rate** | `CompressionRateEvaluate` | 原始tokens / 压缩后tokens |
| **ROUGE-L** | `RougeLEvaluate` | ROUGE-L F1分数 |
| **Recall** | `RecallEvaluate` | 召回率 |

### 2. 支持的算法

| 算法 | 类型 | 特点 | 依赖 |
|------|------|------|------|
| **Baseline** | 截断 | 无压缩，仅截断 | 无 |
| **LongRefiner** | LLM-based | 三阶段LLM指导，高质量 | 需要LLM |
| **REFORM** | Attention-based | 注意力头驱动，快速 | 需要LLM |
| **Provence** | Provenance-aware | 句子级剪枝 | 需要DeBERTa |
| **LLMLingua-2** | BERT-based | 快速，多语言 | 需要BERT |
| **LongLLMLingua** | PPL-based | 问题感知 | 需要LLM |
| **RECOMP** | Extractive/Abstractive | 重组合式压缩 | 需要LLM |

### 3. 实验框架

**RefinerExperimentRunner** 提供统一的实验管理：

```python
from benchmarks import RefinerExperimentRunner, RefinerExperimentConfig

# 快速对比
runner = RefinerExperimentRunner()
result = runner.quick_compare(
    algorithms=["baseline", "longrefiner", "reform"],
    max_samples=100,
)

# 完整实验
config = RefinerExperimentConfig(
    algorithms=["longrefiner", "reform"],
    budgets=[512, 1024, 2048],
    datasets=["nq", "msmarco"],
)
runner = RefinerExperimentRunner(config)
runner.run_all()
```

## 使用流程

### 1. 命令行使用

```bash
# 快速对比多算法
sage-refiner-bench compare \
    --algorithms baseline,longrefiner,reform \
    --samples 100 \
    --output results/comparison.json

# 单个算法评测
sage-refiner-bench run baseline \
    --dataset nq \
    --samples 10 \
    --output results/baseline.json

# Budget扫描
sage-refiner-bench sweep \
    --algorithm longrefiner \
    --budgets 512,1024,2048 \
    --output results/sweep.json
```

### 2. Python API使用

```python
# 导入实验框架
from benchmarks.experiments import (
    RefinerExperimentRunner,
    RefinerExperimentConfig,
)

# 配置实验
config = RefinerExperimentConfig(
    algorithms=["baseline", "longrefiner", "reform"],
    budgets=[1024, 2048],
    max_samples=100,
)

# 运行实验
runner = RefinerExperimentRunner(config)
results = runner.run()

# 分析结果
print(f"LongRefiner F1: {results['longrefiner'].metrics.f1:.4f}")
print(f"Compression: {results['longrefiner'].metrics.compression_rate:.2f}x")
```

### 3. Pipeline集成

每个算法都提供独立的RAG Pipeline：

```python
# LongRefiner Pipeline
from benchmarks.implementations.pipelines import LongRefinerRAGPipeline

pipeline = LongRefinerRAGPipeline()
compressed_context = pipeline(question="query", documents=[...], budget=2048)
```

## 扩展指南

### 添加新算法

1. 在 `implementations/pipelines/` 中创建 `new_algorithm_rag.py`
2. 实现 `RAGPipeline` 接口
3. 在 `config/` 中添加 `config_new_algorithm.yaml`
4. 更新 `__init__.py` 导出新算法

### 添加新指标

1. 在 SAGE 框架的 `sage.middleware.operators.rag.evaluate` 中定义
2. 更新 `experiments/base_experiment.py` 中的指标收集
3. 更新 `results_collector.py` 中的结果类

## 依赖关系

```
benchmarks/
├── 核心依赖:
│   ├── sage.middleware.operators.rag.evaluate (指标)
│   ├── sage.kernel.api.local_environment (环境管理)
│   └── sage.libs.foundation.io.batch (数据批处理)
│
├── 算法实现依赖:
│   ├── sage_refiner (核心压缩算法)
│   ├── sage_middleware (Operator框架)
│   ├── torch & transformers (深度学习)
│   └── 其他算法库 (BERT, DeBERTa 等)
│
└── 工具依赖:
    ├── scipy (统计检验)
    ├── matplotlib (可视化)
    ├── pandas (数据处理)
    └── datasets (数据集加载)
```

## 注意事项

1. **GPU资源**: LongRefiner, REFORM等算法需要GPU，推荐RTX 3090+
2. **内存**: 评测大数据集时需要充足内存，推荐48GB+
3. **时间**: 完整评测可能需要数小时，建议先用小样本测试
4. **配置**: 所有配置都在 `config/` 目录，可根据需要调整参数

## 故障排查

| 问题 | 解决方案 |
|------|---------|
| CUDA OOM | 减小 `max_model_len`, 启用 `gpu_memory_utilization` |
| 模型加载失败 | 检查网络连接，预下载模型权重 |
| 评测缓慢 | 减少样本数，使用更小的model, 启用多进程 |
| 指标异常 | 检查数据集格式，验证prompt工程 |

## 更新历史

- 2026-02-01: 初始结构，集成从sage-benchmark提取的benchmark_refiner
- 2026-02-02: 完善目录结构和文档
