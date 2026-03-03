# ADR-0003: Torch/Transformers Usage Surface Convergence

- Status: Accepted
- Date: 2026-03-01
- Related: #16, #14

## Context

`sageRefiner` 在算子编排层（`operator.py`）存在 `torch` 的直接依赖，用于设备探测（CPU/CUDA）。 这会把重依赖的使用面扩散到非核心算法层，增加边界复杂度。

Issue #16 要求在不引入兼容层的前提下，收敛 `torch/transformers` 使用面：

1. 仅保留不可替代调用。
1. 删除冗余路径。
1. 通过测试与文档形成可审阅证据。

## Decision

执行以下边界收敛：

1. 从以下编排层文件删除 `torch` 依赖和设备探测逻辑：

   - `algorithms/recomp_abst/operator.py`
   - `algorithms/recomp_extr/operator.py`
   - `algorithms/reform/operator.py`

1. 统一由核心算法实现层负责设备自动探测：

   - `RECOMPAbstractiveCompressor` / `RECOMPExtractiveCompressor`（已有 `device=None` 自动探测）
   - `AttentionHookExtractor`（新增 `device: str | None`，`None` 时内部使用 `torch.cuda.is_available()`）

1. 新增 issue #16 回归测试，确保：

   - operator 层不再直接 import `torch`/`transformers`
   - 核心算法模块保留必要的 `torch`/`transformers` 依赖
   - operator 压缩异常路径不再回退原文，统一 fail-fast 抛错（覆盖
     RECOMP/REFORM/LongLLMLingua/Provence/LLMLingua2/LongRefiner）

## Consequences

- 重依赖使用边界更清晰：编排层不触碰 `torch/transformers`，核心实现层独占。
- 设备选择策略集中化，减少重复逻辑与未来漂移。
- 删除 operator 侧异常回退兼容路径，压缩失败直接报错。
- 删除 compressor 侧异常回退兼容路径（RECOMP Abstractive 生成失败与空摘要不再降级处理）。
- 未引入 shim/re-export/fallback；调用方直接走收敛后的路径。

## Verification

- `ruff check src/sage_refiner/algorithms/recomp_abst/operator.py src/sage_refiner/algorithms/recomp_extr/operator.py src/sage_refiner/algorithms/reform/operator.py src/sage_refiner/algorithms/reform/model_utils.py src/sage_refiner/algorithms/recomp_abst/compressor.py src/sage_refiner/algorithms/recomp_extr/compressor.py tests/test_issue16_torch_transformers_usage_convergence.py tests/test_issue16_operator_failfast.py tests/test_issue17_compression_extraction_regression.py docs/adr/0003-torch-transformers-usage-convergence.md`
- `pytest -q tests/test_issue16_torch_transformers_usage_convergence.py tests/test_issue16_operator_failfast.py tests/test_issue17_compression_extraction_regression.py`
