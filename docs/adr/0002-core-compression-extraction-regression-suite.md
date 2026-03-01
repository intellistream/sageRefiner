# ADR-0002: Core Compression/Extraction Regression Suite

- Status: Accepted
- Date: 2026-03-01
- Related: #17, #14

## Context

`sageRefiner` 需要补齐关键算法回归保障，覆盖“压缩 + 抽取”的核心执行路径与边界条件，避免后续重构在无感知情况下破坏行为合同。

Issue #17 要求：

1. 覆盖压缩/抽取核心流程与边界条件。
2. 不引入过渡兼容层。
3. 形成可回归、可审阅的测试与说明。

## Decision

新增 issue #17 回归测试文件，聚焦两个高风险路径：

1. **LongLLMLingua（压缩）**
   - 空上下文边界返回零统计。
   - 空问题 fail-fast（`ValueError`）。
   - 压缩参数合同验证（`rank_method=longllmlingua`、默认 `condition_compare=True`、`rate` clamp、RAG 包装参数透传）。

2. **RECOMP Extractive（抽取）**
   - 阈值过滤全空时执行 top-k 选择路径。
   - 空上下文边界返回空结果统计。
   - score→select→compress 主流程回归，验证选择索引与拼接输出。

## Consequences

- 在不加载真实大模型的前提下，回归测试覆盖核心行为合同。
- 后续若修改压缩参数映射、边界处理或抽取选择逻辑，将被测试及时拦截。
- 未新增兼容层；行为约束通过测试直接体现。

## Verification

- `ruff check tests/test_issue17_compression_extraction_regression.py docs/adr/0002-core-compression-extraction-regression-suite.md`
- `pytest -q tests/test_issue17_compression_extraction_regression.py`
