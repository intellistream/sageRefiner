# ADR 0001: Refiner algorithm layer and inference layer decoupling

## Status

Accepted

## Context

Issue `intellistream/sageRefiner#15` requires that refiner keeps algorithm logic only and does not carry inference-service binding conventions.

Audit findings:

- `LongLLMLinguaCompressor` exposed `open_api_config` and passed it into `PromptCompressor` initialization.
- `LongRefiner` prompt template contained OpenAI-specific branch flags (`is_openai`) and branch logic.

These paths mixed provider-specific inference concerns into algorithm modules.

## Decision

1. Remove provider-specific API binding from `LongLLMLinguaCompressor`.
   - Delete `open_api_config` constructor parameter.
   - Delete forwarding of `open_api_config` to `PromptCompressor`.
2. Remove OpenAI-specific prompt-branch logic from `LongRefiner` prompt template.
   - Delete `is_openai` state and all dependent branches.
3. Keep a single canonical algorithm path and fail-fast behavior in algorithm modules.
4. Add regression tests to lock this boundary.

## Consequences

- Algorithm modules remain provider-neutral and easier to reason about.
- Callers must bind inference/provider behavior outside refiner algorithm modules.
- Boundary assumptions are now explicitly test-covered.
