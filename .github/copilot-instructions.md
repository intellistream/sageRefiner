# sageRefiner Copilot Instructions

## Scope

- Package: `isage-refiner`, import path `sage_refiner`.
- Layer: **L3** — algorithm library for context refinement/compression; no L4+ dependencies.
- Purpose: Context compression, recomputation, extraction algorithms for LLM/RAG pipelines.

## Polyrepo Context (Important)

SAGE was restructured from a monorepo into a polyrepo. `sageRefiner` is a **standalone L3 algorithm
repo** providing context refinement strategies for RAG and LLM inference pipelines. It integrates
with `sage-libs` interfaces and can be used standalone or via `sage-middleware`.

## Critical rules

- Keep runtime/service-neutral; no L4+ dependencies.
- Do not create new local virtual environments (`venv`/`.venv`); use the existing configured Python
  environment.
- In conda environments, use `python -m pip` (never plain `pip`).
- `_version.py` is the **sole version source**.
- No fallback logic; fail fast.

## Version Source of Truth (Mandatory)

- Only hardcode version in `src/sage_refiner/_version.py`.
- `pyproject.toml` uses `dynamic = ["version"]` with `attr = "sage_refiner._version.__version__"`.
- To bump version: update only `_version.py`.

## Architecture focus

- `algorithms/` — refinement algorithms (recomputation extraction, compression, summarization).
- `config.py` — configuration management.
- `__init__.py` — public API surface.

## Dependencies

- **Depends on**: `isage-common` (L1), `isage-libs` (L3 interfaces).
- **Depended on by**: `sage-refiner-benchmark`, RAG application pipelines.

## Workflow

1. Make minimal changes under `src/sage_refiner/`.
1. Keep public imports stable in `__init__.py`.
1. Run `pytest tests/ -v` and update docs for behavior changes.

## Development setup

```bash
./quickstart.sh       # installs hooks + pip install -e .[dev]
./quickstart.sh --doctor  # diagnose env issues
```

## Git Hooks (Mandatory)

- Never use `git commit --no-verify` or `git push --no-verify`.
- If hooks fail, fix the issue first.

## 🚫 NEVER_CREATE_DOT_VENV_MANDATORY

- 永远不要创建 `.venv` 或 `venv`（无任何例外）。
- NEVER create `.venv`/`venv` in this repository under any circumstance.
- 必须复用当前已配置的非-venv Python 环境（如现有 conda 环境）。
- If any script/task suggests creating a virtualenv, skip that step and continue with the existing
  environment.
