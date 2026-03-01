from __future__ import annotations

import ast
from pathlib import Path


def _collect_imported_modules(file_path: Path) -> set[str]:
    source = file_path.read_text(encoding="utf-8")
    module = ast.parse(source)

    imported_modules: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    return imported_modules


def _contains_module(imported_modules: set[str], target: str) -> bool:
    return any(mod == target or mod.startswith(f"{target}.") for mod in imported_modules)


def test_operator_layer_has_no_torch_or_transformers_imports() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    operator_files = [
        repo_root / "src" / "sage_refiner" / "algorithms" / "recomp_abst" / "operator.py",
        repo_root / "src" / "sage_refiner" / "algorithms" / "recomp_extr" / "operator.py",
        repo_root / "src" / "sage_refiner" / "algorithms" / "reform" / "operator.py",
    ]

    for file_path in operator_files:
        imported_modules = _collect_imported_modules(file_path)
        assert not _contains_module(imported_modules, "torch"), (
            f"Unexpected torch import in operator layer: {file_path}"
        )
        assert not _contains_module(imported_modules, "transformers"), (
            f"Unexpected transformers import in operator layer: {file_path}"
        )


def test_core_algorithm_modules_keep_heavy_dependency_imports() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    core_files = [
        repo_root / "src" / "sage_refiner" / "algorithms" / "recomp_abst" / "compressor.py",
        repo_root / "src" / "sage_refiner" / "algorithms" / "recomp_extr" / "compressor.py",
        repo_root / "src" / "sage_refiner" / "algorithms" / "reform" / "model_utils.py",
    ]

    for file_path in core_files:
        imported_modules = _collect_imported_modules(file_path)
        assert _contains_module(imported_modules, "torch"), (
            f"Expected torch import missing in core module: {file_path}"
        )

    transformers_holders = [
        repo_root / "src" / "sage_refiner" / "algorithms" / "recomp_abst" / "compressor.py",
        repo_root / "src" / "sage_refiner" / "algorithms" / "recomp_extr" / "compressor.py",
        repo_root / "src" / "sage_refiner" / "algorithms" / "reform" / "model_utils.py",
    ]

    for file_path in transformers_holders:
        imported_modules = _collect_imported_modules(file_path)
        assert _contains_module(imported_modules, "transformers"), (
            f"Expected transformers import missing in core module: {file_path}"
        )
