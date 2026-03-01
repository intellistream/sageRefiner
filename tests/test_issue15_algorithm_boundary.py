from __future__ import annotations

import ast
from pathlib import Path


def test_longllmlingua_compressor_init_has_no_open_api_config() -> None:
    compressor_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "sage_refiner"
        / "algorithms"
        / "longllmlingua"
        / "compressor.py"
    )
    source = compressor_path.read_text(encoding="utf-8")
    module = ast.parse(source)

    init_args: list[str] = []
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "LongLLMLinguaCompressor":
            for class_node in node.body:
                if isinstance(class_node, ast.FunctionDef) and class_node.name == "__init__":
                    init_args = [arg.arg for arg in class_node.args.args]
                    break
            break

    assert init_args, "LongLLMLinguaCompressor.__init__ not found"
    assert "open_api_config" not in init_args


def test_prompt_template_has_no_openai_branch_state() -> None:
    prompt_template_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "sage_refiner"
        / "algorithms"
        / "LongRefiner"
        / "prompt_template.py"
    )
    source = prompt_template_path.read_text(encoding="utf-8")
    assert "is_openai" not in source
