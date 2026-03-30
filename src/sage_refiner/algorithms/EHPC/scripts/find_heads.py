#!/usr/bin/env python3
"""
EHPC Head Selection Script
==========================

为新模型找到最佳的 Evaluator Heads。

使用方法:
    python -m sage_refiner.algorithms.EHPC.scripts.find_heads \
        --model "your-model-name" \
        --context_length 16000 \
        --num_heads 8

输出:
    - 最佳层索引 (best_layer)
    - 最佳 heads 列表 (best_heads)
    - 可选的 heatmap 图片
"""

import argparse
import json
import logging

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Find best Evaluator Heads for EHPC")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path (e.g., meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=16000,
        help="Context length for needle probe (default: 16000)",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="Number of heads to select (default: 8)",
    )
    parser.add_argument(
        "--depths",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        help="Comma-separated depths for needle insertion (default: 0.1-0.9)",
    )
    parser.add_argument(
        "--context_path",
        type=str,
        default=None,
        help="Path to background text files (glob pattern, e.g., 'texts/*.txt')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ehpc_config.json",
        help="Output file for configuration (default: ehpc_config.json)",
    )
    parser.add_argument(
        "--save_heatmap",
        action="store_true",
        help="Save attention heatmap visualization",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (default: cuda)",
    )

    args = parser.parse_args()

    # Parse depths
    depths = [float(d.strip()) for d in args.depths.split(",")]

    print("=" * 60)
    print("EHPC Evaluator Heads Selection")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Context Length: {args.context_length}")
    print(f"Depths: {depths}")
    print(f"Num Heads to Select: {args.num_heads}")
    print("=" * 60)

    # Load model with EHPC patches
    print("\n[1/4] Loading model with EHPC patches...")
    from sage_refiner.algorithms.EHPC.models.gemfilter import load_ehpc_model

    model, tokenizer = load_ehpc_model(
        args.model,
        torch_dtype=torch.float16,
        flash_attention_2=True,
    )
    print(
        f"Model loaded: {model.config.num_hidden_layers} layers, "
        f"{model.config.num_attention_heads} heads per layer"
    )

    # Create probe
    print("\n[2/4] Creating Needle Probe...")
    from sage_refiner.algorithms.EHPC.probe import NeedleProbe

    probe = NeedleProbe(model, tokenizer, device=args.device)

    # Run probe
    print("\n[3/4] Running Needle-in-a-Haystack probe...")
    print("This may take a few minutes depending on model size and context length.")

    # 使用所有层来找最佳层
    num_layers = model.config.num_hidden_layers
    select_layer_idx = num_layers - 1  # 先用最后一层来收集所有层的 scores

    best_layer, best_heads = probe.find_evaluator_heads(
        context_path=args.context_path or "*.txt",
        context_length=args.context_length,
        depths=depths,
        select_layer_idx=select_layer_idx,
        num_heads_to_select=args.num_heads,
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Best Layer: {best_layer}")
    print(f"Best Heads: {best_heads}")
    print("=" * 60)

    # Save configuration
    print(f"\n[4/4] Saving configuration to {args.output}...")
    config = {
        "model_name": args.model,
        "layer": best_layer,
        "heads": best_heads,
        "topk": 2048,
        "window_size": 32,
        "pool": "avg_pool",
        "kernel_size": 4,
        "_probe_params": {
            "context_length": args.context_length,
            "depths": depths,
            "num_heads": args.num_heads,
        },
    }

    with open(args.output, "w") as f:
        json.dump(config, f, indent=2)

    print("Configuration saved!")

    # Print usage example
    print("\n" + "=" * 60)
    print("HOW TO USE")
    print("=" * 60)
    print(
        f"""
# 方法 1: 直接使用配置文件
from sage_refiner.algorithms.EHPC import EHPCCompressor, EHPCConfig
import json

with open('{args.output}') as f:
    config_dict = json.load(f)
config = EHPCConfig.from_dict(config_dict)

# 方法 2: 手动指定
config = EHPCConfig(
    layer={best_layer},
    heads={best_heads},
    topk=2048,
    window_size=32,
)
"""
    )

    # Optional: save heatmap
    if args.save_heatmap:
        print("\nSaving attention heatmap...")
        # This would require storing the scores during probe
        print("(Heatmap saving not implemented in this version)")


if __name__ == "__main__":
    main()
