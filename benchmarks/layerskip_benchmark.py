#!/usr/bin/env python3
"""Benchmark LayerSkip self-speculative decoding vs standard generation.

Usage:
    uv run python benchmarks/layerskip_benchmark.py \
        --model ~/Models/Qwen3.5-122B-A10B-4bit \
        --prompts benchmarks/prompts/coding_eval.json \
        --output benchmarks/results_layerskip.tsv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LayerSkip benchmark")
    p.add_argument("--model", required=True, help="Path to MLX model directory")
    p.add_argument("--prompts", required=True, help="Path to prompts JSON file")
    p.add_argument("--output", required=True, help="Output TSV path")
    p.add_argument(
        "--exit-layers",
        default="16,24,32",
        help="Comma-separated exit layer values to sweep (default: 16,24,32)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Confidence threshold (default: 0.95)",
    )
    p.add_argument("--max-tokens", type=int, default=200, help="Max tokens per prompt")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    exit_layers = [int(x) for x in args.exit_layers.split(",")]

    with open(args.prompts) as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} prompts")

    # Load model
    import mlx.core as mx
    from mlx_lm import load, generate

    print(f"Loading model: {args.model}")
    t0 = time.monotonic()
    model, tokenizer = load(args.model)
    load_time = time.monotonic() - t0
    num_layers = len(model.model.layers)
    print(f"Loaded in {load_time:.1f}s — {num_layers} layers")

    # Import PoC
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
    from experiments.early_exit import self_speculative_generate

    results = []

    for prompt_info in prompts:
        prompt_id = prompt_info["id"]
        text = prompt_info["messages"][-1]["content"]
        max_tokens = min(prompt_info.get("max_tokens", 200), args.max_tokens)

        # Format as chat
        if hasattr(tokenizer, "apply_chat_template"):
            formatted = tokenizer.apply_chat_template(
                prompt_info["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted = text

        # --- Baseline: standard generate ---
        print(f"\n[{prompt_id}] standard generate...", end="", flush=True)
        mx.metal.reset_peak_memory()
        t_start = time.monotonic()
        baseline_output = generate(
            model, tokenizer, prompt=formatted, max_tokens=max_tokens, verbose=False
        )
        t_end = time.monotonic()

        baseline_tokens = len(tokenizer.encode(baseline_output))
        baseline_elapsed = t_end - t_start
        baseline_tps = baseline_tokens / baseline_elapsed if baseline_elapsed > 0 else 0
        print(f" {baseline_tps:.1f} tok/s, {baseline_tokens} tokens")

        results.append(
            {
                "prompt_id": prompt_id,
                "method": "standard",
                "exit_layer": num_layers,
                "threshold": "-",
                "tokens": baseline_tokens,
                "accepted": "-",
                "verified": "-",
                "acceptance_rate": "-",
                "tok_per_sec": round(baseline_tps, 2),
                "elapsed_s": round(baseline_elapsed, 3),
                "output_preview": baseline_output[:150].replace("\n", "\\n"),
            }
        )

        # --- LayerSkip: sweep exit layers ---
        for exit_layer in exit_layers:
            if exit_layer >= num_layers:
                print(f"  [skip] exit_layer={exit_layer} >= num_layers={num_layers}")
                continue

            print(
                f"  [{prompt_id}] layerskip exit={exit_layer} threshold={args.threshold}...",
                end="",
                flush=True,
            )
            result = self_speculative_generate(
                model,
                tokenizer,
                prompt=formatted,
                exit_layer=exit_layer,
                confidence_threshold=args.threshold,
                max_tokens=max_tokens,
            )
            print(
                f" {result['tok_per_sec']:.1f} tok/s, "
                f"accept={result['acceptance_rate']:.1%}, "
                f"{result['tokens']} tokens"
            )

            results.append(
                {
                    "prompt_id": prompt_id,
                    "method": "layerskip",
                    "exit_layer": exit_layer,
                    "threshold": args.threshold,
                    "tokens": result["tokens"],
                    "accepted": result["accepted"],
                    "verified": result["verified"],
                    "acceptance_rate": round(result["acceptance_rate"], 4),
                    "tok_per_sec": round(result["tok_per_sec"], 2),
                    "elapsed_s": result["elapsed_s"],
                    "output_preview": result["text"][:150].replace("\n", "\\n"),
                }
            )

    # Write TSV
    columns = list(results[0].keys())
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\t".join(columns) + "\n")
        for row in results:
            f.write("\t".join(str(row.get(c, "")) for c in columns) + "\n")

    print(f"\nResults written to {args.output}")

    # Summary
    print("\n--- Summary ---")
    print(
        f"{'prompt_id':<20} {'method':<12} {'exit':<6} "
        f"{'tok/s':>8} {'accept%':>8} {'tokens':>6}"
    )
    for r in results:
        accept_str = (
            f"{r['acceptance_rate']:.1%}" if r["acceptance_rate"] != "-" else "-"
        )
        print(
            f"{r['prompt_id']:<20} {r['method']:<12} {r['exit_layer']:<6} "
            f"{r['tok_per_sec']:>8} {accept_str:>8} {r['tokens']:>6}"
        )


if __name__ == "__main__":
    main()
