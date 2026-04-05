#!/usr/bin/env python3
"""Benchmark MLX and flash-moe models on coding evaluation prompts.

Usage:
    # MLX model (122B, 35B, etc.)
    uv run python benchmarks/model_benchmark.py \
        --model ~/Models/Qwen3.5-122B-A10B-4bit \
        --prompts benchmarks/prompts/coding_eval.json \
        --output benchmarks/results_122b.tsv

    # Flash-MoE model (397B)
    uv run python benchmarks/model_benchmark.py \
        --flashmoe-binary ~/projects/flash-moe/metal_infer/infer \
        --flashmoe-model ~/Models/flash_mlx_4bit \
        --prompts benchmarks/prompts/coding_eval.json \
        --output benchmarks/results_397b.tsv \
        --q3-experts --cache-io-split 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark MLX/flash-moe models")
    # Model source (one of these required)
    p.add_argument("--model", help="Path to MLX model directory")
    p.add_argument("--flashmoe-binary", help="Path to flash-moe infer binary")
    p.add_argument("--flashmoe-model", help="Path to flash-moe model directory")
    # Prompts and output
    p.add_argument("--prompts", required=True, help="Path to prompts JSON file")
    p.add_argument("--output", required=True, help="Output TSV path")
    # MLX options
    p.add_argument(
        "--kv-bits", type=int, default=4, help="KV cache quantization bits (default: 4)"
    )
    # Flash-MoE options
    p.add_argument("--q3-experts", action="store_true")
    p.add_argument("--cache-io-split", type=int, default=4)
    p.add_argument("--malloc-cache", type=int, default=0)
    # General
    p.add_argument(
        "--warmup", type=int, default=1, help="Warmup runs per prompt (default: 1)"
    )
    p.add_argument(
        "--runs", type=int, default=1, help="Measured runs per prompt (default: 1)"
    )
    return p.parse_args(argv)


def _run_mlx_benchmark(args: argparse.Namespace, prompts: list[dict]) -> list[dict]:
    """Benchmark an MLX model using mlx-lm."""
    import mlx.core as mx
    from mlx_lm import load, generate

    print(f"Loading MLX model: {args.model}")
    t0 = time.monotonic()
    model, tokenizer = load(args.model)
    load_time = time.monotonic() - t0
    print(f"Model loaded in {load_time:.1f}s")

    # Memory after load
    mem_after_load = mx.metal.get_active_memory() / 1e9
    print(f"Metal memory after load: {mem_after_load:.2f} GB")

    results = []
    model_name = os.path.basename(args.model)

    for prompt_info in prompts:
        prompt_id = prompt_info["id"]
        messages = prompt_info["messages"]
        max_tokens = prompt_info.get("max_tokens", 512)

        # Format as chat
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = messages[-1]["content"]

        # Warmup runs
        for w in range(args.warmup):
            print(f"  [{prompt_id}] warmup {w+1}/{args.warmup}...", end="", flush=True)
            _ = generate(
                model,
                tokenizer,
                prompt=text,
                max_tokens=min(max_tokens, 64),
                verbose=False,
            )
            print(" done")

        # Measured runs
        for r in range(args.runs):
            print(f"  [{prompt_id}] run {r+1}/{args.runs}...", end="", flush=True)
            mx.metal.reset_peak_memory()

            t_start = time.monotonic()
            output = generate(
                model,
                tokenizer,
                prompt=text,
                max_tokens=max_tokens,
                verbose=False,
            )
            t_end = time.monotonic()

            elapsed = t_end - t_start
            peak_mem = mx.metal.get_peak_memory() / 1e9
            active_mem = mx.metal.get_active_memory() / 1e9

            # Count tokens generated
            out_tokens = tokenizer.encode(output)
            num_gen_tokens = len(out_tokens)

            # Approximate prompt tokens
            prompt_tokens = tokenizer.encode(text)
            num_prompt_tokens = len(prompt_tokens)

            gen_tps = num_gen_tokens / elapsed if elapsed > 0 else 0
            # TTFT is hard to measure exactly without hooks; approximate as
            # total_time * (1 token / total tokens)
            ttft_approx_ms = (elapsed / max(num_gen_tokens, 1)) * 1000

            result = {
                "model": model_name,
                "backend": "mlx",
                "prompt_id": prompt_id,
                "run": r + 1,
                "prompt_tokens": num_prompt_tokens,
                "gen_tokens": num_gen_tokens,
                "elapsed_s": round(elapsed, 3),
                "gen_tps": round(gen_tps, 2),
                "ttft_approx_ms": round(ttft_approx_ms, 1),
                "peak_mem_gb": round(peak_mem, 2),
                "active_mem_gb": round(active_mem, 2),
                "output_preview": output[:200].replace("\n", "\\n"),
            }
            results.append(result)
            print(
                f" {gen_tps:.1f} tok/s, {peak_mem:.1f}GB peak, {num_gen_tokens} tokens"
            )

    return results


def _run_flashmoe_benchmark(
    args: argparse.Namespace, prompts: list[dict]
) -> list[dict]:
    """Benchmark flash-moe model via subprocess binary."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
    from flashmoe_worker import FlashMoeWorker

    binary = os.path.expanduser(args.flashmoe_binary)
    model_path = os.path.expanduser(args.flashmoe_model)

    print(f"Starting flash-moe: {binary}")
    worker = FlashMoeWorker(
        binary_path=binary,
        model_path=model_path,
        q3_experts=args.q3_experts,
        cache_io_split=args.cache_io_split,
        malloc_cache=args.malloc_cache,
    )
    worker.start()
    print(f"Flash-MoE ready on port {worker._port}")

    results = []
    model_name = os.path.basename(model_path) + ("-q3" if args.q3_experts else "")

    try:
        for prompt_info in prompts:
            prompt_id = prompt_info["id"]
            messages = prompt_info["messages"]
            max_tokens = prompt_info.get("max_tokens", 512)

            # Warmup
            for w in range(args.warmup):
                print(
                    f"  [{prompt_id}] warmup {w+1}/{args.warmup}...", end="", flush=True
                )
                tokens = list(
                    worker.generate(
                        messages=messages,
                        max_tokens=min(max_tokens, 64),
                        timeout=600.0,
                    )
                )
                print(f" done ({len(tokens)} chunks)")

            # Measured runs
            for r in range(args.runs):
                print(f"  [{prompt_id}] run {r+1}/{args.runs}...", end="", flush=True)

                t_start = time.monotonic()
                output_chunks = list(
                    worker.generate(
                        messages=messages,
                        max_tokens=max_tokens,
                        timeout=600.0,
                    )
                )
                t_end = time.monotonic()

                elapsed = t_end - t_start
                output_text = "".join(output_chunks)
                metrics = worker.last_generation_metrics

                gen_tps = metrics.get("generation_tps", 0)
                if gen_tps == 0 and elapsed > 0:
                    gen_tps = (
                        metrics.get("tokens_generated", len(output_chunks)) / elapsed
                    )

                result = {
                    "model": model_name,
                    "backend": "flash-moe",
                    "prompt_id": prompt_id,
                    "run": r + 1,
                    "prompt_tokens": 0,  # flash-moe doesn't report this easily
                    "gen_tokens": metrics.get("tokens_generated", len(output_chunks)),
                    "elapsed_s": round(elapsed, 3),
                    "gen_tps": round(gen_tps, 2),
                    "ttft_approx_ms": 0,  # not measurable via HTTP proxy
                    "peak_mem_gb": metrics.get("peak_memory_gb", 0),
                    "active_mem_gb": 0,
                    "output_preview": output_text[:200].replace("\n", "\\n"),
                }
                results.append(result)
                worker.reset_consecutive_crashes()
                print(
                    f" {gen_tps:.1f} tok/s, {metrics.get('tokens_generated', '?')} tokens"
                )
    finally:
        worker.shutdown()

    return results


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # Load prompts
    with open(args.prompts) as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} prompts from {args.prompts}")

    # Route to backend
    if args.model:
        results = _run_mlx_benchmark(args, prompts)
    elif args.flashmoe_binary and args.flashmoe_model:
        results = _run_flashmoe_benchmark(args, prompts)
    else:
        print("Error: specify --model (MLX) or --flashmoe-binary + --flashmoe-model")
        sys.exit(1)

    # Write TSV
    if not results:
        print("No results collected!")
        sys.exit(1)

    columns = list(results[0].keys())
    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as f:
        f.write("\t".join(columns) + "\n")
        for row in results:
            f.write("\t".join(str(row.get(c, "")) for c in columns) + "\n")

    print(f"\nResults written to {output_path}")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Runs per prompt: {args.runs}")
    print(f"  Total results: {len(results)}")

    # Summary table
    print("\n--- Summary ---")
    print(
        f"{'prompt_id':<20} {'gen_tps':>10} {'gen_tokens':>10} {'elapsed_s':>10} {'peak_mem_gb':>12}"
    )
    for r in results:
        print(
            f"{r['prompt_id']:<20} {r['gen_tps']:>10.2f} {r['gen_tokens']:>10} "
            f"{r['elapsed_s']:>10.3f} {r['peak_mem_gb']:>12.2f}"
        )


if __name__ == "__main__":
    main()
