"""CLI for running interfere benchmarks.

Usage:
    uv run python -m server.benchmark_cli --model <path_or_name>
    uv run python -m server.benchmark_cli --model <path> --max-tokens 200 --json
    uv run python -m server.benchmark_cli --model <path> --save results/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from .benchmark import print_summary, run_benchmark


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="interfere-benchmark",
        description="Benchmark interfere inference pipeline",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Max tokens per prompt (default: 100)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warm-up generation (measures cold start)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output results as JSON instead of table",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save JSON results to this directory",
    )

    args = parser.parse_args(argv)

    summary = run_benchmark(
        model_name=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        warm_up=not args.no_warmup,
    )

    if args.output_json:
        print(json.dumps(summary.to_dict(), indent=2))
    else:
        print_summary(summary)

    if args.save:
        save_dir = Path(args.save)
        save_dir.mkdir(parents=True, exist_ok=True)
        model_slug = Path(args.model).name or args.model.replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp}-{model_slug}.json"
        save_path = save_dir / filename
        with open(save_path, "w") as f:
            json.dump(summary.to_dict(), f, indent=2)
        print(f"Results saved to {save_path}")


if __name__ == "__main__":
    main()
