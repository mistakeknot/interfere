"""Code-correctness benchmark for flash-MoE / MLX / cloud models.

Wires LiveCodeBench v6 (primary, April 2026) into the holistic benchmark
harness. SWE-bench Lite support is stubbed — see Sylveste-r8g for the
full runner.

Usage:
    # Dry-run — no MLX, uses a stub model + fixture problems. Fast CI check.
    uv run python -m benchmarks.code_correctness \
        --model=local:qwen3.5-122b \
        --suite=livecodebench-v6 \
        --dry-run

    # Real run — full LCB v6 against 122B MLX baseline.
    uv run python -m benchmarks.code_correctness \
        --model=local:qwen3.5-122b \
        --suite=livecodebench-v6 \
        --limit=200 \
        --output benchmarks/code_correctness_results/

    # Full matrix against holistic results dir (one line per (model, problem))
    uv run python -m benchmarks.code_correctness \
        --models=local:qwen3.5-35b,local:qwen3.5-122b,flash-moe:397b,cloud \
        --suite=livecodebench-v6 \
        --output benchmarks/holistic_results/

Verification (dry-run):
    uv run python -m benchmarks.code_correctness \
        --model=local:qwen3.5-122b \
        --suite=swe-bench-lite --dry-run
    # emits pass@1 = 0/2 on fixture problems without loading MLX.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# Reuse the holistic benchmark's model registry and generators so we only
# maintain config in one place.
from benchmarks.holistic_benchmark import (
    CONFIG_REGISTRY,
    _generate_cloud,
    _generate_flashmoe,
    _generate_mlx,
)
from benchmarks.suites import livecodebench, swe_bench_lite

# ---------------------------------------------------------------------------
# Model aliases — maps CLI names like `local:qwen3.5-122b` to CONFIG_REGISTRY.
# ---------------------------------------------------------------------------

MODEL_ALIASES: dict[str, str] = {
    "local:qwen3.5-9b": "9b",
    "local:qwen3.5-35b": "35b",
    "local:qwen3.5-122b": "122b",
    "local:qwen3.6-27b": "27b-3.6",
    "local:qwen3.6-35b": "35b-3.6",
    "local:qwen3.6-35b-dwq": "35b-3.6-dwq",
    "local:deepseek-v3.2": "deepseek-v3.2",
    "local:glm-5": "glm-5",
    "local:kimi-k2.5": "kimi-k2.5",
    "flash-moe:397b": "flashmoe-q3",
    "flash-moe:397b-q3": "flashmoe-q3",
    "flash-moe:397b-4bit": "flashmoe-4bit",
    "cloud": "cloud",
    "cloud:claude-sonnet-4": "cloud",
}


def resolve_config(model_name: str) -> tuple[str, dict]:
    """Resolve a CLI model identifier to (config_name, config_dict)."""
    config_name = MODEL_ALIASES.get(model_name, model_name)
    if config_name not in CONFIG_REGISTRY:
        available = sorted(set(MODEL_ALIASES) | set(CONFIG_REGISTRY))
        raise ValueError(
            f"Unknown model '{model_name}'. Known aliases: {', '.join(available)}"
        )
    return config_name, CONFIG_REGISTRY[config_name]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ProblemResult:
    suite: str
    model: str
    problem_id: str
    passed: bool
    tests_total: int
    tests_passed: int
    error: str
    exec_elapsed_s: float
    gen_elapsed_s: float
    ttft_s: float
    tokens_generated: int
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SuiteScorecard:
    suite: str
    model: str
    n_problems: int
    n_passed: int
    pass_at_1: float
    median_gen_elapsed_s: float
    median_ttft_s: float
    median_exec_elapsed_s: float
    errors: int
    dry_run: bool
    limit: int | None
    results_file: str
    started_at: str
    finished_at: str


# ---------------------------------------------------------------------------
# Generator dispatch
# ---------------------------------------------------------------------------

StubGenerator = Callable[[str], dict[str, Any]]


def _dispatch_generator(
    config: dict, prompt: str, max_tokens: int, timeout: float
) -> dict[str, Any]:
    """Route a single prompt to the right backend."""
    messages = [{"role": "user", "content": prompt}]
    backend = config["backend"]
    if backend == "mlx":
        return _generate_mlx(config["model"], messages, max_tokens, timeout=timeout)
    if backend == "flash-moe":
        return _generate_flashmoe(config, messages, max_tokens, timeout=timeout)
    if backend == "cloud":
        return _generate_cloud(config["model"], messages, max_tokens, timeout=timeout)
    raise ValueError(f"Unknown backend: {backend}")


def _stub_generator(prompt: str) -> dict[str, Any]:
    """Canned response that never passes a real test — used for --dry-run.

    Emits an obviously-wrong but syntactically valid Python snippet, so the
    executor still exercises extract → subprocess → normalise → compare.
    """
    stub = (
        "```python\n"
        "import sys\n"
        "data = sys.stdin.read()\n"
        "# stub solution — intentionally incorrect\n"
        "print('__STUB__')\n"
        "```"
    )
    return {
        "output_text": stub,
        "tokens_generated": 24,
        "elapsed_s": 0.001,
        "ttft_s": 0.0001,
        "gen_tps": 24000.0,
        "peak_mem_gb": 0.0,
        "timed_out": False,
    }


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

LCB_PROMPT = """You are given a competitive programming problem. Write a \
complete Python 3 program that reads input from standard input and prints \
output to standard output. Include only the program, no explanation.

Problem: {title}

{body}

{starter}
"""

SWEBENCH_PROMPT = """You are fixing a bug in a Python repository. Produce a \
unified diff that resolves the failing test(s). Output only the diff, no \
explanation.

Repository: {repo}
Base commit: {base_commit}
Problem statement:
{body}

Failing tests: {fail_to_pass}
"""


def _format_lcb_prompt(problem: livecodebench.LCBProblem) -> str:
    starter = (
        f"\nStarter code (extend this):\n```python\n{problem.starter_code}\n```\n"
        if problem.starter_code.strip()
        else ""
    )
    return LCB_PROMPT.format(
        title=problem.question_title or problem.question_id,
        body=problem.question_content,
        starter=starter,
    )


def _format_swebench_prompt(problem: swe_bench_lite.SWEBenchProblem) -> str:
    return SWEBENCH_PROMPT.format(
        repo=problem.repo,
        base_commit=problem.base_commit,
        body=problem.problem_statement,
        fail_to_pass=", ".join(problem.fail_to_pass) or "(none listed)",
    )


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def run_suite(
    suite: str,
    model_name: str,
    output_dir: Path,
    limit: int | None = None,
    timeout: float = 300.0,
    dry_run: bool = False,
    generator: StubGenerator | None = None,
) -> SuiteScorecard:
    """Run a single (suite, model) pair and return an aggregated scorecard."""
    config_name, config = resolve_config(model_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "code_correctness.jsonl"
    summary_file = output_dir / "code_correctness_summary.json"

    # Allow resumption by re-using prior (suite, model, problem_id) records.
    cached: dict[str, dict] = {}
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("suite") == suite and r.get("model") == model_name:
                    cached[r["problem_id"]] = r

    if suite == "livecodebench-v6":
        problems = livecodebench.load_problems(limit=limit, dry_run=dry_run)
        fmt_prompt = _format_lcb_prompt
        exec_fn = livecodebench.run_problem
        problem_id_attr = "question_id"
    elif suite == "swe-bench-lite":
        problems = swe_bench_lite.load_problems(limit=limit, dry_run=dry_run)
        fmt_prompt = _format_swebench_prompt
        exec_fn = swe_bench_lite.run_problem
        problem_id_attr = "instance_id"
    else:
        raise ValueError(
            f"Unknown suite '{suite}'. Choose from: livecodebench-v6, swe-bench-lite"
        )

    started_at = datetime.now(timezone.utc).isoformat()
    gen_times: list[float] = []
    ttfts: list[float] = []
    exec_times: list[float] = []
    passes = 0
    errors = 0
    results: list[ProblemResult] = []

    print(
        f"Running {suite} × {model_name} ({config.get('label', config_name)}) — "
        f"{len(problems)} problems, dry_run={dry_run}"
    )

    for i, problem in enumerate(problems, start=1):
        problem_id = getattr(problem, problem_id_attr)
        if problem_id in cached:
            c = cached[problem_id]
            print(
                f"  [{i}/{len(problems)}] {problem_id} — cached "
                f"({'PASS' if c.get('passed') else 'FAIL'})"
            )
            if c.get("passed"):
                passes += 1
            gen_times.append(c.get("gen_elapsed_s", 0.0))
            ttfts.append(c.get("ttft_s", 0.0))
            exec_times.append(c.get("exec_elapsed_s", 0.0))
            results.append(
                ProblemResult(
                    **{k: c[k] for k in c if k in ProblemResult.__dataclass_fields__}
                )
            )
            continue

        prompt_text = fmt_prompt(problem)

        try:
            if generator is not None:
                gen = generator(prompt_text)
            elif dry_run:
                gen = _stub_generator(prompt_text)
            else:
                gen = _dispatch_generator(
                    config, prompt_text, max_tokens=2048, timeout=timeout
                )
        except Exception as e:
            print(f"  [{i}/{len(problems)}] {problem_id} — GENERATION ERROR: {e}")
            errors += 1
            continue

        try:
            exec_res = exec_fn(problem, gen["output_text"])
        except Exception as e:
            print(f"  [{i}/{len(problems)}] {problem_id} — EXEC ERROR: {e}")
            errors += 1
            continue

        passed = bool(exec_res.get("passed", False))
        if passed:
            passes += 1
        gen_times.append(gen.get("elapsed_s", 0.0))
        ttfts.append(gen.get("ttft_s", 0.0))
        exec_times.append(exec_res.get("elapsed_s", 0.0))

        pr = ProblemResult(
            suite=suite,
            model=model_name,
            problem_id=problem_id,
            passed=passed,
            tests_total=exec_res.get("tests_total", 0),
            tests_passed=exec_res.get("tests_passed", 0),
            error=exec_res.get("error", "") or "",
            exec_elapsed_s=exec_res.get("elapsed_s", 0.0),
            gen_elapsed_s=gen.get("elapsed_s", 0.0),
            ttft_s=gen.get("ttft_s", 0.0),
            tokens_generated=gen.get("tokens_generated", 0),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        results.append(pr)
        with open(results_file, "a") as f:
            f.write(json.dumps(pr.to_dict()) + "\n")

        tag = "PASS" if passed else "FAIL"
        print(
            f"  [{i}/{len(problems)}] {problem_id} — {tag} "
            f"({exec_res.get('tests_passed', 0)}/{exec_res.get('tests_total', 0)})"
        )

    finished_at = datetime.now(timezone.utc).isoformat()
    n = len(results)
    scorecard = SuiteScorecard(
        suite=suite,
        model=model_name,
        n_problems=n,
        n_passed=passes,
        pass_at_1=(passes / n) if n else 0.0,
        median_gen_elapsed_s=_median(gen_times),
        median_ttft_s=_median(ttfts),
        median_exec_elapsed_s=_median(exec_times),
        errors=errors,
        dry_run=dry_run,
        limit=limit,
        results_file=str(results_file),
        started_at=started_at,
        finished_at=finished_at,
    )

    # Merge into summary_file (append-or-replace per (suite, model))
    existing: list[dict] = []
    if summary_file.exists():
        try:
            existing = json.loads(summary_file.read_text())
        except json.JSONDecodeError:
            existing = []
    existing = [
        e
        for e in existing
        if not (e.get("suite") == suite and e.get("model") == model_name)
    ]
    existing.append(asdict(scorecard))
    summary_file.write_text(json.dumps(existing, indent=2) + "\n")

    print(
        f"\n  {suite} × {model_name}: pass@1 = {scorecard.n_passed}/{scorecard.n_problems} "
        f"({scorecard.pass_at_1:.2%})  errors={errors}"
    )
    return scorecard


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="code_correctness",
        description="Code-correctness benchmark (LiveCodeBench v6 + SWE-bench Lite)",
    )
    parser.add_argument(
        "--model",
        help="Single model alias, e.g. local:qwen3.5-122b, flash-moe:397b, cloud",
    )
    parser.add_argument(
        "--models",
        help="Comma-separated model aliases (alternative to --model)",
    )
    parser.add_argument(
        "--suite",
        default="livecodebench-v6",
        choices=["livecodebench-v6", "swe-bench-lite"],
        help="Benchmark suite (default: livecodebench-v6)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/code_correctness_results"),
        help="Output directory for JSONL + summary (default: benchmarks/code_correctness_results)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Cap problems per suite (debug)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-prompt generation timeout seconds (default: 300)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use stub generator + fixture problems; skip MLX load",
    )

    args = parser.parse_args(argv)
    if not args.model and not args.models:
        parser.error("Must provide --model or --models")

    models = (
        [args.model]
        if args.model
        else [m.strip() for m in args.models.split(",") if m.strip()]
    )

    # In dry-run with swe-bench-lite we cap at 2 to match the fixture exactly.
    effective_limit = args.limit
    if args.dry_run and effective_limit is None:
        effective_limit = 2

    scorecards: list[SuiteScorecard] = []
    for model_name in models:
        sc = run_suite(
            suite=args.suite,
            model_name=model_name,
            output_dir=args.output,
            limit=effective_limit,
            timeout=args.timeout,
            dry_run=args.dry_run,
        )
        scorecards.append(sc)

    # Terminal summary table
    print("\n=== code_correctness scorecard ===")
    print(f"{'suite':<20} {'model':<28} {'pass@1':<10} {'n':<6} {'errors':<6}")
    for sc in scorecards:
        print(
            f"{sc.suite:<20} {sc.model:<28} "
            f"{sc.pass_at_1:.2%} ({sc.n_passed}/{sc.n_problems})".ljust(40)
            + f" {sc.n_problems:<6} {sc.errors:<6}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
