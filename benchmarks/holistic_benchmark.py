#!/usr/bin/env python3
"""Holistic benchmark for interfer: quality + latency + memory + reliability.

Three-stage pipeline:
  1. GENERATE — run each prompt through each model config, save outputs
  2. EXECUTE — run Codex exec on code outputs for pass/fail
  3. JUDGE — run Claude -p for qualitative scoring on a rubric

Each stage is independently re-runnable. Stage 1 is the expensive part
(hours of GPU time). Stages 2-3 are cheap and can be re-run with improved
rubrics without regenerating outputs.

Usage:
    # Stage 1: Generate outputs (slow — runs inference)
    uv run python benchmarks/holistic_benchmark.py generate \
        --prompts benchmarks/prompts/holistic_eval.json \
        --output benchmarks/holistic_results/ \
        --configs 35b,122b,flashmoe-q3,flashmoe-4bit,cloud

    # Stage 2: Execute code tests (fast)
    uv run python benchmarks/holistic_benchmark.py execute \
        --results benchmarks/holistic_results/

    # Stage 3: Judge with Claude (fast, costs ~$2-5)
    uv run python benchmarks/holistic_benchmark.py judge \
        --results benchmarks/holistic_results/

    # Stage 4: Generate scorecard
    uv run python benchmarks/holistic_benchmark.py report \
        --results benchmarks/holistic_results/

    # All stages sequentially
    uv run python benchmarks/holistic_benchmark.py all \
        --prompts benchmarks/prompts/holistic_eval.json \
        --output benchmarks/holistic_results/ \
        --configs 35b,flashmoe-q3,cloud
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    """Output from a single model × prompt run."""

    config: str
    prompt_id: str
    category: str
    difficulty: str
    output_text: str
    tokens_generated: int
    elapsed_s: float
    ttft_s: float
    gen_tps: float
    peak_mem_gb: float
    thermal_start: str
    thermal_end: str
    timestamp: str
    run: int = 1
    timed_out: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecResult:
    """Result from executing generated code."""

    config: str
    prompt_id: str
    passed: bool
    error: str
    exec_time_s: float


@dataclass
class JudgeResult:
    """Result from Claude judge scoring."""

    config: str
    prompt_id: str
    correctness: int  # 0-5
    completeness: int  # 0-5
    code_quality: int  # 0-5
    instruction_following: int  # 0-5
    reasoning: str


@dataclass
class ScoreCard:
    """Aggregated scorecard for a single config."""

    config: str
    # Quality
    exec_pass_rate: float  # fraction of executable prompts that passed
    judge_correctness_avg: float
    judge_completeness_avg: float
    judge_quality_avg: float
    judge_instruction_avg: float
    judge_composite: float  # weighted average
    # Speed
    median_tps: float
    p5_tps: float
    p95_tps: float
    median_ttft_s: float
    # Resource
    peak_mem_gb: float
    thermal_end: str
    # Reliability
    error_count: int
    total_prompts: int


# ---------------------------------------------------------------------------
# Config registry — maps short names to model load instructions
# ---------------------------------------------------------------------------

CONFIG_REGISTRY: dict[str, dict[str, Any]] = {
    "0.5b": {
        "backend": "mlx",
        "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "label": "Qwen2.5-0.5B-4bit",
        "description": "Tiny smoke-test model (cached)",
    },
    "3b": {
        "backend": "mlx",
        "model": "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "label": "Qwen2.5-3B-4bit",
        "description": "Small dense model (cached)",
    },
    "0.8b": {
        "backend": "mlx",
        "model": "mlx-community/Qwen3.5-0.8B-OptiQ-4bit",
        "label": "Qwen3.5-0.8B-4bit",
        "description": "Tiny dense model, draft-size reference (weights not cached)",
    },
    "35b": {
        "backend": "mlx",
        "model": "mlx-community/Qwen3.5-35B-A3B-4bit",
        "label": "Qwen3.5-35B-A3B-4bit",
        "description": "MoE 35B, 3B active — C2 routing tier",
    },
    "122b": {
        "backend": "mlx",
        "model": "mlx-community/Qwen3.5-122B-A10B-4bit",
        "label": "Qwen3.5-122B-A10B-4bit",
        "description": "MoE 122B, 10B active — large local model",
    },
    "27b-3.6": {
        "backend": "mlx",
        "model": "mlx-community/Qwen3.6-27B-4bit",
        "label": "Qwen3.6-27B-4bit",
        "description": "Qwen3.6 dense 27B (released 2026-04-22) — between 9B and 35B tier",
    },
    "35b-3.6": {
        "backend": "mlx",
        "model": "mlx-community/Qwen3.6-35B-A3B-4bit",
        "label": "Qwen3.6-35B-A3B-4bit",
        "description": "Qwen3.6 MoE 35B/3B-active — drop-in C2 successor to 35b (released 2026-04-16)",
    },
    "35b-3.6-dwq": {
        "backend": "mlx",
        "model": "mlx-community/Qwen3.6-35B-A3B-4bit-DWQ",
        "label": "Qwen3.6-35B-A3B-4bit-DWQ",
        "description": "Qwen3.6 35B/3B-active with DWQ quant — +1-3% expected vs plain 4bit",
    },
    "flashmoe-q3": {
        "backend": "flash-moe",
        "binary": "~/projects/flash-moe/metal_infer/infer",
        "model": "~/Models/flash_mlx_4bit",
        "label": "flash-moe-397B-Q3",
        "q3_experts": True,
        "cache_io_split": 4,
        "malloc_cache": 0,
        "description": "Flash-MoE 397B Q3 GGUF — recommended config",
    },
    "flashmoe-4bit": {
        "backend": "flash-moe",
        "binary": "~/projects/flash-moe/metal_infer/infer",
        "model": "~/Models/flash_mlx_4bit",
        "label": "flash-moe-397B-4bit",
        "q3_experts": False,
        "cache_io_split": 4,
        "malloc_cache": 0,
        "description": "Flash-MoE 397B 4-bit MLX — higher quality, slower",
    },
    "deepseek-v3.2": {
        "backend": "mlx",
        "model": "mlx-community/DeepSeek-V3.2-4bit",
        "label": "DeepSeek-V3.2-4bit",
        "description": "DeepSeek V3.2 MoE 671B — may not fit alongside others",
    },
    "glm-5": {
        "backend": "mlx",
        "model": "mlx-community/GLM-5-4bit",
        "label": "GLM-5-4bit",
        "description": "GLM-5 MoE — untested on M5 Max",
    },
    "kimi-k2.5": {
        "backend": "mlx",
        "model": "mlx-community/Kimi-K2.5-3bit",
        "label": "Kimi-K2.5-3bit",
        "description": "Kimi K2.5 1T MoE — 3-bit quantization",
    },
    "cloud": {
        "backend": "cloud",
        "model": "claude-sonnet-4-20250514",
        "label": "Claude Sonnet (cloud)",
        "description": "Cloud baseline via Anthropic API",
    },
}


# ---------------------------------------------------------------------------
# Stage 1: Generate
# ---------------------------------------------------------------------------


def _generate_mlx(
    model_id: str,
    messages: list[dict],
    max_tokens: int,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Generate with an MLX model via mlx-lm.

    Uses stream_generate for token-level timeout support. If the timeout
    fires mid-generation, returns whatever was produced so far.
    """
    import mlx.core as mx
    from mlx_lm import load, stream_generate

    # Cache model loads across calls
    if not hasattr(_generate_mlx, "_cache"):
        _generate_mlx._cache = {}

    cache = _generate_mlx._cache
    if model_id not in cache:
        print(f"    Loading MLX model: {model_id}")
        t0 = time.monotonic()
        model, tokenizer = load(model_id)
        print(f"    Loaded in {time.monotonic() - t0:.1f}s")
        cache[model_id] = (model, tokenizer)

    model, tokenizer = cache[model_id]

    # Format as chat
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = messages[-1]["content"]

    mx.metal.reset_peak_memory()
    chunks: list[str] = []
    ttft = 0.0
    timed_out = False

    t_start = time.monotonic()
    for chunk in stream_generate(model, tokenizer, prompt=text, max_tokens=max_tokens):
        # stream_generate yields dicts with 'text' (or objects with .text)
        token_text = (
            chunk.text if hasattr(chunk, "text") else chunk.get("text", str(chunk))
        )
        if not chunks:
            ttft = time.monotonic() - t_start
        chunks.append(token_text)
        if time.monotonic() - t_start > timeout:
            timed_out = True
            break
    t_end = time.monotonic()

    elapsed = t_end - t_start
    output = "".join(chunks)
    n_tokens = len(chunks)
    tps = n_tokens / elapsed if elapsed > 0 else 0.0

    return {
        "output_text": output,
        "tokens_generated": n_tokens,
        "elapsed_s": round(elapsed, 3),
        "ttft_s": round(ttft, 4),
        "gen_tps": round(tps, 2),
        "peak_mem_gb": round(mx.metal.get_peak_memory() / 1e9, 2),
        "timed_out": timed_out,
    }


def _generate_flashmoe(
    config: dict,
    messages: list[dict],
    max_tokens: int,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Generate with flash-moe binary via FlashMoeWorker."""
    from server.flashmoe_worker import FlashMoeWorker

    if not hasattr(_generate_flashmoe, "_worker"):
        binary = os.path.expanduser(config["binary"])
        model_path = os.path.expanduser(config["model"])
        worker = FlashMoeWorker(
            binary_path=binary,
            model_path=model_path,
            q3_experts=config.get("q3_experts", False),
            cache_io_split=config.get("cache_io_split", 0),
            malloc_cache=config.get("malloc_cache", 0),
        )
        worker.start()
        _generate_flashmoe._worker = worker
        _generate_flashmoe._config_key = json.dumps(
            {k: config[k] for k in sorted(config) if k != "description"}, sort_keys=True
        )

    worker = _generate_flashmoe._worker
    # Check if config changed — restart if so
    config_key = json.dumps(
        {k: config[k] for k in sorted(config) if k != "description"}, sort_keys=True
    )
    if config_key != _generate_flashmoe._config_key:
        worker.shutdown()
        binary = os.path.expanduser(config["binary"])
        model_path = os.path.expanduser(config["model"])
        worker = FlashMoeWorker(
            binary_path=binary,
            model_path=model_path,
            q3_experts=config.get("q3_experts", False),
            cache_io_split=config.get("cache_io_split", 0),
            malloc_cache=config.get("malloc_cache", 0),
        )
        worker.start()
        _generate_flashmoe._worker = worker
        _generate_flashmoe._config_key = config_key

    timed_out = False
    t_start = time.monotonic()
    chunks: list[str] = []
    # Iterate the generator so we can break on timeout
    for chunk in worker.generate(
        messages=messages, max_tokens=max_tokens, timeout=timeout + 30
    ):
        chunks.append(chunk)
        if time.monotonic() - t_start > timeout:
            timed_out = True
            break
    t_end = time.monotonic()

    elapsed = t_end - t_start
    output_text = "".join(chunks)
    metrics = worker.last_generation_metrics
    tps = metrics.get("generation_tps", 0)
    if tps == 0 and elapsed > 0:
        tps = len(chunks) / elapsed
    worker.reset_consecutive_crashes()

    return {
        "output_text": output_text,
        "tokens_generated": metrics.get("tokens_generated", len(chunks)),
        "elapsed_s": round(elapsed, 3),
        "ttft_s": 0,  # not measurable via HTTP proxy
        "gen_tps": round(tps, 2),
        "peak_mem_gb": metrics.get("peak_memory_gb", 0),
        "timed_out": timed_out,
    }


def _generate_cloud(
    model: str,
    messages: list[dict],
    max_tokens: int,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Generate with Claude API."""
    from anthropic import Anthropic

    client = Anthropic(timeout=timeout)

    # Separate system message if present
    system_msg = None
    chat_messages = []
    for m in messages:
        if m["role"] == "system":
            system_msg = m["content"]
        else:
            chat_messages.append(m)

    t_start = time.monotonic()
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": chat_messages,
    }
    if system_msg:
        kwargs["system"] = system_msg

    response = client.messages.create(**kwargs)
    t_end = time.monotonic()

    elapsed = t_end - t_start
    output_text = response.content[0].text if response.content else ""
    n_tokens = response.usage.output_tokens

    return {
        "output_text": output_text,
        "tokens_generated": n_tokens,
        "elapsed_s": round(elapsed, 3),
        "ttft_s": 0,  # not meaningful for cloud
        "gen_tps": round(n_tokens / elapsed if elapsed > 0 else 0, 2),
        "peak_mem_gb": 0,
    }


def _read_thermal() -> str:
    """Read macOS thermal state."""
    try:
        from server.thermal import ThermalMonitor

        return ThermalMonitor().read().level
    except Exception:
        return "unknown"


def stage_generate(
    prompts_path: str,
    output_dir: str,
    configs: list[str],
    runs: int = 1,
    timeout: float = 300.0,
) -> None:
    """Stage 1: Generate outputs from all configs."""
    with open(prompts_path) as f:
        prompts = json.load(f)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load existing results to allow resumption
    results_file = out_path / "generations.jsonl"
    existing: set[tuple[str, str, int]] = set()
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                r = json.loads(line)
                existing.add((r["config"], r["prompt_id"], r.get("run", 1)))
        print(f"Found {len(existing)} existing results — will skip those")

    total = len(configs) * len(prompts) * runs
    done = 0

    for config_name in configs:
        if config_name not in CONFIG_REGISTRY:
            print(f"WARNING: Unknown config '{config_name}', skipping")
            continue

        config = CONFIG_REGISTRY[config_name]
        print(f"\n{'='*60}")
        print(f"  Config: {config_name} — {config.get('label', '')}")
        print(f"{'='*60}")

        for prompt in prompts:
            for run_i in range(1, runs + 1):
                done += 1
                key = (config_name, prompt["id"], run_i)
                if key in existing:
                    print(
                        f"  [{done}/{total}] {prompt['id']} run {run_i} — cached, skipping"
                    )
                    continue

                print(
                    f"  [{done}/{total}] {prompt['id']} run {run_i}...",
                    end="",
                    flush=True,
                )

                thermal_start = _read_thermal()
                try:
                    if config["backend"] == "mlx":
                        result = _generate_mlx(
                            config["model"],
                            prompt["messages"],
                            prompt.get("max_tokens", 512),
                            timeout=timeout,
                        )
                    elif config["backend"] == "flash-moe":
                        result = _generate_flashmoe(
                            config,
                            prompt["messages"],
                            prompt.get("max_tokens", 512),
                            timeout=timeout,
                        )
                    elif config["backend"] == "cloud":
                        result = _generate_cloud(
                            config["model"],
                            prompt["messages"],
                            prompt.get("max_tokens", 512),
                            timeout=timeout,
                        )
                    else:
                        print(f" unknown backend: {config['backend']}")
                        continue
                except Exception as e:
                    print(f" ERROR: {e}")
                    # Record the error as a result with empty output
                    result = {
                        "output_text": f"ERROR: {e}",
                        "tokens_generated": 0,
                        "elapsed_s": 0,
                        "ttft_s": 0,
                        "gen_tps": 0,
                        "peak_mem_gb": 0,
                    }

                thermal_end = _read_thermal()

                gen_result = GenerationResult(
                    config=config_name,
                    prompt_id=prompt["id"],
                    category=prompt.get("category", ""),
                    difficulty=prompt.get("difficulty", ""),
                    output_text=result["output_text"],
                    tokens_generated=result["tokens_generated"],
                    elapsed_s=result["elapsed_s"],
                    ttft_s=result["ttft_s"],
                    gen_tps=result["gen_tps"],
                    peak_mem_gb=result["peak_mem_gb"],
                    thermal_start=thermal_start,
                    thermal_end=thermal_end,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    run=run_i,
                    timed_out=result.get("timed_out", False),
                )

                # Append to JSONL (crash-safe — one line at a time)
                with open(results_file, "a") as f:
                    f.write(json.dumps(gen_result.to_dict()) + "\n")

                timeout_tag = " [TIMEOUT]" if result.get("timed_out") else ""
                print(
                    f" {result['gen_tps']} tok/s, "
                    f"{result['tokens_generated']} tokens, "
                    f"{result['elapsed_s']}s{timeout_tag}"
                )

    # Shutdown flash-moe worker if it was started
    if hasattr(_generate_flashmoe, "_worker"):
        _generate_flashmoe._worker.shutdown()
        del _generate_flashmoe._worker

    print(f"\nGeneration complete. Results in {results_file}")


# ---------------------------------------------------------------------------
# Stage 2: Execute (Codex sandbox)
# ---------------------------------------------------------------------------


def _extract_code(text: str) -> str:
    """Extract Python code from markdown fences or raw text."""
    # Try to find ```python ... ``` blocks
    matches = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if matches:
        return "\n\n".join(matches)
    # Try generic ``` blocks
    matches = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    if matches:
        return "\n\n".join(matches)
    # Fall back to raw text (might be code without fences)
    return text


def stage_execute(results_dir: str) -> None:
    """Stage 2: Run executable tests on generated code outputs."""
    results_path = Path(results_dir)
    gen_file = results_path / "generations.jsonl"
    if not gen_file.exists():
        print("ERROR: No generations.jsonl found. Run 'generate' first.")
        sys.exit(1)

    # Load prompts to get exec_test definitions
    # Try to find the prompts file
    prompts_file = results_path / "prompts.json"
    if not prompts_file.exists():
        # Fall back to default location
        prompts_file = Path(__file__).parent / "prompts" / "holistic_eval.json"
    with open(prompts_file) as f:
        prompts = json.load(f)

    exec_tests: dict[str, str | None] = {}
    for p in prompts:
        exec_tests[p["id"]] = p.get("exec_test")

    # Load generations
    generations: list[dict] = []
    with open(gen_file) as f:
        for line in f:
            generations.append(json.loads(line))

    # Run exec tests
    exec_results: list[dict] = []
    for gen in generations:
        prompt_id = gen["prompt_id"]
        config = gen["config"]
        test_code = exec_tests.get(prompt_id)

        if test_code is None:
            # No executable test for this prompt
            continue

        output_text = gen["output_text"]
        if output_text.startswith("ERROR:"):
            exec_results.append(
                {
                    "config": config,
                    "prompt_id": prompt_id,
                    "run": gen.get("run", 1),
                    "passed": False,
                    "error": "Generation failed",
                    "exec_time_s": 0,
                }
            )
            continue

        code = _extract_code(output_text)

        # Write code to temp dir and run test
        with tempfile.TemporaryDirectory() as tmpdir:
            # For code that outputs to stdout (tool_call tests), save raw output
            if "output.txt" in test_code:
                with open(os.path.join(tmpdir, "output.txt"), "w") as f:
                    f.write(output_text)

            # Write extracted code as solution.py
            with open(os.path.join(tmpdir, "solution.py"), "w") as f:
                f.write(code)

            # Write test
            with open(os.path.join(tmpdir, "test_eval.py"), "w") as f:
                f.write(test_code)

            print(
                f"  [{config}] {prompt_id}...",
                end="",
                flush=True,
            )

            t_start = time.monotonic()
            try:
                proc = subprocess.run(
                    [sys.executable, os.path.join(tmpdir, "test_eval.py")],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=tmpdir,
                )
                elapsed = time.monotonic() - t_start
                passed = proc.returncode == 0 and "PASS" in proc.stdout
                error = proc.stderr.strip() if not passed else ""
                if not passed and not error:
                    error = proc.stdout.strip()[:500]
            except subprocess.TimeoutExpired:
                elapsed = 30.0
                passed = False
                error = "Execution timed out (30s)"
            except Exception as e:
                elapsed = time.monotonic() - t_start
                passed = False
                error = str(e)

            status = "PASS" if passed else "FAIL"
            print(f" {status} ({elapsed:.2f}s)")

            exec_results.append(
                {
                    "config": config,
                    "prompt_id": prompt_id,
                    "run": gen.get("run", 1),
                    "passed": passed,
                    "error": error[:500],
                    "exec_time_s": round(elapsed, 3),
                }
            )

    # Write results
    exec_file = results_path / "exec_results.jsonl"
    with open(exec_file, "w") as f:
        for r in exec_results:
            f.write(json.dumps(r) + "\n")

    # Summary
    total = len(exec_results)
    passed = sum(1 for r in exec_results if r["passed"])
    print(f"\nExecution: {passed}/{total} passed")
    print(f"Results in {exec_file}")


# ---------------------------------------------------------------------------
# Stage 3: Judge (Claude -p)
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are evaluating the output of a coding AI model. Score it on these dimensions (0-5 each):

**TASK:**
{task}

**MODEL OUTPUT:**
{output}

{rubric_section}

Score each dimension 0-5:
- **correctness**: Does the code/answer actually solve the problem correctly? 5=perfect, 0=completely wrong
- **completeness**: Are all requirements addressed? Edge cases handled? 5=all requirements met, 0=major omissions
- **code_quality**: Clean code, good naming, Pythonic, proper error handling? 5=exemplary, 0=unreadable
- **instruction_following**: Did it follow the exact format/constraints requested? 5=perfect compliance, 0=ignored instructions

Respond with EXACTLY this format (no other text):
correctness: N
completeness: N
code_quality: N
instruction_following: N
reasoning: One sentence explaining the most important observation."""


def _build_rubric_section(prompt: dict) -> str:
    """Build rubric instructions for the judge."""
    rubric = prompt.get("judge_rubric")
    if not rubric:
        return ""

    parts = ["**EVALUATION RUBRIC:**"]
    if "must_identify" in rubric:
        parts.append("Must identify: " + ", ".join(rubric["must_identify"]))
    if "must_include" in rubric:
        parts.append("Must include: " + ", ".join(rubric["must_include"]))
    if "should_identify" in rubric:
        parts.append("Should identify: " + ", ".join(rubric["should_identify"]))
    if "should_include" in rubric:
        parts.append("Should include: " + ", ".join(rubric["should_include"]))
    if "correct_answer" in rubric:
        parts.append(f"Correct answer: {rubric['correct_answer']}")
    if "common_errors" in rubric:
        parts.append("Common errors to penalize: " + ", ".join(rubric["common_errors"]))
    return "\n".join(parts)


def _parse_judge_response(text: str) -> dict[str, Any]:
    """Parse the judge's structured response.

    Looks for 'key: N' patterns anywhere in the text. Works for both
    claude -p (clean output) and codex exec (chatty output with session info).
    """
    scores: dict[str, Any] = {
        "correctness": 0,
        "completeness": 0,
        "code_quality": 0,
        "instruction_following": 0,
        "reasoning": "",
    }
    for line in text.split("\n"):
        stripped = line.strip()
        for key in [
            "correctness",
            "completeness",
            "code_quality",
            "instruction_following",
        ]:
            m = re.search(rf"\b{key}\s*:\s*(\d+)", stripped, re.IGNORECASE)
            if m and scores[key] == 0:
                try:
                    scores[key] = int(m.group(1))
                except ValueError:
                    pass
        m = re.search(r"\breasoning\s*:\s*(.+)", stripped, re.IGNORECASE)
        if m and not scores["reasoning"]:
            scores["reasoning"] = m.group(1).strip()[:300]
    return scores


def _call_judge_claude(
    judge_input: str, timeout: float = 60.0
) -> tuple[bool, str, str]:
    """Call claude -p and return (success, stdout, stderr)."""
    try:
        proc = subprocess.run(
            ["claude", "-p", "--model", "sonnet"],
            input=judge_input,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        ok = proc.returncode == 0 and bool(proc.stdout.strip())
        return ok, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return False, "", "timeout"
    except FileNotFoundError:
        return False, "", "claude CLI not found"


def _call_judge_codex(
    judge_input: str, timeout: float = 180.0
) -> tuple[bool, str, str]:
    """Call codex exec and return (success, stdout, stderr).

    codex exec reads prompt from stdin. Its output is chatty (session info,
    token counts) but _parse_judge_response searches all lines so it works.
    """
    try:
        proc = subprocess.run(
            ["codex", "exec"],
            input=judge_input,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        ok = proc.returncode == 0 and bool(proc.stdout.strip())
        return ok, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return False, "", "timeout"
    except FileNotFoundError:
        return False, "", "codex CLI not found"


def stage_judge(results_dir: str, judge_backend: str = "codex") -> None:
    """Stage 3: Judge outputs with an external LLM judge.

    Args:
        results_dir: Directory containing generations.jsonl
        judge_backend: "codex" (codex exec) or "claude" (claude -p)
    """
    results_path = Path(results_dir)
    gen_file = results_path / "generations.jsonl"
    if not gen_file.exists():
        print("ERROR: No generations.jsonl found. Run 'generate' first.")
        sys.exit(1)

    # Load prompts
    prompts_file = results_path / "prompts.json"
    if not prompts_file.exists():
        prompts_file = Path(__file__).parent / "prompts" / "holistic_eval.json"
    with open(prompts_file) as f:
        prompts = json.load(f)
    prompt_map = {p["id"]: p for p in prompts}

    # Load generations
    generations: list[dict] = []
    with open(gen_file) as f:
        for line in f:
            generations.append(json.loads(line))

    # Load existing judgments for resumption
    judge_file = results_path / "judge_results.jsonl"
    existing: set[tuple[str, str, int]] = set()
    if judge_file.exists():
        with open(judge_file) as f:
            for line in f:
                r = json.loads(line)
                existing.add((r["config"], r["prompt_id"], r.get("run", 1)))
        print(f"Found {len(existing)} existing judgments — will skip those")

    total = len(generations)
    for i, gen in enumerate(generations):
        config = gen["config"]
        prompt_id = gen["prompt_id"]
        run_i = gen.get("run", 1)

        if (config, prompt_id, run_i) in existing:
            print(f"  [{i+1}/{total}] {config}/{prompt_id} — cached, skipping")
            continue

        if gen["output_text"].startswith("ERROR:"):
            # Record zero scores for failed generations
            result = {
                "config": config,
                "prompt_id": prompt_id,
                "run": run_i,
                "correctness": 0,
                "completeness": 0,
                "code_quality": 0,
                "instruction_following": 0,
                "reasoning": "Generation failed",
            }
            with open(judge_file, "a") as f:
                f.write(json.dumps(result) + "\n")
            continue

        prompt = prompt_map.get(prompt_id, {})
        task_text = prompt.get("messages", [{}])[-1].get("content", prompt_id)
        rubric_section = _build_rubric_section(prompt)

        judge_input = JUDGE_PROMPT.format(
            task=task_text,
            output=gen["output_text"][:3000],  # truncate for cost
            rubric_section=rubric_section,
        )

        print(f"  [{i+1}/{total}] {config}/{prompt_id}...", end="", flush=True)

        if judge_backend == "claude":
            ok, stdout, stderr = _call_judge_claude(judge_input)
            label = "claude -p"
        else:
            ok, stdout, stderr = _call_judge_codex(judge_input)
            label = "codex exec"

        if not ok:
            # Permanent failure on the very first call → abort stage entirely
            # so the user can fix their environment
            if i == 0:
                print(f" {label} failed: {stderr[:200] or 'no output'}")
                print(
                    f"  ABORT: judge backend '{judge_backend}' is not working. "
                    f"Try --judge {'codex' if judge_backend == 'claude' else 'claude'}"
                )
                return
            print(f" {label} failed: {stderr[:200] or 'no output'}")
            scores = {
                "correctness": 0,
                "completeness": 0,
                "code_quality": 0,
                "instruction_following": 0,
                "reasoning": f"Judge error: {stderr[:200]}",
            }
        else:
            scores = _parse_judge_response(stdout)
            # Sanity check: if parsing produced all zeros, the judge output
            # didn't match the expected format
            if all(
                scores[k] == 0
                for k in [
                    "correctness",
                    "completeness",
                    "code_quality",
                    "instruction_following",
                ]
            ):
                scores["reasoning"] = f"Parse failed. Raw: {stdout[:200]}"

        result = {
            "config": config,
            "prompt_id": prompt_id,
            "run": run_i,
            **scores,
        }
        with open(judge_file, "a") as f:
            f.write(json.dumps(result) + "\n")

        composite = (
            scores["correctness"]
            + scores["completeness"]
            + scores["code_quality"]
            + scores["instruction_following"]
        ) / 4
        print(f" {composite:.1f}/5")

    print(f"\nJudging complete. Results in {judge_file}")


# ---------------------------------------------------------------------------
# Stage 4: Report
# ---------------------------------------------------------------------------


def _percentile(data: list[float], p: float) -> float:
    """Simple percentile."""
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (k - f) * (s[c] - s[f])


def stage_report(results_dir: str) -> None:
    """Stage 4: Generate scorecard from all results."""
    import statistics

    results_path = Path(results_dir)

    # Load generations
    gen_file = results_path / "generations.jsonl"
    generations: list[dict] = []
    if gen_file.exists():
        with open(gen_file) as f:
            for line in f:
                generations.append(json.loads(line))

    # Load exec results
    exec_file = results_path / "exec_results.jsonl"
    exec_results: list[dict] = []
    if exec_file.exists():
        with open(exec_file) as f:
            for line in f:
                exec_results.append(json.loads(line))

    # Load judge results
    judge_file = results_path / "judge_results.jsonl"
    judge_results: list[dict] = []
    if judge_file.exists():
        with open(judge_file) as f:
            for line in f:
                judge_results.append(json.loads(line))

    # Group by config
    configs = sorted(set(g["config"] for g in generations))

    # Build scorecard
    scorecards: list[dict] = []
    for config in configs:
        config_gens = [g for g in generations if g["config"] == config]
        config_execs = [e for e in exec_results if e["config"] == config]
        config_judges = [j for j in judge_results if j["config"] == config]

        # Speed metrics
        tps_values = [g["gen_tps"] for g in config_gens if g["gen_tps"] > 0]
        ttft_values = [g["ttft_s"] for g in config_gens if g["ttft_s"] > 0]
        mem_values = [g["peak_mem_gb"] for g in config_gens if g["peak_mem_gb"] > 0]

        # Exec pass rate
        exec_total = len(config_execs)
        exec_passed = sum(1 for e in config_execs if e["passed"])

        # Judge averages
        judge_fields = [
            "correctness",
            "completeness",
            "code_quality",
            "instruction_following",
        ]
        judge_avgs = {}
        for field_name in judge_fields:
            vals = [
                j[field_name]
                for j in config_judges
                if isinstance(j[field_name], (int, float))
            ]
            judge_avgs[field_name] = statistics.mean(vals) if vals else 0

        composite = statistics.mean(judge_avgs.values()) if judge_avgs else 0

        # Error and timeout counts
        errors = sum(1 for g in config_gens if g["output_text"].startswith("ERROR:"))
        timeouts = sum(1 for g in config_gens if g.get("timed_out", False))

        # Thermal end state — take the worst
        thermals = [g["thermal_end"] for g in config_gens]
        thermal_worst = "nominal"
        for level in ["sleeping", "trapping", "heavy", "moderate", "nominal"]:
            if level in thermals:
                thermal_worst = level
                break

        label = CONFIG_REGISTRY.get(config, {}).get("label", config)

        card = {
            "config": config,
            "label": label,
            # Quality
            "exec_pass_rate": round(exec_passed / exec_total, 3)
            if exec_total > 0
            else None,
            "exec_passed": exec_passed,
            "exec_total": exec_total,
            "judge_correctness": round(judge_avgs.get("correctness", 0), 2),
            "judge_completeness": round(judge_avgs.get("completeness", 0), 2),
            "judge_quality": round(judge_avgs.get("code_quality", 0), 2),
            "judge_instruction": round(judge_avgs.get("instruction_following", 0), 2),
            "judge_composite": round(composite, 2),
            # Speed
            "median_tps": round(statistics.median(tps_values), 1) if tps_values else 0,
            "p5_tps": round(_percentile(tps_values, 5), 1) if tps_values else 0,
            "p95_tps": round(_percentile(tps_values, 95), 1) if tps_values else 0,
            "median_ttft_s": round(statistics.median(ttft_values), 3)
            if ttft_values
            else 0,
            # Resource
            "peak_mem_gb": round(max(mem_values), 1) if mem_values else 0,
            "thermal_worst": thermal_worst,
            # Reliability
            "errors": errors,
            "timeouts": timeouts,
            "total_prompts": len(config_gens),
        }
        scorecards.append(card)

    # Write JSON scorecard
    scorecard_file = results_path / "scorecard.json"
    with open(scorecard_file, "w") as f:
        json.dump(scorecards, f, indent=2)

    # Print markdown table
    print(f"\n{'='*100}")
    print("  HOLISTIC BENCHMARK SCORECARD")
    print(f"{'='*100}\n")

    # Header
    header = (
        f"{'Config':<25} {'Exec':>5} {'Judge':>6} "
        f"{'Corr':>5} {'Comp':>5} {'Qual':>5} {'Inst':>5} "
        f"{'med t/s':>8} {'p5':>6} {'p95':>6} {'TTFT':>6} "
        f"{'Mem GB':>7} {'Therm':>8} {'Err':>4} {'T/O':>4}"
    )
    print(header)
    print("-" * len(header))

    for card in scorecards:
        exec_str = (
            f"{card['exec_pass_rate']:.0%}"
            if card["exec_pass_rate"] is not None
            else "n/a"
        )
        print(
            f"{card['label']:<25} {exec_str:>5} {card['judge_composite']:>6.2f} "
            f"{card['judge_correctness']:>5.2f} {card['judge_completeness']:>5.2f} "
            f"{card['judge_quality']:>5.2f} {card['judge_instruction']:>5.2f} "
            f"{card['median_tps']:>8.1f} {card['p5_tps']:>6.1f} {card['p95_tps']:>6.1f} "
            f"{card['median_ttft_s']:>6.3f} "
            f"{card['peak_mem_gb']:>7.1f} {card['thermal_worst']:>8} "
            f"{card['errors']:>4} {card['timeouts']:>4}"
        )

    print(f"\n  Scorecard saved to {scorecard_file}")

    # Per-prompt breakdown
    print(f"\n{'='*80}")
    print("  PER-PROMPT EXEC RESULTS")
    print(f"{'='*80}\n")

    prompt_ids = sorted(set(e["prompt_id"] for e in exec_results))
    header2 = f"{'Prompt':<25} " + " ".join(f"{c:>12}" for c in configs)
    print(header2)
    print("-" * len(header2))
    for pid in prompt_ids:
        row = f"{pid:<25} "
        for config in configs:
            matches = [
                e
                for e in exec_results
                if e["config"] == config and e["prompt_id"] == pid
            ]
            if matches:
                status = "PASS" if matches[0]["passed"] else "FAIL"
            else:
                status = "-"
            row += f"{status:>12} "
        print(row)

    # Write TSV for easy spreadsheet import
    tsv_file = results_path / "scorecard.tsv"
    if scorecards:
        columns = list(scorecards[0].keys())
        with open(tsv_file, "w") as f:
            f.write("\t".join(columns) + "\n")
            for card in scorecards:
                f.write("\t".join(str(card.get(c, "")) for c in columns) + "\n")
        print(f"  TSV saved to {tsv_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="holistic-benchmark",
        description="Holistic benchmark: quality + latency + memory + reliability",
    )
    sub = parser.add_subparsers(dest="stage", required=True)

    # Generate
    gen_p = sub.add_parser("generate", help="Stage 1: Generate outputs")
    gen_p.add_argument("--prompts", required=True, help="Path to prompts JSON")
    gen_p.add_argument("--output", required=True, help="Output directory for results")
    gen_p.add_argument(
        "--configs",
        required=True,
        help=f"Comma-separated config names: {','.join(CONFIG_REGISTRY.keys())}",
    )
    gen_p.add_argument(
        "--runs", type=int, default=1, help="Runs per prompt (default: 1)"
    )
    gen_p.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-prompt timeout in seconds (default: 300). Partial output is kept.",
    )

    # Execute
    exec_p = sub.add_parser("execute", help="Stage 2: Execute code tests")
    exec_p.add_argument(
        "--results", required=True, help="Results directory from generate"
    )

    # Judge
    judge_p = sub.add_parser("judge", help="Stage 3: LLM-as-judge scoring")
    judge_p.add_argument(
        "--results", required=True, help="Results directory from generate"
    )
    judge_p.add_argument(
        "--judge",
        choices=["codex", "claude"],
        default="codex",
        help="Judge backend: 'codex' (codex exec) or 'claude' (claude -p). Default: codex",
    )

    # Report
    report_p = sub.add_parser("report", help="Stage 4: Generate scorecard")
    report_p.add_argument("--results", required=True, help="Results directory")

    # All stages
    all_p = sub.add_parser("all", help="Run all stages sequentially")
    all_p.add_argument("--prompts", required=True, help="Path to prompts JSON")
    all_p.add_argument("--output", required=True, help="Output directory")
    all_p.add_argument(
        "--configs",
        required=True,
        help=f"Comma-separated config names: {','.join(CONFIG_REGISTRY.keys())}",
    )
    all_p.add_argument("--runs", type=int, default=1, help="Runs per prompt")
    all_p.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-prompt timeout in seconds (default: 300)",
    )
    all_p.add_argument(
        "--judge",
        choices=["codex", "claude"],
        default="codex",
        help="Judge backend for stage 3. Default: codex",
    )

    # List configs
    sub.add_parser("list-configs", help="List available model configs")

    args = parser.parse_args(argv)

    if args.stage == "list-configs":
        print(f"\n{'Config':<20} {'Backend':<10} {'Label':<30} {'Description'}")
        print("-" * 90)
        for name, cfg in CONFIG_REGISTRY.items():
            print(
                f"{name:<20} {cfg['backend']:<10} "
                f"{cfg.get('label', ''):<30} {cfg.get('description', '')}"
            )
        return

    if args.stage == "generate":
        configs = [c.strip() for c in args.configs.split(",")]
        # Copy prompts to output dir for reproducibility
        out_path = Path(args.output)
        out_path.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy2(args.prompts, out_path / "prompts.json")
        stage_generate(
            args.prompts, args.output, configs, args.runs, timeout=args.timeout
        )

    elif args.stage == "execute":
        stage_execute(args.results)

    elif args.stage == "judge":
        stage_judge(args.results, judge_backend=args.judge)

    elif args.stage == "report":
        stage_report(args.results)

    elif args.stage == "all":
        configs = [c.strip() for c in args.configs.split(",")]
        out_path = Path(args.output)
        out_path.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy2(args.prompts, out_path / "prompts.json")

        print("\n" + "=" * 60)
        print("  STAGE 1: GENERATE")
        print("=" * 60)
        stage_generate(
            args.prompts, args.output, configs, args.runs, timeout=args.timeout
        )

        print("\n" + "=" * 60)
        print("  STAGE 2: EXECUTE")
        print("=" * 60)
        stage_execute(args.output)

        print("\n" + "=" * 60)
        print("  STAGE 3: JUDGE")
        print("=" * 60)
        stage_judge(args.output, judge_backend=args.judge)

        print("\n" + "=" * 60)
        print("  STAGE 4: REPORT")
        print("=" * 60)
        stage_report(args.output)


if __name__ == "__main__":
    main()
