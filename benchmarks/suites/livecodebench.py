"""LiveCodeBench v6 suite — stdin/stdout code-correctness executor.

LCB problems are `(input_string, expected_output)` pairs that the generated
program must satisfy when invoked as `python solution.py < input`. The
release_version v6 subset is time-segmented to late-2025+ so it is
contamination-free for models trained through mid-2025.

Loading order:
  1. HuggingFace `livecodebench/code_generation_lite` with release_version=v6
     (requires `datasets` + network).
  2. Local cache at benchmarks/datasets_cache/livecodebench_v6.jsonl.
  3. Built-in DRY_RUN_FIXTURE (2 problems) when --dry-run and no cache.

Execution: write the extracted code to a temp file, run with the problem's
stdin, compare stdout against expected output (stripped + newline-normalised).
Per-problem timeout defaults to 15s.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

DATASET_CACHE = (
    Path(__file__).resolve().parent.parent / "datasets_cache" / "livecodebench_v6.jsonl"
)

# Minimal fixture for --dry-run. Neither problem clones a repo; each is pure
# stdin → stdout. Keep these simple so the dry-run path does not surprise.
DRY_RUN_FIXTURE: list[dict] = [
    {
        "question_id": "lcb_dryrun_echo",
        "question_title": "Echo a line",
        "question_content": (
            "Read a single line from stdin and print it verbatim to stdout."
        ),
        "starter_code": "",
        "tests": [
            {"input": "hello\n", "output": "hello\n"},
            {"input": "world\n", "output": "world\n"},
        ],
        "difficulty": "trivial",
        "release": "v6-dryrun",
    },
    {
        "question_id": "lcb_dryrun_sum",
        "question_title": "Sum two integers",
        "question_content": (
            "Read two space-separated integers from a single stdin line and "
            "print their sum on one line."
        ),
        "starter_code": "",
        "tests": [
            {"input": "2 3\n", "output": "5\n"},
            {"input": "-4 10\n", "output": "6\n"},
        ],
        "difficulty": "trivial",
        "release": "v6-dryrun",
    },
]


@dataclass
class LCBProblem:
    question_id: str
    question_title: str
    question_content: str
    starter_code: str
    tests: list[dict]
    difficulty: str
    release: str

    @classmethod
    def from_raw(cls, raw: dict) -> LCBProblem:
        tests = raw.get("tests")
        if tests is None:
            # HF format: public_test_cases is JSON-encoded list
            pub = raw.get("public_test_cases", "[]")
            priv = raw.get("private_test_cases", "[]")
            parsed: list[dict] = []
            for blob in (pub, priv):
                if isinstance(blob, str):
                    try:
                        parsed.extend(json.loads(blob))
                    except json.JSONDecodeError:
                        continue
                elif isinstance(blob, list):
                    parsed.extend(blob)
            tests = [
                {"input": t.get("input", ""), "output": t.get("output", "")}
                for t in parsed
            ]
        return cls(
            question_id=str(raw.get("question_id", raw.get("id", "unknown"))),
            question_title=raw.get("question_title", ""),
            question_content=raw.get("question_content", raw.get("problem", "")),
            starter_code=raw.get("starter_code", ""),
            tests=tests,
            difficulty=raw.get("difficulty", "unknown"),
            release=raw.get("release") or raw.get("release_version", "v6"),
        )


def load_problems(limit: int | None = None, dry_run: bool = False) -> list[LCBProblem]:
    """Load LCB v6 problems, preferring cache → HuggingFace → fixture."""
    if dry_run and not DATASET_CACHE.exists():
        return [LCBProblem.from_raw(p) for p in DRY_RUN_FIXTURE[: limit or None]]

    if DATASET_CACHE.exists():
        raws = []
        with open(DATASET_CACHE) as f:
            for line in f:
                line = line.strip()
                if line:
                    raws.append(json.loads(line))
        problems = [LCBProblem.from_raw(r) for r in raws]
        return problems[:limit] if limit else problems

    if dry_run:
        return [LCBProblem.from_raw(p) for p in DRY_RUN_FIXTURE[: limit or None]]

    # LiveCodeBench publishes the dataset as 6 monolithic JSONL files
    # (test.jsonl + test2-6.jsonl) and a Python loading script. `datasets>=4.0`
    # dropped script-based loaders, so we fetch test6.jsonl directly — that
    # file is the time-segmented v6 release delta (134 MB) and is what makes
    # v6 contamination-free.
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            f"No cached dataset at {DATASET_CACHE} and `huggingface_hub` not "
            "installed. Install or populate the cache."
        ) from e

    print(f"Fetching LCB v6 (test6.jsonl) → {DATASET_CACHE}")
    src = hf_hub_download(
        repo_id="livecodebench/code_generation_lite",
        filename="test6.jsonl",
        repo_type="dataset",
    )

    DATASET_CACHE.parent.mkdir(parents=True, exist_ok=True)
    raws: list[dict] = []
    with open(src) as fin, open(DATASET_CACHE, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            raws.append(json.loads(line))
            fout.write(line + "\n")

    problems = [LCBProblem.from_raw(r) for r in raws]
    return problems[:limit] if limit else problems


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _extract_python(text: str) -> str:
    """Pull Python code out of the model output, preferring fenced blocks."""
    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if fenced:
        # Prefer the longest block — models sometimes emit short snippets first.
        return max(fenced, key=len)
    return text


def _normalise(out: str) -> str:
    """Strip trailing whitespace per line + drop trailing blank lines."""
    lines = [line.rstrip() for line in out.splitlines()]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def run_problem(
    problem: LCBProblem, model_output: str, per_test_timeout_s: float = 15.0
) -> dict:
    """Execute one LCB problem. Returns a dict with pass/fail details."""
    code = _extract_python(model_output).strip()
    if not code:
        return {
            "question_id": problem.question_id,
            "passed": False,
            "tests_total": len(problem.tests),
            "tests_passed": 0,
            "error": "no_code_extracted",
            "elapsed_s": 0.0,
        }

    with tempfile.TemporaryDirectory(prefix="lcb-") as tmp:
        solution = Path(tmp) / "solution.py"
        solution.write_text(code)

        tests_passed = 0
        first_error = ""
        t_start = time.monotonic()
        for i, test in enumerate(problem.tests):
            try:
                proc = subprocess.run(
                    ["python", str(solution)],
                    input=test.get("input", ""),
                    capture_output=True,
                    text=True,
                    timeout=per_test_timeout_s,
                )
            except subprocess.TimeoutExpired:
                first_error = first_error or f"test_{i}: timeout"
                continue
            except Exception as e:  # pragma: no cover — defensive
                first_error = first_error or f"test_{i}: launch_error: {e}"
                continue

            if proc.returncode != 0:
                first_error = first_error or (
                    f"test_{i}: returncode={proc.returncode} stderr={proc.stderr[:200]}"
                )
                continue

            if _normalise(proc.stdout) == _normalise(test.get("output", "")):
                tests_passed += 1
            else:
                first_error = first_error or f"test_{i}: output mismatch"

        elapsed = time.monotonic() - t_start

    return {
        "question_id": problem.question_id,
        "passed": tests_passed == len(problem.tests) and len(problem.tests) > 0,
        "tests_total": len(problem.tests),
        "tests_passed": tests_passed,
        "error": first_error if tests_passed < len(problem.tests) else "",
        "elapsed_s": round(elapsed, 3),
    }
