#!/usr/bin/env bash
# Run LiveCodeBench v6 across the Qwen3.5/3.6 matrix, one model per process.
#
# Why process-per-model: _generate_mlx caches loaded models indefinitely in an
# in-process lru. Loading 4 × 18GB models in one process would exceed the M5
# Max's 128GB working set once OS+apps are accounted for. Spawning fresh
# Python per model lets the OS reclaim weight RAM between runs. The CLI's
# resumption logic means each invocation only does its own model's problems.
#
# Order chosen so the headline hypothesis (DWQ vs 3.5) lands first; 3.5-35b
# runs last as the regression baseline so a partial run still gives an A/B.
#
# Usage:
#   bash scripts/run_lcb_matrix.sh                       # full matrix
#   LCB_LIMIT=20 bash scripts/run_lcb_matrix.sh          # subset for smoke
#   LCB_OUT=/tmp/lcb bash scripts/run_lcb_matrix.sh      # alt output dir

set -euo pipefail

OUT="${LCB_OUT:-benchmarks/lcb_v6_matrix}"
LIMIT_FLAG=""
[ -n "${LCB_LIMIT:-}" ] && LIMIT_FLAG="--limit=${LCB_LIMIT}"

mkdir -p "$OUT"
echo "LCB v6 matrix → $OUT  (limit=${LCB_LIMIT:-none})"
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

MODELS=(
  cloud
  local:qwen3.6-35b-dwq
  local:qwen3.6-35b
  local:qwen3.6-27b
  local:qwen3.5-35b
)

for m in "${MODELS[@]}"; do
  echo
  echo "=========================================="
  echo "  $(date -u +%H:%M:%S)  $m"
  echo "=========================================="
  uv run python -m benchmarks.code_correctness \
    --model="$m" \
    --suite=livecodebench-v6 \
    --output="$OUT" \
    --timeout=120 \
    $LIMIT_FLAG \
    2>&1 | tee -a "$OUT/run.log"
done

echo
echo "=========================================="
echo "  Matrix complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=========================================="
echo
echo "Summary:"
cat "$OUT/code_correctness_summary.json"
