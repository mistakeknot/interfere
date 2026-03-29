#!/usr/bin/env bash
# Flash-MoE expert cache sweep benchmark
# Tests different --malloc-cache sizes and measures tok/s.
#
# Usage:
#   ./benchmarks/flashmoe_cache_sweep.sh [CACHE_SIZES...]
#   ./benchmarks/flashmoe_cache_sweep.sh 0 2581 5000 10000 15000
#
# Requires: flash-moe binary, Qwen3.5-397B model files

set -euo pipefail

BINARY="${FLASHMOE_BINARY:-$HOME/projects/flash-moe/metal_infer/infer}"
MODEL="${FLASHMOE_MODEL:-$HOME/Models/flash_mlx_4bit}"
PORT="${FLASHMOE_PORT:-8423}"
THINK_BUDGET="${FLASHMOE_THINK_BUDGET:-512}"

# Default sweep: 0 (no cache), 2581, 5000, 10000, 15000
CACHE_SIZES=("${@:-0 2581 5000 10000 15000}")
if [[ "${#CACHE_SIZES[@]}" -eq 1 && "${CACHE_SIZES[0]}" == *" "* ]]; then
    read -ra CACHE_SIZES <<< "${CACHE_SIZES[0]}"
fi

WARMUP_PROMPTS=3
BENCH_PROMPTS=5
MAX_TOKENS=32
RESULTS_FILE="benchmarks/results_$(date +%Y%m%d_%H%M%S).tsv"
LOGFILE="/tmp/flashmoe_bench.log"

# Benchmark prompts (varied complexity to exercise different expert routes)
PROMPTS=(
    "What is the capital of France? Answer in one word."
    "Explain quantum entanglement in two sentences."
    "Write a haiku about autumn."
    "What are the first 5 prime numbers?"
    "Translate 'hello world' to Japanese."
    "What is the derivative of x^3?"
    "Name three programming languages created in the 1990s."
    "What color is the sky on Mars?"
)

kill_flashmoe() {
    pkill -f "infer.*--serve.*$PORT" 2>/dev/null || true
    sleep 2
}

wait_for_ready() {
    local timeout="${1:-300}"
    local elapsed=0
    while [[ $elapsed -lt $timeout ]]; do
        if curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q "ok"; then
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    echo "ERROR: flash-moe did not become ready within ${timeout}s" >&2
    return 1
}

send_prompt() {
    local prompt="$1"
    local max_tokens="$2"
    # Use curl with streaming; flash-moe will log timing to stderr/logfile
    curl -s --max-time 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":$max_tokens,\"stream\":true}" \
        >/dev/null 2>&1
}

extract_metrics() {
    # Parse flash-moe's stderr log for generation metrics
    # Format: [serve] chatcmpl-N generated=M tokens in Xms (Y.YY tok/s)
    local log="$1"
    local skip_first="$2"  # skip first N entries (warmup)

    grep "generated=" "$log" | tail -n +"$((skip_first + 1))" | while read -r line; do
        local toks=$(echo "$line" | grep -o 'generated=[0-9]*' | cut -d= -f2)
        local tps=$(echo "$line" | grep -o '[0-9.]*  *tok/s' | awk '{print $1}')
        echo "$toks $tps"
    done
}

extract_cache_stats() {
    # Parse malloc_cache final stats from log
    local log="$1"
    grep "malloc_cache.*Final\|malloc_cache.*hit rate\|cache.*hits.*misses" "$log" | tail -1
}

echo "═══════════════════════════════════════════════════════"
echo "Flash-MoE Expert Cache Sweep Benchmark"
echo "═══════════════════════════════════════════════════════"
echo "Binary:    $BINARY"
echo "Model:     $MODEL"
echo "Port:      $PORT"
echo "Cache sizes: ${CACHE_SIZES[*]}"
echo "Warmup:    $WARMUP_PROMPTS prompts"
echo "Benchmark: $BENCH_PROMPTS prompts × $MAX_TOKENS tokens"
echo "Results:   $RESULTS_FILE"
echo "═══════════════════════════════════════════════════════"
echo ""

# Write TSV header
mkdir -p "$(dirname "$RESULTS_FILE")"
printf "cache_entries\tcache_gb\tstartup_s\tmean_tps\tmedian_tps\tmin_tps\tmax_tps\tcache_hits\tcache_misses\thit_rate\tpredict\n" > "$RESULTS_FILE"

for cache_size in "${CACHE_SIZES[@]}"; do
    echo ""
    echo "───────────────────────────────────────────────────"
    echo "Testing: --malloc-cache $cache_size"
    echo "───────────────────────────────────────────────────"

    kill_flashmoe
    : > "$LOGFILE"

    # Build command
    CMD=("$BINARY" --model "$MODEL" --q3-experts --think-budget "$THINK_BUDGET" --serve "$PORT")
    if [[ "$cache_size" -gt 0 ]]; then
        CMD+=(--malloc-cache "$cache_size")
    fi
    # Always enable cache-entries as LRU fallback
    CMD+=(--cache-entries 2500)

    echo "Command: ${CMD[*]}"
    t_start=$(date +%s)

    # Start flash-moe
    "${CMD[@]}" > "$LOGFILE" 2>&1 &
    FM_PID=$!

    if ! wait_for_ready 300; then
        echo "SKIP: flash-moe failed to start with cache=$cache_size"
        kill_flashmoe
        continue
    fi

    t_ready=$(date +%s)
    startup_s=$((t_ready - t_start))
    echo "Startup: ${startup_s}s"

    # Warmup phase
    echo "Warming up ($WARMUP_PROMPTS prompts)..."
    for i in $(seq 1 "$WARMUP_PROMPTS"); do
        prompt_idx=$(( (i - 1) % ${#PROMPTS[@]} ))
        echo -n "  warmup $i/$WARMUP_PROMPTS..."
        send_prompt "${PROMPTS[$prompt_idx]}" "$MAX_TOKENS"
        echo " done"
    done

    # Benchmark phase
    echo "Benchmarking ($BENCH_PROMPTS prompts)..."
    for i in $(seq 1 "$BENCH_PROMPTS"); do
        prompt_idx=$(( (WARMUP_PROMPTS + i - 1) % ${#PROMPTS[@]} ))
        echo -n "  bench $i/$BENCH_PROMPTS..."
        send_prompt "${PROMPTS[$prompt_idx]}" "$MAX_TOKENS"
        echo " done"
    done

    # Extract metrics
    mapfile -t tps_values < <(extract_metrics "$LOGFILE" "$WARMUP_PROMPTS" | awk '{print $2}')

    if [[ ${#tps_values[@]} -eq 0 ]]; then
        echo "WARNING: No generation metrics found in log"
        mean_tps="0"
        median_tps="0"
        min_tps="0"
        max_tps="0"
    else
        # Calculate stats
        sorted=($(printf '%s\n' "${tps_values[@]}" | sort -g))
        n=${#sorted[@]}
        min_tps="${sorted[0]}"
        max_tps="${sorted[$((n-1))]}"
        median_tps="${sorted[$((n/2))]}"

        sum=0
        for v in "${tps_values[@]}"; do
            sum=$(echo "$sum + $v" | bc -l 2>/dev/null || echo "$sum")
        done
        mean_tps=$(echo "scale=2; $sum / $n" | bc -l 2>/dev/null || echo "0")
    fi

    # Cache stats
    cache_stats=$(extract_cache_stats "$LOGFILE")
    cache_gb=$(echo "scale=1; $cache_size * 5439488 / 1000000000" | bc -l 2>/dev/null || echo "0")

    # Parse hit/miss from log
    cache_hits=$(echo "$cache_stats" | grep -o '[0-9]* hits' | awk '{print $1}' || echo "0")
    cache_misses=$(echo "$cache_stats" | grep -o '[0-9]* misses' | awk '{print $1}' || echo "0")
    hit_rate=$(echo "$cache_stats" | grep -o '[0-9.]*%' | head -1 || echo "0%")

    echo ""
    echo "Results: mean=${mean_tps} tok/s, median=${median_tps}, range=[${min_tps}, ${max_tps}]"
    echo "Cache:   ${cache_gb} GB, hit_rate=${hit_rate}"
    echo "Startup: ${startup_s}s"

    # Write TSV row
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$cache_size" "$cache_gb" "$startup_s" \
        "$mean_tps" "$median_tps" "$min_tps" "$max_tps" \
        "${cache_hits:-0}" "${cache_misses:-0}" "${hit_rate:-0%}" \
        "off" >> "$RESULTS_FILE"

    kill_flashmoe
done

echo ""
echo "═══════════════════════════════════════════════════════"
echo "Results saved to: $RESULTS_FILE"
echo "═══════════════════════════════════════════════════════"
cat "$RESULTS_FILE" | column -t -s $'\t'
