---
bead: sylveste-14g
date: 2026-03-28
type: plan
complexity: C4
reviewed_by: 8-agent flux-drive synthesis
---

# Plan: SSD Streaming Inference for 700B+ MoE Models

## Context

8 review agents unanimously recommended Option D (mlx-lm fork with
StreamingSwitchGLU) over the brainstorm's Option A (parameterize flash-moe).
Key reasons: zero-copy Python path exists (mx.array memoryview + ctypes pread
= 47.5 GB/s measured), flash-moe Metal contexts are incompatible with MLX,
and the I/O bottleneck is SSD bandwidth not language overhead.

## Phase 0: Starter Experiment (Day 0, resolves core uncertainty)

**Goal**: Measure whether Python pread-based expert streaming on Qwen 397B
reaches within 20% of flash-moe's 11.1 tok/s. This one data point collapses
the entire A-vs-D decision.

### Task 0.1: Expert file preparation for Qwen 397B

Qwen 397B already has repacked expert files from flash-moe at
`~/Models/mlx-community-Qwen3.5-397B-A17B-4bit/packed_experts/`.
Verify these exist and note the layout (expert_size, num_experts, components).

If not available, run `repack_experts.py` from flash-moe to generate them.

### Task 0.2: Proof-of-concept StreamingSwitchGLU

Write a minimal `streaming_switch.py` that:

1. Pre-allocates mx.array buffer pool (K * num_components for double-buffering)
2. Overrides `SwitchGLU.__call__`:
   a. Run `self.gate(x)` to get routing scores
   b. `mx.eval(inds)` to synchronize — get expert indices on CPU
   c. pread selected experts from layer files via ctypes into mx.array memoryview
   d. Stack loaded experts, assign to projection weights
   e. Continue with standard SwitchGLU forward
3. Monkey-patch into Qwen's MoE layers at load time

### Task 0.3: Benchmark vs flash-moe

Run both on identical prompts (5 standard prompts, 200 tokens each):
- flash-moe: `./metal_infer/infer --model <path> --stream --max-tokens 200`
- Streaming mlx-lm: custom generate loop with StreamingSwitchGLU

Measure: tok/s decode, TTFT, peak memory, page cache hit rate (via vm_stat).

### Task 0.4: Decision gate

- **Within 20%**: Proceed with Option D (this plan). flash-moe is reference only.
- **20-30% gap**: Profile where time is lost. Fix if fixable (router sync, buffer copy).
- **>30% gap**: Reassess. Option A may be justified for the performance delta.

## Phase 1: StreamingSwitchGLU Implementation (Days 1-3)

### Task 1.1: Expert index generator

Python script that scans a model's safetensors files and produces an
`expert_layout.json` mapping each layer's experts to file offsets.
Similar to flash-moe's `repack_experts.py` but for the mlx-lm weight format.

Input: model directory (safetensors + config.json)
Output: `expert_layout.json` with per-layer expert offsets

### Task 1.2: Expert repacker

Repack scattered safetensors into one binary per MoE layer:
`packed_experts/layer_XX.bin` — experts at fixed offsets, contiguous.
Per-layer file layout (matching flash-moe's proven design).

### Task 1.3: StreamingSwitchGLU class

Full implementation with:
- Pre-allocated mx.array buffer pool via memoryview
- ThreadPoolExecutor(max_workers=4) for parallel pread
- Router sync point (`mx.eval(inds)`)
- Weight replacement via `mx.stack()` + assignment
- Shared expert pinning (427MB resident, always loaded)
- `--streaming-weights` CLI flag to enable

### Task 1.4: Wire into InferenceEngine

- New `streaming=True` parameter on `InferenceEngine.__init__`
- Load expert layout at init, open layer file descriptors
- Monkey-patch SwitchGLU at model load time
- MetalWorker protocol unchanged (GENERATE command works as-is)

### Task 1.5: Tests

- Unit test: expert index generation for Qwen safetensors layout
- Unit test: pread into mx.array memoryview correctness
- Unit test: StreamingSwitchGLU produces same output as standard SwitchGLU (on tiny model)
- Integration test: generate with streaming flag in dry-run mode

## Phase 2: Model Onboarding (Days 3-5)

### Task 2.1: DeepSeek V3.2 expert index + repack

Generate expert_layout.json and packed layer files for DeepSeek V3.2.
mlx-lm already has the model definition — no new attention code needed.

### Task 2.2: Benchmark DeepSeek V3.2

Target: 8+ tok/s. Measure decode, TTFT, quality (5 standard prompts).
Record page cache hit rate and compare to estimates (~85%).

### Task 2.3: GLM-5 expert index + repack + benchmark

Same process. Target: 6+ tok/s (borderline per I/O analysis).

### Task 2.4: Kimi K2.5 expert index + repack + benchmark

Same process. Target: 5+ tok/s (at risk per I/O analysis).
May need Q3 quantization for viable performance.

## Phase 3: Production Hardening (Days 5-7)

### Task 3.1: Performance fixes from fd-performance review

- Default `kv_bits=4` for 60L+ MoE models
- PromptCacheManager orphan cleanup at init
- Cap `_latency_samples` with deque(maxlen=1000)
- Batch confidence computation post-generation (remove per-token mx.eval sync)

### Task 3.2: Prometheus metrics for streaming

Add to prom.py:
- `interfere_ssd_pread_bytes_total` (counter)
- `interfere_expert_cache_hit_rate` (gauge, from vm_stat sampling)
- `interfere_streaming_io_seconds` (histogram, per-layer I/O time)

### Task 3.3: Cascade integration

When cascade is enabled with streaming models:
- Probe uses streaming path (same expert loading)
- Wire KV state handoff from probe to continuation (fix double-prefill bug)

### Task 3.4: Shadow cost logging for streaming models

Log to `local_routing_shadow` table with actual model used and tok/s achieved.
`infer_cloud_model()` already maps model sizes to cloud tiers.

## Phase 4: Parallel Track — flash-moe Reference (ongoing)

### Task 4.1: flash-moe subprocess benchmark harness

Use flash-moe's `--serve PORT` HTTP/SSE mode as a reference ceiling.
interfere proxies to it for A/B comparison.

### Task 4.2: Upstream contribution (Option E)

File design proposal to mlx-lm for native SSD streaming support.
Reference benchmark data from Phase 2. If accepted, migrate from
StreamingSwitchGLU monkey-patch to upstream API.

## File Change Summary

| File | Change |
|------|--------|
| `server/streaming_switch.py` | NEW — StreamingSwitchGLU, expert layout loader, pread pool |
| `server/inference.py` | Add streaming=True parameter, monkey-patch at load |
| `server/metal_worker.py` | Pass streaming flag through to InferenceEngine |
| `server/__main__.py` | Add --streaming-weights CLI flag |
| `server/prom.py` | Add streaming I/O metrics |
| `server/main.py` | Wire streaming flag, fix latency_samples cap |
| `server/prompt_cache.py` | Add orphan cleanup at init |
| `scripts/expert_index.py` | NEW — generate expert_layout.json from safetensors |
| `scripts/repack_experts.py` | NEW — repack into per-layer binary files |
| `tests/test_streaming_switch.py` | NEW — streaming unit + integration tests |

## Success Criteria

| Model | Target tok/s | Confidence |
|-------|-------------|------------|
| Qwen 397B (4-bit) | 9+ | High (baseline 11.1 via flash-moe) |
| DeepSeek V3.2 (4-bit) | 8+ | Medium-high (85% cache hit estimated) |
| GLM-5 (4-bit) | 6+ | Medium (borderline I/O budget) |
| Kimi K2.5 (3-bit) | 5+ | Medium-low (3.3x ratio, may need Q3) |

## Kill Criteria

- If Phase 0 shows >30% gap vs flash-moe: stop, reassess Option A
- If DeepSeek V3.2 < 5 tok/s after tuning: the approach is fundamentally limited
- If mx.array memoryview API breaks in future MLX: pin version, escalate upstream

## Risks

| Risk | Mitigation |
|------|------------|
| Router sync (mx.eval) adds >2ms/layer | Profile in Phase 0; if bad, explore async routing |
| Page cache cold-start on task switches | Pre-warm from Clavain task classifier signal |
| KV cache competes with page cache | Default kv_bits=4 for large models |
| mx.array memoryview API unstable | Pin mlx-lm version; pursue Option E upstream |
| Qwen repacked experts not available | Re-run flash-moe's repack_experts.py |
