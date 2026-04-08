---
artifact_type: brainstorm
bead: sylveste-qbv
stage: brainstorm
---

# Brainstorm: LayerSkip / Self-Speculative Early Exit — Feasibility Probe

**Bead:** sylveste-qbv
**Date:** 2026-04-07

## Prior Work in This Bead

- Confidence profiles calibrated across 3 models (0.8B, 35B MoE, 122B MoE)
- 67.5% of 122B tokens have >99% confidence — strong LayerSkip candidate
- Coding/instruction tasks most confident (67-84% >99%)
- Theoretical speedup: ~1.5x if 50% of layers skippable for high-confidence tokens

## Feasibility Assessment

### Architecture Support: FEASIBLE

MLX and mlx-lm fully support partial forward passes:

1. **Layer access**: `model.model.layers[i]` gives direct access to any layer
2. **Partial iteration**: Already proven in `inference.py:137-172` (`_extract_hidden_state`)
3. **lm_head access**: `model.lm_head(h)` or `model.model.embed_tokens.as_linear(h)` for tied embeddings
4. **Norm requirement**: Must apply `model.model.norm(h)` before lm_head — the final RMSNorm normalizes hidden states into the projection space (line 268 of qwen3_5.py)

### Self-Speculative Decoding Protocol (LayerSkip)

```
for each token:
  1. Run DRAFT pass: layers[0..N-1] → norm → lm_head → draft_logit
  2. Check confidence: if max(softmax(draft_logit)) > threshold → ACCEPT draft token
  3. If not confident: run VERIFY pass: layers[N..end] on same hidden state → full logits
  4. Use verified token
```

**Key advantage over standard speculative decoding**: shares the first N layers between draft and verify. No separate draft model needed. KV cache for layers 0..N-1 is reused.

### Blockers and Risks

#### B1: Mask handling for Qwen3.5 hybrid layers (MEDIUM)

Qwen3.5 uses mixed SSM (linear attention) and full-attention layers. Each layer type needs a different mask:
- Full attention layers: `create_attention_mask(h, cache[fa_idx])`
- Linear (SSM) layers: `create_ssm_mask(h, cache[ssm_idx])`

The existing `_extract_hidden_state` passes `mask=None` which may work for prompt processing but could fail during decode. Need to replicate the mask logic from `Qwen3_5TextModel.__call__`.

#### B2: Cache sharing between draft and verify (LOW)

LayerSkip requires that the KV cache from the draft pass (layers 0..N-1) be reused in the verify pass. MLX's `KVCache` and `ArraysCache` objects support this — they accumulate key/value states that persist across calls.

#### B3: MoE expert routing consistency (LOW)

For MoE layers, the draft pass routes to specific experts based on the hidden state at that layer. The verify pass (continuing from the same hidden state) will route to the same experts. No inconsistency.

#### B4: No concurrent inference (MEDIUM)

MLX's single-threaded Metal inference means we can't run draft and verify in parallel. Each pass is sequential. The speedup comes from skipping layers entirely for high-confidence tokens, not from parallelism.

#### B5: Which layer to exit at? (RESEARCH QUESTION)

The LayerSkip paper uses ~50% of layers for the draft pass. For Qwen3.5-35B (64 layers), that's layer 32. But:
- Earlier exit = faster draft but less accurate → more verifications needed
- Later exit = slower draft but more accurate → fewer verifications
- Optimal exit layer depends on the model and task type

The confidence profile data shows coding tasks have 67-84% tokens above 99% confidence, suggesting aggressive early exit (layer ~20-24) could work for coding-heavy workloads.

## Implementation Approach

### Phase 1: Proof of Concept (~50 lines)

Add a `self_speculative_generate` function to `inference.py` that:
1. Loads a model via mlx-lm
2. For each token, runs partial forward pass through first N layers
3. Applies norm + lm_head to get draft logits
4. If confident (>threshold), accepts draft token
5. If not, runs remaining layers for verified token
6. Measures: speedup, acceptance rate, quality (perplexity change)

### Phase 2: Integration

If PoC shows >1.2x speedup with >0.95 acceptance rate:
- Wire into `InferenceEngine.generate()` as an optional mode
- Expose via `--early-exit-layer N` CLI flag
- Track via interlab campaign (sylveste-9tc)

### Phase 3: Tuning (sylveste-9tc)

- Sweep exit layer (16, 24, 32, 40 for 64-layer model)
- Sweep confidence threshold (0.8, 0.9, 0.95, 0.99)
- Compare quality vs speedup Pareto frontier

## Decision

**FEASIBLE — proceed to PoC.** All architectural requirements are met by MLX/mlx-lm. The main risk is B1 (mask handling) which is solvable by replicating the Qwen3.5TextModel mask logic. The speedup potential (1.3-1.5x on coding tasks) justifies the ~50-line PoC investment.
