---
artifact_type: reflection
bead: sylveste-qbv
stage: reflect
---

# Reflection: LayerSkip Self-Speculative Early Exit — PoC Results

**Bead:** sylveste-qbv
**Date:** 2026-04-07

## What happened

Built a self-speculative decoding PoC for Qwen3.5-122B-A10B (48 layers, MoE).
Ran partial forward passes through first N layers, applied norm + lm_head to get
draft logits, checked confidence against 0.95 threshold.

**Result: 0% acceptance rate across all 16 configurations (4 prompts × 3 exit layers).**

No draft token was confident enough at the intermediate layer to be accepted.
This means intermediate layers of Qwen3.5 MoE models do NOT produce useful
logits through the final lm_head — the model distributes its reasoning across
all layers, with no clean "early exit point."

## Full results

| Prompt | Standard | Exit=12 | Exit=24 | Exit=36 |
|--------|---------|---------|---------|---------|
| lcs_algorithm | 4.5 tok/s | 6.2 (0%) | 6.2 (0%) | 5.7 (0%) |
| code_review | 6.4 tok/s | 7.4 (0%) | 8.0 (0%) | 8.6 (0%) |
| refactor | 10.7 tok/s | 8.4 (0%) | 5.5 (0%) | 4.3 (0%) |
| system_design | 2.2 tok/s | 1.0 (0%) | 0.4 (0%) | 0.4 (0%) |

## What we learned

1. **The confidence profile was misleading.** The earlier calibration showed 67.5%
   of 122B tokens have >99% confidence — but that measures confidence at the
   FINAL layer (after all 48 layers). Intermediate layers have much lower
   confidence because the model hasn't finished processing. LayerSkip only
   works when intermediate layers can independently produce confident predictions,
   which Qwen3.5's architecture doesn't support.

2. **MoE expert routing distributes computation.** In a dense model, early layers
   might capture "easy" token patterns. In MoE, the routing decision at each
   layer sends the token to specialized experts. The model's representation is
   built incrementally across all expert choices — there's no "early completion"
   point.

3. **Manual layer iteration has different performance characteristics.** The PoC
   sometimes ran faster than mlx-lm's `generate()` (lcs: 6.2 vs 4.5, code_review:
   8.6 vs 6.4) because it bypasses mlx-lm's chat template processing, sampler
   overhead, and repetition penalty. Sometimes slower (system_design: 0.4 vs 2.2)
   due to cache management overhead in our manual loop.

4. **LayerSkip requires model-specific training.** The original LayerSkip paper
   (arXiv 2404.16710) uses early exit loss during training to teach intermediate
   layers to produce useful logits. Without this training, the technique cannot
   work on pretrained models like Qwen3.5.

## Decision

**Close sylveste-qbv as not-viable for pretrained Qwen3.5 MoE models.**

LayerSkip requires training-time support (early exit loss) that Qwen3.5 doesn't
have. The technique is architecturally feasible in MLX (partial forward passes
work) but the model's representations aren't structured for early exit.

**Future paths if this becomes important:**
- Fine-tune with early exit loss on a Qwen3.5 variant (significant investment)
- Standard speculative decoding with a separate draft model (already supported via `draft_model_name`)
- Wait for models explicitly trained with LayerSkip support
