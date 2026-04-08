"""Entropy-based early exit hook and self-speculative LayerSkip PoC."""

from __future__ import annotations

import time

import mlx.core as mx


class EarlyExitHook:
    """Decides whether to exit generation early based on token confidence.

    When the model is highly confident in its next-token prediction
    (softmax probability exceeds *threshold*), downstream layers can be
    skipped to save compute.
    """

    def __init__(self, threshold: float = 0.95, enabled: bool = True) -> None:
        self.threshold = threshold
        self.enabled = enabled
        self._exit_count = 0
        self._total_count = 0

    def check(self, logits: mx.array) -> tuple[bool, float]:
        """Evaluate whether early exit is warranted for *logits*.

        Returns a ``(should_exit, confidence)`` tuple where *confidence*
        is the maximum softmax probability across the vocabulary.
        """
        probs = mx.softmax(logits, axis=-1)
        confidence = float(mx.max(probs))

        self._total_count += 1
        should_exit = self.enabled and confidence > self.threshold
        if should_exit:
            self._exit_count += 1

        return should_exit, confidence

    @property
    def exit_rate(self) -> float:
        """Fraction of checks that triggered an early exit."""
        if self._total_count == 0:
            return 0.0
        return self._exit_count / self._total_count

    def reset_stats(self) -> None:
        """Zero both counters."""
        self._exit_count = 0
        self._total_count = 0


def self_speculative_generate(
    model,
    tokenizer,
    prompt: str,
    exit_layer: int = 32,
    confidence_threshold: float = 0.95,
    max_tokens: int = 100,
) -> dict:
    """Self-speculative decoding PoC (LayerSkip arXiv 2404.16710).

    Runs a partial forward pass through the first *exit_layer* layers to
    produce draft logits.  When the model is confident (above *threshold*),
    the draft token is accepted directly.  Otherwise the remaining layers
    run to produce a verified token.

    Returns a dict with: text, tokens, accepted, verified, acceptance_rate,
    tok_per_sec, elapsed_s.
    """
    from mlx_lm.models.cache import make_prompt_cache

    tokens = mx.array(tokenizer.encode(prompt))[None]  # (1, seq_len)
    num_layers = len(model.model.layers)
    if exit_layer >= num_layers:
        raise ValueError(f"exit_layer={exit_layer} must be < num_layers={num_layers}")

    cache = make_prompt_cache(model)
    norm = model.model.norm
    # lm_head: tied embeddings or separate linear
    if hasattr(model, "lm_head"):
        lm_head = model.lm_head
    else:
        lm_head = model.model.embed_tokens.as_linear

    # --- Prefill: run ALL layers on the full prompt ---
    h = model.model.embed_tokens(tokens)
    for i, layer in enumerate(model.model.layers):
        h = layer(h, mask=None, cache=cache[i])
    h = norm(h)
    logits = lm_head(h[:, -1:, :])
    next_token = mx.argmax(logits, axis=-1)
    # Materialize lazy MLX computation graph (not Python eval)
    mx.eval(next_token)  # noqa: S307

    generated = [next_token.item()]
    accepted_count = 0
    verified_count = 0
    t0 = time.monotonic()

    for _ in range(max_tokens - 1):
        tok = next_token
        # --- Draft pass: first exit_layer layers ---
        h = model.model.embed_tokens(tok)
        for i in range(exit_layer):
            h = model.model.layers[i](h, mask=None, cache=cache[i])

        draft_logits = lm_head(norm(h))
        draft_probs = mx.softmax(draft_logits, axis=-1)
        draft_confidence = float(mx.max(draft_probs))
        draft_token = mx.argmax(draft_logits, axis=-1)

        if draft_confidence >= confidence_threshold:
            # Accept draft — still run remaining layers to keep KV cache consistent.
            # In a full implementation, we'd skip these layers and use a cache
            # fill strategy instead. For the PoC, this measures acceptance rate
            # accurately while maintaining cache correctness.
            for i in range(exit_layer, num_layers):
                h = model.model.layers[i](h, mask=None, cache=cache[i])
            next_token = draft_token
            accepted_count += 1
        else:
            # Verify — run remaining layers
            for i in range(exit_layer, num_layers):
                h = model.model.layers[i](h, mask=None, cache=cache[i])
            full_logits = lm_head(norm(h))
            next_token = mx.argmax(full_logits, axis=-1)
            verified_count += 1

        # Materialize lazy MLX computation graph (not Python eval)
        mx.eval(next_token)  # noqa: S307
        tok_id = next_token.item()
        generated.append(tok_id)

        # Stop on EOS
        if tok_id == tokenizer.eos_token_id:
            break

    elapsed = time.monotonic() - t0
    total = accepted_count + verified_count
    text = tokenizer.decode(generated)

    return {
        "text": text,
        "tokens": len(generated),
        "accepted": accepted_count,
        "verified": verified_count,
        "acceptance_rate": accepted_count / total if total > 0 else 0,
        "tok_per_sec": len(generated) / elapsed if elapsed > 0 else 0,
        "elapsed_s": round(elapsed, 3),
        "exit_layer": exit_layer,
        "confidence_threshold": confidence_threshold,
    }
