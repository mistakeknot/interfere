"""Tests for the confidence cascade."""

from __future__ import annotations

from server.cascade import (
    CascadeConfig,
    CascadeDecision,
    CascadeResult,
    CascadeStats,
    ConfidenceCascade,
)


def test_cascade_decision_accept() -> None:
    """High confidence maps to ACCEPT."""
    cascade = ConfidenceCascade.__new__(ConfidenceCascade)
    cascade._config = CascadeConfig()
    assert cascade._decide(0.9) == CascadeDecision.ACCEPT
    assert cascade._decide(0.8) == CascadeDecision.ACCEPT


def test_cascade_decision_escalate() -> None:
    """Mid confidence maps to ESCALATE."""
    cascade = ConfidenceCascade.__new__(ConfidenceCascade)
    cascade._config = CascadeConfig()
    assert cascade._decide(0.7) == CascadeDecision.ESCALATE
    assert cascade._decide(0.6) == CascadeDecision.ESCALATE


def test_cascade_decision_cloud() -> None:
    """Low confidence maps to CLOUD."""
    cascade = ConfidenceCascade.__new__(ConfidenceCascade)
    cascade._config = CascadeConfig()
    assert cascade._decide(0.5) == CascadeDecision.CLOUD
    assert cascade._decide(0.1) == CascadeDecision.CLOUD


def test_cascade_stats_rates() -> None:
    """Stats compute rates correctly."""
    stats = CascadeStats(
        total_requests=10,
        accepts=6,
        escalations=3,
        cloud_fallbacks=1,
        total_probe_time_s=2.0,
    )
    assert stats.accept_rate == 0.6
    assert stats.escalation_rate == 0.3
    assert stats.cloud_rate == 0.1
    d = stats.to_dict()
    assert d["accept_rate"] == 0.6


def test_cascade_stats_empty() -> None:
    """Empty stats return zero rates."""
    stats = CascadeStats()
    assert stats.accept_rate == 0.0
    assert stats.cloud_rate == 0.0


def test_cascade_config_defaults() -> None:
    """Default config has expected thresholds."""
    cfg = CascadeConfig()
    assert cfg.accept_threshold == 0.8
    assert cfg.cloud_threshold == 0.6
    assert cfg.probe_tokens == 3
    assert cfg.enabled is True


def test_cascade_result_fields() -> None:
    """CascadeResult holds all expected fields."""
    r = CascadeResult(
        decision=CascadeDecision.ACCEPT,
        model_used="local:test",
        probe_confidence=0.85,
        probe_tokens=["Hello", " world"],
        probe_time_s=0.05,
        models_tried=["local:test"],
        escalation_count=0,
    )
    assert r.decision == CascadeDecision.ACCEPT
    assert r.model_used == "local:test"
    assert len(r.probe_tokens) == 2
