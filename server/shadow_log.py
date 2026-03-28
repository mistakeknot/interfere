"""Shadow cost logging for interstat.

Logs counterfactual routing decisions from the confidence cascade to
interstat's local_routing_shadow table. This data feeds sprint cost
summaries showing how much cloud spend was avoided by local routing.

The logger writes directly to SQLite (not via hooks) since it runs
inside the interfere server process, not a Claude Code session.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from dataclasses import dataclass

log = logging.getLogger("interfere.shadow_log")

# Default cloud model used for cost estimation when no specific model is known
DEFAULT_CLOUD_MODEL = "claude-sonnet-4-6"

# API pricing per million tokens (matches interstat/scripts/cost-query.sh defaults)
_PRICING: dict[str, tuple[float, float]] = {
    # (input_per_mtok, output_per_mtok)
    "haiku": (0.80, 4.00),
    "sonnet": (3.00, 15.00),
    "opus": (15.00, 75.00),
}


def _cloud_cost_usd(model: str, tokens: int) -> float:
    """Estimate cloud cost for a given model and token count.

    Uses output pricing (conservative estimate since we're estimating
    what the cloud model would have generated).
    """
    key = "sonnet"  # default
    model_lower = model.lower()
    for tier in ("haiku", "sonnet", "opus"):
        if tier in model_lower:
            key = tier
            break
    _, output_rate = _PRICING[key]
    return round(tokens * output_rate / 1_000_000, 6)


@dataclass
class ShadowEntry:
    """A single cascade routing decision to log."""

    cascade_decision: str  # "accept", "escalate", "cloud"
    confidence: float
    local_model: str
    local_tokens: int
    cloud_model: str = DEFAULT_CLOUD_MODEL
    cloud_tokens_est: int = 0
    probe_time_s: float = 0.0
    models_tried: str = ""
    escalation_count: int = 0
    session_id: str = ""
    bead_id: str = ""


class ShadowLogger:
    """Writes cascade routing decisions to interstat's SQLite DB."""

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or os.path.join(
            os.path.expanduser("~"), ".claude", "interstat", "metrics.db"
        )
        self._conn: sqlite3.Connection | None = None

    def _ensure_db(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        try:
            self._conn = sqlite3.connect(self._db_path, timeout=5.0)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
        except Exception:
            log.warning("shadow_log: cannot connect to %s", self._db_path)
            raise
        return self._conn

    def log(self, entry: ShadowEntry) -> None:
        """Log a shadow routing entry. Fails silently on DB errors."""
        try:
            conn = self._ensure_db()
        except Exception:
            return  # DB not available — degrade gracefully

        # Estimate cloud tokens as same as local (conservative)
        cloud_tokens = entry.cloud_tokens_est or entry.local_tokens
        cloud_cost = _cloud_cost_usd(entry.cloud_model, cloud_tokens)
        local_cost = 0.0  # local inference has no API cost
        savings = cloud_cost - local_cost

        try:
            conn.execute(
                """INSERT INTO local_routing_shadow (
                    timestamp, session_id, bead_id,
                    cascade_decision, confidence,
                    local_model, local_tokens,
                    cloud_model, cloud_tokens_est,
                    local_cost_usd, cloud_cost_usd, hypothetical_savings_usd,
                    probe_time_s, models_tried, escalation_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    entry.session_id,
                    entry.bead_id,
                    entry.cascade_decision,
                    entry.confidence,
                    entry.local_model,
                    entry.local_tokens,
                    entry.cloud_model,
                    cloud_tokens,
                    local_cost,
                    cloud_cost,
                    savings,
                    entry.probe_time_s,
                    entry.models_tried,
                    entry.escalation_count,
                ),
            )
            conn.commit()
        except Exception as exc:
            log.warning("shadow_log: INSERT failed: %s", exc)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
