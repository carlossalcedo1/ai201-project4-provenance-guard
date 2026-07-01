"""Audit store — structured, file-backed log of every attribution decision.

M3: writes one entry per /submit (timestamp, content_id, creator_id,
attribution, confidence, structural_score, status). Extended in M4 (llm_score +
fused values) and M5 (appeals append to the same record).

Persistence is a JSON file on disk so GET /log survives restarts and the log is
a real artifact rather than in-memory state. This is not concurrency-safe under
heavy parallel writes — acceptable for this project and documented as a known
limitation.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

_LOG_PATH = os.path.join(os.path.dirname(__file__), "audit_log.json")


def _utc_now_iso() -> str:
    """ISO-8601 UTC timestamp with millisecond precision and a trailing 'Z'."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def _read_all() -> list[dict]:
    if not os.path.exists(_LOG_PATH):
        return []
    try:
        with open(_LOG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # Corrupt/unreadable log should not crash the endpoint.
        return []


def _write_all(entries: list[dict]) -> None:
    with open(_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def record_decision(
    *,
    content_id: str,
    creator_id: str,
    attribution: str,
    combined_score: float,
    confidence: float,
    structural_score: float,
    llm_score: float | None = None,
    status: str = "classified",
) -> dict:
    """Append one classification decision to the audit log and return the entry."""
    entry = {
        "content_id": content_id,
        "creator_id": creator_id,
        "timestamp": _utc_now_iso(),
        "attribution": attribution,            # combined verdict
        "combined_score": combined_score,      # fused p_ai (both signals combined)
        "confidence": confidence,              # combined confidence
        "structural_score": structural_score,  # Signal 1 (structural) score
        "llm_score": llm_score,                # Signal 2 (LLM judge) score
        "status": status,
    }
    entries = _read_all()
    entries.append(entry)
    _write_all(entries)
    return entry


def get_log(limit: int = 50) -> list[dict]:
    """Return the most recent entries, newest first."""
    return list(reversed(_read_all()))[:limit]
