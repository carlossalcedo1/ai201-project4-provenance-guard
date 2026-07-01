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
        "type": "classification",
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


def get_decision(content_id: str) -> dict | None:
    """Return the classification entry for `content_id`, or None if not found."""
    for entry in _read_all():
        if entry.get("type") == "classification" and entry.get("content_id") == content_id:
            return entry
    return None


# Sentinel returned by record_appeal when the creator_id doesn't match.
FORBIDDEN = "forbidden"


def record_appeal(
    *,
    content_id: str,
    creator_reasoning: str,
    creator_id: str | None = None,
    claimed_origin: str | None = None,
):
    """Log an appeal against an existing decision.

    Returns:
        None          if no classification exists for `content_id` (-> 404)
        FORBIDDEN     if `creator_id` is supplied but doesn't match (-> 403)
        the updated decision dict on success (-> 200)

    On success it (a) appends the appeal to the decision and flips its status to
    "under_review", and (b) writes a separate audit entry of type "appeal" that
    references the original decision and copies its verdict/scores.
    """
    entries = _read_all()
    decision = next(
        (e for e in entries
         if e.get("type") == "classification" and e.get("content_id") == content_id),
        None,
    )
    if decision is None:
        return None
    # Optional authorization: enforce a creator_id match only if one was supplied.
    if creator_id is not None and decision.get("creator_id") != creator_id:
        return FORBIDDEN

    ts = _utc_now_iso()
    appeal_record = {
        "appeal_reasoning": creator_reasoning,
        "claimed_origin": claimed_origin,
        "timestamp": ts,
    }

    # (a) attach to the original decision + change status
    decision.setdefault("appeals", []).append(appeal_record)
    decision["status"] = "under_review"

    # (b) separate audit entry referencing the original decision
    entries.append({
        "type": "appeal",
        "content_id": content_id,
        "creator_id": decision.get("creator_id"),
        "timestamp": ts,
        "appeal_reasoning": creator_reasoning,
        "claimed_origin": claimed_origin,
        "status": "under_review",
        "original_attribution": decision.get("attribution"),
        "original_combined_score": decision.get("combined_score"),
        "original_confidence": decision.get("confidence"),
        "original_structural_score": decision.get("structural_score"),
        "original_llm_score": decision.get("llm_score"),
    })

    _write_all(entries)
    return decision


def get_log(limit: int = 50) -> list[dict]:
    """Return the most recent entries, newest first."""
    return list(reversed(_read_all()))[:limit]
