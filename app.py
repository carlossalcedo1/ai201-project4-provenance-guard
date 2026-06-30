"""Provenance Guard — Flask application.

M3 skeleton: the POST /submit route, input validation, and the JSON response
shape from planning.md > Architecture. The first detection signal (Signal A —
structural) is wired in; the rest of the pipeline is stubbed with TODOs that
name the milestone that fills them:

    M4 -> Signal B (LLM judge), fusion, real confidence + verdict
    M5 -> transparency labels, /appeal endpoint, audit log, rate limiting

The response intentionally already has the final shape (result, confidence,
label, content_id, breakdown) so callers don't have to change when M4/M5 land.
"""

from __future__ import annotations

import uuid

from dotenv import load_dotenv
from flask import Flask, jsonify, request

import store
from signals import structural

load_dotenv()  # GROQ_API_KEY (used by Signal B in M4); harmless here

app = Flask(__name__)

# --- Input validation bounds (planning.md > Architecture, step 3) -------------
MIN_WORDS = 5        # below this the structural signal has nothing to measure
MAX_CHARS = 20_000   # bounds abuse and (in M4) LLM cost

# Placeholders until the real pipeline lands (see planning.md milestones).
PLACEHOLDER_CONFIDENCE = 0.5  # TODO(M4): replace with fused confidence
PLACEHOLDER_LABEL = "[placeholder — final transparency label added in M5]"

# TODO(M5): wrap /submit with flask-limiter using the limits documented in README.


def _word_count(text: str) -> int:
    return len(text.split())


def _provisional_attribution(p_ai: float) -> str:
    """Single-signal, provisional attribution from Signal A only (M3).

    Mirrors the directional thresholds in planning.md (>=0.65 ai, <=0.35 human,
    else uncertain) so the response is meaningful now. M4 fusion replaces this
    with the real cross-signal verdict.
    """
    if p_ai >= 0.65:
        return "likely_ai"
    if p_ai <= 0.35:
        return "likely_human"
    return "uncertain"


@app.post("/submit")
def submit():
    """Accept text for attribution analysis and return a structured result.

    M3 returns Signal A's score in the breakdown. `confidence`, `verdict`, and
    `label` are provisional placeholders until fusion (M4) and labels (M5).
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "request body must be JSON"}), 400
    if "text" not in data or "creator_id" not in data:
        return jsonify({"error": "JSON body must include 'text' and 'creator_id'"}), 400

    text = data["text"]
    creator_id = data["creator_id"]
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "'text' must be a non-empty string"}), 400
    if not isinstance(creator_id, str) or not creator_id.strip():
        return jsonify({"error": "'creator_id' must be a non-empty string"}), 400
    if _word_count(text) < MIN_WORDS:
        return jsonify({"error": f"'text' must be at least {MIN_WORDS} words"}), 400
    if len(text) > MAX_CHARS:
        return jsonify({"error": f"'text' must be at most {MAX_CHARS} characters"}), 400

    # --- Detection: Signal A (structural). Signal B + fusion arrive in M4. -----
    signal_a = structural.analyze(text)

    content_id = str(uuid.uuid4())

    # Provisional attribution from Signal A alone; M4 fusion makes it final.
    attribution = _provisional_attribution(signal_a["p_ai"])

    response = {
        "content_id": content_id,
        "creator_id": creator_id,             # echoed back; persisted to audit log in M5
        "attribution": attribution,           # from Signal A only until M4 fusion
        "confidence": PLACEHOLDER_CONFIDENCE,  # TODO(M4): real confidence from fusion
        "label": PLACEHOLDER_LABEL,            # TODO(M5): real transparency label text
        "breakdown": {
            "structural": signal_a,
            # TODO(M4): "llm_judge": {...}
        },
    }

    # Audit log: every submission writes one structured entry (M4 adds llm_score).
    store.record_decision(
        content_id=content_id,
        creator_id=creator_id,
        attribution=attribution,
        confidence=PLACEHOLDER_CONFIDENCE,
        structural_score=signal_a["p_ai"],
        status="classified",
    )

    return jsonify(response), 200


@app.get("/log")
def log():
    """Return the most recent audit-log entries (open for grading visibility)."""
    return jsonify({"entries": store.get_log()}), 200


@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(debug=True)
