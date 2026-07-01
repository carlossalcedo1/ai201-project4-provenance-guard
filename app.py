"""Provenance Guard — Flask application.

POST /submit runs the full two-signal detection pipeline (structural + LLM
judge), fuses them into a verdict + confidence (planning.md > Uncertainty
representation), and writes a structured audit entry. GET /log surfaces the log.

Still stubbed for M5:
    - the transparency label is a placeholder (real text in M5)
    - the /appeal endpoint and flask-limiter rate limiting
"""

from __future__ import annotations

import uuid

from dotenv import load_dotenv
from flask import Flask, jsonify, request

import pipeline
import store

load_dotenv()  # GROQ_API_KEY for Signal B (LLM judge)

app = Flask(__name__)

# --- Input validation bounds (planning.md > Architecture, step 3) -------------
MIN_WORDS = 5        # below this the structural signal has nothing to measure
MAX_CHARS = 20_000   # bounds abuse and LLM cost

# Placeholder until the label builder lands in M5.
PLACEHOLDER_LABEL = "[placeholder — final transparency label added in M5]"

# TODO(M5): wrap /submit with flask-limiter using the limits documented in README.


def _word_count(text: str) -> int:
    return len(text.split())


@app.post("/submit")
def submit():
    """Accept text for attribution analysis and return a structured result.

    Returns the fused attribution + real confidence; `label` stays a placeholder
    until M5.
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

    # --- Detection: both signals, fused (planning.md > Uncertainty rep.) -------
    result = pipeline.analyze(text)

    content_id = str(uuid.uuid4())

    response = {
        "content_id": content_id,
        "creator_id": creator_id,             # echoed back; persisted to audit log
        "attribution": result["attribution"],
        "confidence": result["confidence"],
        "label": PLACEHOLDER_LABEL,            # TODO(M5): real transparency label text
        "breakdown": result["breakdown"],
    }

    # Audit log: one structured entry per submission.
    store.record_decision(
        content_id=content_id,
        creator_id=creator_id,
        attribution=result["attribution"],
        combined_score=result["p_ai"],
        confidence=result["confidence"],
        structural_score=result["breakdown"]["structural"]["p_ai"],
        llm_score=result["breakdown"]["llm_judge"]["p_ai"],
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
