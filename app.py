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
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import labels
import pipeline
import store

load_dotenv()  # GROQ_API_KEY for Signal B (LLM judge)

app = Flask(__name__)

# Rate limiting: protect /submit because every call triggers a cost-bearing Groq
# request. Keyed by client IP; in-memory storage is fine for this single-process
# dev/grading setup. Chosen limits + reasoning are documented in the README.
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)
SUBMIT_RATE_LIMIT = "10 per minute;100 per day"

# --- Input validation bounds (planning.md > Architecture, step 3) -------------
MIN_WORDS = 5        # below this the structural signal has nothing to measure
MAX_CHARS = 20_000   # bounds abuse and LLM cost


def _word_count(text: str) -> int:
    return len(text.split())


@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "rate limit exceeded", "detail": str(e.description)}), 429


@app.post("/submit")
@limiter.limit(SUBMIT_RATE_LIMIT)
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
        "label": labels.build_label(result["attribution"], result["confidence"]),
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


@app.post("/appeal")
def appeal():
    """Contest a classification (planning.md > Appeals workflow).

    Body: content_id (required), creator_reasoning (required), optional creator_id
    (enforced only if supplied), optional claimed_origin ("human"/"ai"). On success
    the decision's status flips to "under_review" and the appeal is logged beside
    it. No automated re-classification — a human reviews.
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "request body must be JSON"}), 400

    content_id = data.get("content_id")
    creator_reasoning = data.get("creator_reasoning") or data.get("reason")
    creator_id = data.get("creator_id")  # optional
    claimed_origin = data.get("claimed_origin")
    if not isinstance(content_id, str) or not content_id.strip():
        return jsonify({"error": "'content_id' is required"}), 400
    if not isinstance(creator_reasoning, str) or not creator_reasoning.strip():
        return jsonify({"error": "'creator_reasoning' is required"}), 400
    if claimed_origin is not None and claimed_origin not in ("human", "ai"):
        return jsonify({"error": "'claimed_origin' must be 'human' or 'ai'"}), 400

    result = store.record_appeal(
        content_id=content_id,
        creator_reasoning=creator_reasoning,
        creator_id=creator_id,
        claimed_origin=claimed_origin,
    )
    if result is None:
        return jsonify({"error": "no decision found for that content_id"}), 404
    if result == store.FORBIDDEN:
        return jsonify({"error": "creator_id does not match the original submission"}), 403

    return jsonify({
        "message": "appeal received",
        "content_id": content_id,
        "status": result["status"],
        "appeal_reasoning": creator_reasoning,
    }), 200


@app.get("/log")
def log():
    """Return the most recent audit-log entries (open for grading visibility)."""
    return jsonify({"entries": store.get_log()}), 200


@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(debug=True)
