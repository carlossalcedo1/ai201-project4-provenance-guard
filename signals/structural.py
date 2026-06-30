"""Signal A — Structural statistics (deterministic, local, no network).

From planning.md > Detection signals > Signal A:

    Measures the *rhythm* of the writing — primarily the variance/standard
    deviation of sentence lengths ("burstiness"), supported by a lexical
    repetition / diversity measure. Human writing is bursty (uneven sentence
    rhythm + richer vocabulary); autoregressive models smooth output into
    uniform, medium-length sentences (low burstiness). Uniformity is the tell.

Output (per the Uncertainty-representation section):
    p_ai_A : float in [0, 1]   -- probability the text is AI-generated
    evidence : dict            -- the raw measurements behind the score

This signal is deterministic: the same input always yields the same score.
The thresholds below are initial defaults and will move to config.py and be
tuned against the fixture set in M4.
"""

from __future__ import annotations

import re
import statistics

# --- Tunable constants (initial defaults; will move to config.py in M4) -------
# Burstiness is measured as the coefficient of variation (CV = std / mean) of
# sentence lengths. Low CV  -> uniform -> AI-like; high CV -> bursty -> human-like.
CV_AI_BELOW = 0.25     # CV at/below this maps to p_ai ~ 1.0 (very uniform)
CV_HUMAN_ABOVE = 0.70  # CV at/above this maps to p_ai ~ 0.0 (very bursty)

# Type-token ratio (unique words / total words). Low TTR -> repetitive -> mildly
# AI-leaning; high TTR -> diverse vocabulary -> human-leaning. Weaker, weighted low.
TTR_AI_BELOW = 0.40
TTR_HUMAN_ABOVE = 0.70

# Blend weights: burstiness is the primary feature, TTR is a weak supporting one.
W_BURST = 0.7
W_TTR = 0.3

# Below this many sentences, burstiness is meaningless (planning.md blind spot:
# "Short texts have too few sentences for variance to mean anything").
MIN_SENTENCES = 3


def _split_sentences(text: str) -> list[str]:
    """Split into sentences on ., !, ? (and newlines, for poems/line-broken text)."""
    parts = re.split(r"[.!?]+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _words(text: str) -> list[str]:
    """Lowercased word tokens (letters/numbers/apostrophes)."""
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def _linear_map(value: float, lo: float, hi: float) -> float:
    """Map `value` to [0, 1] where value<=lo -> 1.0 and value>=hi -> 0.0.

    Used so that a *low* measurement (uniform / repetitive = AI-like) yields a
    *high* p_ai. Linear in between, clamped at the ends.
    """
    if value <= lo:
        return 1.0
    if value >= hi:
        return 0.0
    return (hi - value) / (hi - lo)


def analyze(text: str) -> dict:
    """Score `text` on structural grounds.

    Returns a dict:
        {
            "signal": "structural",
            "p_ai": float in [0, 1],
            "evidence": { ...raw measurements... },
        }

    For text too short to measure (fewer than MIN_SENTENCES sentences), returns
    p_ai = 0.5 with evidence flagging the insufficiency, so the downstream
    fusion step (M4) can treat it as low-confidence rather than a real verdict.
    """
    sentences = _split_sentences(text)
    sentence_lengths = [len(_words(s)) for s in sentences]
    sentence_lengths = [n for n in sentence_lengths if n > 0]
    words = _words(text)

    # Guard: not enough structure to judge.
    if len(sentence_lengths) < MIN_SENTENCES or len(words) < 2:
        return {
            "signal": "structural",
            "p_ai": 0.5,
            "evidence": {
                "sufficient_text": False,
                "reason": f"need >= {MIN_SENTENCES} sentences and >= 2 words",
                "sentence_count": len(sentence_lengths),
                "word_count": len(words),
            },
        }

    # --- Feature 1: burstiness (coefficient of variation of sentence lengths) ---
    mean_len = statistics.mean(sentence_lengths)
    std_len = statistics.pstdev(sentence_lengths)  # population std (deterministic)
    cv = (std_len / mean_len) if mean_len else 0.0
    burst_p_ai = _linear_map(cv, CV_AI_BELOW, CV_HUMAN_ABOVE)

    # --- Feature 2: lexical diversity (type-token ratio) ------------------------
    ttr = len(set(words)) / len(words)
    ttr_p_ai = _linear_map(ttr, TTR_AI_BELOW, TTR_HUMAN_ABOVE)

    # --- Blend into a single probability ---------------------------------------
    p_ai = W_BURST * burst_p_ai + W_TTR * ttr_p_ai
    p_ai = max(0.0, min(1.0, p_ai))

    return {
        "signal": "structural",
        "p_ai": round(p_ai, 4),
        "evidence": {
            "sufficient_text": True,
            "sentence_count": len(sentence_lengths),
            "word_count": len(words),
            "mean_sentence_length": round(mean_len, 2),
            "sentence_length_cv": round(cv, 4),
            "type_token_ratio": round(ttr, 4),
            "burstiness_subscore": round(burst_p_ai, 4),
            "ttr_subscore": round(ttr_p_ai, 4),
        },
    }


if __name__ == "__main__":
    # M3 verification step (planning.md AI Tool Plan): test the signal directly
    # on known-human vs known-AI style samples BEFORE wiring it into the endpoint.
    samples = {
        "human-ish (bursty, varied)": (
            "Rain. It came without warning, hammering the tin roof until the "
            "whole house seemed to shiver. I ran. My sister, who had been "
            "reading by the window all afternoon, only laughed and turned "
            "another page, unbothered by the noise or by me."
        ),
        "ai-ish (uniform, even)": (
            "The weather changed quickly during the afternoon. The rain began "
            "to fall steadily across the region. Many people decided to stay "
            "inside their homes. The situation continued for several hours "
            "without any significant change."
        ),
        "too short": "Hello there. Nice day.",
    }
    for name, txt in samples.items():
        result = analyze(txt)
        print(f"\n[{name}]")
        print(f"  p_ai = {result['p_ai']}")
        for k, v in result["evidence"].items():
            print(f"    {k}: {v}")
