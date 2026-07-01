"""Detection pipeline — runs both signals and fuses them into one verdict.

Fusion + confidence (planning.md > Uncertainty representation), verbatim:

    p_ai        = W_STRUCTURAL * p_ai_A + W_LLM * p_ai_B
    disagreement = |p_ai_A - p_ai_B|
    agreement   = 1 - disagreement
    dist        = |p_ai - 0.5| / 0.5          # 0 at coin-flip, 1 at extremes
    confidence  = dist * agreement            # both must be high to be confident

Verdict (planning.md thresholds, applied IN ORDER):

    1. disagreement > DISAGREE_THRESHOLD  -> uncertain   (signals disagree)
    2. confidence   < CONFIDENCE_FLOOR    -> uncertain   (not sure enough)
    3. p_ai >= AI_THRESHOLD               -> likely_ai
    4. p_ai <= HUMAN_THRESHOLD            -> likely_human
    5. otherwise (middle band)            -> uncertain
"""

from __future__ import annotations

from config import (
    AI_THRESHOLD,
    CONFIDENCE_FLOOR,
    DISAGREE_THRESHOLD,
    HUMAN_THRESHOLD,
    W_LLM,
    W_STRUCTURAL,
)
from signals import llm_judge, structural


def fuse(p_ai_a: float, p_ai_b: float) -> dict:
    """Combine two per-signal p_ai values into a fused score, confidence, verdict.

    Pure function of the two inputs — deterministic, so it can be unit-tested
    against the planning thresholds without calling any signal.
    """
    p_ai = W_STRUCTURAL * p_ai_a + W_LLM * p_ai_b
    disagreement = abs(p_ai_a - p_ai_b)
    agreement = 1.0 - disagreement
    dist = abs(p_ai - 0.5) / 0.5
    confidence = dist * agreement

    if disagreement > DISAGREE_THRESHOLD:
        verdict = "uncertain"
    elif confidence < CONFIDENCE_FLOOR:
        verdict = "uncertain"
    elif p_ai >= AI_THRESHOLD:
        verdict = "likely_ai"
    elif p_ai <= HUMAN_THRESHOLD:
        verdict = "likely_human"
    else:
        verdict = "uncertain"

    return {
        "attribution": verdict,
        "p_ai": round(p_ai, 4),
        "confidence": round(confidence, 4),
        "agreement": round(agreement, 4),
    }


def analyze(text: str) -> dict:
    """Run both signals on `text` and return the fused result + per-signal breakdown."""
    signal_a = structural.analyze(text)
    signal_b = llm_judge.analyze(text)
    fused = fuse(signal_a["p_ai"], signal_b["p_ai"])
    return {
        **fused,
        "breakdown": {
            "structural": signal_a,
            "llm_judge": signal_b,
        },
    }
