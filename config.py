"""Tunable thresholds, weights, and model id.

Single source of truth for the scoring spec defined in
planning.md > Uncertainty representation. Change values here, not inline in the
signal or pipeline code.
"""

# --- Fusion weights (planning.md: default 0.5 / 0.5) --------------------------
W_STRUCTURAL = 0.5  # weight of Signal A (structural)
W_LLM = 0.5         # weight of Signal B (LLM judge)

# --- Verdict thresholds (planning.md, applied in order; see pipeline.fuse) -----
DISAGREE_THRESHOLD = 0.4  # |p_ai_A - p_ai_B| strictly above this -> uncertain
CONFIDENCE_FLOOR = 0.5    # confidence below this -> uncertain
AI_THRESHOLD = 0.65       # fused p_ai >= this -> likely_ai
HUMAN_THRESHOLD = 0.35    # fused p_ai <= this -> likely_human

# --- Signal B (Groq) ----------------------------------------------------------
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_TEMPERATURE = 0.0  # low temperature to reduce (not eliminate) non-determinism
