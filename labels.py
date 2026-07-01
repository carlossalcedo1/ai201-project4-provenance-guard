"""Transparency label builder — maps a verdict + confidence to reader-facing text.

The three variants are the verbatim strings from planning.md > Transparency label
variants (and the README, which is the canonical record). `{pct}` is filled with
the confidence as a whole-number percentage.

The pipeline's verdict (`likely_ai` / `likely_human` / `uncertain`) already
encodes the planning thresholds (p_ai bands, agreement, confidence floor), so the
label builder simply maps the verdict to its variant — the two never diverge.
"""

from __future__ import annotations

LABEL_TEMPLATES = {
    "likely_ai": (
        "🤖 Likely AI-Generated — Our analysis indicates this content was most "
        "likely produced by an AI system (confidence: {pct}%). Two independent "
        "signals — writing-structure analysis and a language-model review — agreed "
        "on this assessment. Attribution is an estimate, not proof. If you created "
        "this yourself, you can appeal this label."
    ),
    "likely_human": (
        "✍️ Likely Human-Written — Our analysis indicates this content was most "
        "likely written by a person (confidence: {pct}%). Two independent signals "
        "agreed on this assessment. Attribution is an estimate, not a certainty — "
        "if you disagree, you can appeal this label."
    ),
    "uncertain": (
        "❓ Uncertain Attribution — We could not confidently determine whether this "
        "content was written by a human or by an AI (confidence: {pct}%). Our two "
        "signals were inconclusive or disagreed with each other. Please treat this "
        "result with caution — it should not be used as proof of authorship. You "
        "can appeal this label."
    ),
}


def build_label(attribution: str, confidence: float) -> str:
    """Return the transparency label text for a verdict + confidence.

    `attribution` is one of "likely_ai", "likely_human", "uncertain". Any unknown
    verdict falls back to the uncertain label (fail safe, never over-claim).
    """
    pct = round(confidence * 100)
    template = LABEL_TEMPLATES.get(attribution, LABEL_TEMPLATES["uncertain"])
    return template.format(pct=pct)
