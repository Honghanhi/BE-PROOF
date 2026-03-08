"""
Explainable AI Service
Generates highlighted evidence spans and narrative summaries.
"""

import asyncio
import re
from typing import Any


async def explain_prediction(text: str, trust_score: int, models: list) -> dict[str, Any]:
    """
    Generate explainability data: highlighted signals and summary text.
    In production, use SHAP or LIME to compute feature importance.
    """
    await asyncio.sleep(0.05)

    signals = _extract_signals(text, trust_score)
    summary = _generate_summary(trust_score, signals, models)

    return {"signals": signals, "summary": summary}


def _extract_signals(text: str, trust_score: int) -> list[dict]:
    """
    Extract evidence spans from text.
    In production: use attention weights or SHAP values.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    signals   = []

    ai_markers   = ["furthermore", "in conclusion", "it is worth noting",
                    "it should be noted", "in summary", "as mentioned",
                    "nevertheless", "consequently"]
    fake_markers = ["shocking", "breaking", "secret", "exposed",
                    "they don't want", "hidden truth"]

    for sentence in sentences[:10]:
        s_lower = sentence.lower()

        ai_hits   = sum(1 for m in ai_markers   if m in s_lower)
        fake_hits = sum(1 for m in fake_markers  if m in s_lower)

        if fake_hits > 0:
            signals.append({
                "text":     sentence.strip(),
                "type":     "misinformation",
                "strength": min(1.0, 0.4 + fake_hits * 0.25),
            })
        elif ai_hits > 0 or (trust_score < 50 and len(signals) < 4):
            signals.append({
                "text":     sentence.strip(),
                "type":     "ai-pattern",
                "strength": min(1.0, 0.3 + ai_hits * 0.2),
            })
        else:
            signals.append({
                "text":     sentence.strip(),
                "type":     "neutral",
                "strength": 0.1,
            })

    return signals


def _generate_summary(trust_score: int, signals: list, models: list) -> str:
    ai_count   = sum(1 for s in signals if s["type"] == "ai-pattern")
    fake_count = sum(1 for s in signals if s["type"] == "misinformation")
    n_models   = len(models)

    if trust_score >= 85:
        verdict_text = "exhibits natural linguistic variation consistent with human authorship"
    elif trust_score >= 70:
        verdict_text = "appears mostly authentic with minor stylistic anomalies"
    elif trust_score >= 50:
        verdict_text = "shows mixed authenticity signals requiring further review"
    elif trust_score >= 30:
        verdict_text = "displays multiple indicators of AI-generated or misleading content"
    else:
        verdict_text = "strongly indicates AI-generated or fabricated content"

    summary = f"This content {verdict_text} (Trust Score: {trust_score}/100). "
    if ai_count:
        summary += f"{ai_count} AI-pattern segment(s) detected. "
    if fake_count:
        summary += f"{fake_count} potential misinformation signal(s) flagged. "
    if n_models:
        agreeing = sum(
            1 for m in models
            if (trust_score >= 50 and m["score"] >= 50) or
               (trust_score < 50  and m["score"] < 50)
        )
        summary += f"{agreeing}/{n_models} models in agreement."

    return summary.strip()