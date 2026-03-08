"""
Fake News Detection Service
══════════════════════════════════════════════════════════════
Primary model : cross-encoder/nli-MiniLM2-L6-H768  (zero-shot classification)
Fallback      : sensationalism + credibility heuristic

Zero-shot approach
──────────────────
BART-MNLI is framed as a Natural Language Inference (NLI) task.
We pass the article as the premise and test it against three
hypothesis pairs:

  Credibility axis   "This article contains factual, verified information"
                     "This article contains misinformation or false claims"

  Sensationalism     "This article uses balanced, neutral language"
                     "This article uses sensationalist, misleading language"

  Source quality     "This article cites credible sources and evidence"
                     "This article makes unsubstantiated claims"

Each hypothesis pair produces an entailment probability.
We average the three "fake" probabilities → fake_percent,
and derive real_percent = 100 − fake_percent.

URL fetching
────────────
fetch_url(url) is a standalone async helper so the /detect-fake-news
endpoint can accept either raw text OR a URL in one call.
HTML is stripped with a lightweight regex cleaner + <script>/<style>
removal.  Text is capped at 4 000 chars (≈ 700 tokens) before
inference to stay within BART's 1 024-token limit.

Model cache
───────────
Pipeline loaded once at import, held in _pipeline singleton.
Inference offloaded to asyncio.to_thread() — non-blocking.

Exported surface
────────────────
  detect_fake_news(text)          →  { models: [ModelResult] }
  fetch_url(url, timeout?)        →  str  (plain text)

ModelResult:
  modelId        str
  modelName      str
  fake_percent   float  0–100
  real_percent   float  0–100
  confidence     float  0–1
  score          float  0–100   (= real_percent, trust-score convention)
  signals        list[Signal]
  latencyMs      int
  source         str
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

log = logging.getLogger("aiproof.fake_news")

# ── Model config ──────────────────────────────────────────────────────────────

MODEL_ID    = "cross-encoder/nli-MiniLM2-L6-H768"
MODEL_NAME  = "NLI-MiniLM2 L6 (Cross-Encoder)"
MODEL_WEIGHT = 0.25
MAX_CHARS   = 4_000      # chars to pass to inference (~700 tokens for BART)

# ── Zero-shot hypothesis pairs ────────────────────────────────────────────────
# Each tuple: (real_hypothesis, fake_hypothesis)
# We average fake-entailment across all pairs.

_HYPOTHESES: list[tuple[str, str]] = [
    (
        "This article contains factual, verified information.",
        "This article contains misinformation or false claims.",
    ),
    (
        "This article uses balanced, neutral, objective language.",
        "This article uses sensationalist, exaggerated, or misleading language.",
    ),
    (
        "This article is supported by credible sources and evidence.",
        "This article makes unsubstantiated or unverifiable claims.",
    ),
]

# ── Sensationalist signals (used both for heuristic and signal extraction) ────

_SENSATIONAL_PHRASES = [
    "breaking", "shocking", "bombshell", "explosive", "you won't believe",
    "they don't want you to know", "hidden truth", "mainstream media lies",
    "big pharma", "deep state", "exposed", "wake up", "sheeple",
    "plandemic", "false flag", "hoax", "cover-up", "banned",
    "doctors don't want", "miracle cure", "secret remedy",
]

_CREDIBILITY_PHRASES = [
    "according to", "researchers found", "study published",
    "peer-reviewed", "official statement", "confirmed by",
    "data shows", "report says", "spokesperson said",
]

# ── Pipeline singleton ────────────────────────────────────────────────────────

_pipeline = None
_pipeline_error: str | None = None


def _load_pipeline() -> None:
    global _pipeline, _pipeline_error
    try:
        from transformers import pipeline as hf_pipeline
        log.info("Loading %s …", MODEL_ID)
        t0 = time.time()
        _pipeline = hf_pipeline(
            "zero-shot-classification",
            model=MODEL_ID,
            device=-1,          # CPU; set to 0 for CUDA
        )
        log.info("%s loaded in %.1fs", MODEL_ID, time.time() - t0)
    except Exception as exc:
        _pipeline_error = str(exc)
        log.warning(
            "Could not load %s (%s) — heuristic fallback active",
            MODEL_ID, exc,
        )


_load_pipeline()


# ── URL fetcher ───────────────────────────────────────────────────────────────

async def fetch_url(url: str, timeout: float = 15.0) -> str:
    """
    Fetch a URL and return cleaned plain text.
    Raises ValueError on network error or empty content.
    """
    try:
        import httpx
    except ImportError:
        raise ValueError("httpx is not installed — cannot fetch URLs")

    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "AI-PROOF/2.1 fake-news-detector"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text
    except Exception as exc:
        raise ValueError(f"Could not fetch URL: {exc}") from exc

    return _clean_html(html)


def _clean_html(html: str) -> str:
    """Remove scripts, styles, tags; collapse whitespace."""
    # Drop <script> and <style> blocks entirely
    html = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", " ", html,
                  flags=re.DOTALL | re.IGNORECASE)
    # Strip remaining tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── BART zero-shot inference ──────────────────────────────────────────────────

def _run_zero_shot(text: str) -> tuple[float, float, float, list[dict]]:
    """
    Blocking call — run via asyncio.to_thread().
    Returns (fake_pct, real_pct, confidence, signals).
    """
    pipe    = _pipeline
    snippet = text[:MAX_CHARS]

    fake_scores: list[float] = []
    real_scores: list[float] = []

    for real_hyp, fake_hyp in _HYPOTHESES:
        out = pipe(
            snippet,
            candidate_labels=[real_hyp, fake_hyp],
            hypothesis_template="{}",
            multi_label=False,
        )
        # Map back: which label corresponds to fake?
        scores_by_label = dict(zip(out["labels"], out["scores"]))
        fake_scores.append(scores_by_label.get(fake_hyp, 0.5))
        real_scores.append(scores_by_label.get(real_hyp, 0.5))

    fake_pct = round(sum(fake_scores) / len(fake_scores) * 100, 2)
    real_pct = round(100 - fake_pct, 2)

    # Confidence = how decisive the average verdict is
    # (distance from 50% decision boundary, scaled to 0–1)
    avg_fake = fake_pct / 100
    confidence = round(abs(avg_fake - 0.5) * 2, 4)  # 0 at 50%, 1 at 0% or 100%

    signals = _extract_signals(text, fake_pct)
    return fake_pct, real_pct, confidence, signals


# ── Heuristic fallback ────────────────────────────────────────────────────────

def _heuristic_fallback(text: str) -> tuple[float, float, float, list[dict]]:
    """
    Keyword-density fallback when BART is unavailable.
    Returns (fake_pct, real_pct, confidence, signals).
    Confidence capped at 0.50 to signal fallback mode.
    """
    lower = text.lower()

    sensational_hits = sum(1 for p in _SENSATIONAL_PHRASES if p in lower)
    credible_hits    = sum(1 for p in _CREDIBILITY_PHRASES  if p in lower)

    # Base: sensational pulls toward fake, credible pulls toward real
    fake_raw = 35 + sensational_hits * 9 - credible_hits * 6
    fake_pct = round(min(90, max(10, fake_raw)), 2)
    real_pct = round(100 - fake_pct, 2)

    word_count = len(text.split())
    confidence = round(min(0.50, 0.25 + word_count / 1000 * 0.25), 4)

    signals = _extract_signals(text, fake_pct)
    return fake_pct, real_pct, confidence, signals


# ── Signal extraction ─────────────────────────────────────────────────────────

def _extract_signals(text: str, fake_pct: float) -> list[dict]:
    """
    Build human-readable signal list from keyword matches.
    Each signal: { type, phrase, severity }
    """
    lower   = text.lower()
    signals = []

    for phrase in _SENSATIONAL_PHRASES:
        if phrase in lower:
            signals.append({
                "type":     "sensationalism",
                "phrase":   phrase,
                "severity": "high" if fake_pct > 60 else "medium",
            })

    for phrase in _CREDIBILITY_PHRASES:
        if phrase in lower:
            signals.append({
                "type":     "credibility_marker",
                "phrase":   phrase,
                "severity": "positive",
            })

    return signals[:12]   # cap at 12 for payload size


# ── Public entry point ────────────────────────────────────────────────────────

async def detect_fake_news(text: str) -> dict[str, Any]:
    """
    Detect fake-news / misinformation signals in text.

    Returns:
        {
          "models": [
            {
              "modelId":      "facebook/bart-large-mnli",
              "modelName":    "NLI-MiniLM2 L6 (Cross-Encoder)",
              "fake_percent": float,   # 0–100
              "real_percent": float,   # 0–100
              "confidence":   float,   # 0–1
              "score":        float,   # = real_percent  (trust-score convention)
              "signals":      list,
              "latencyMs":    int,
              "source":       str,
            }
          ]
        }
    """
    t0 = time.time()

    if _pipeline is not None:
        try:
            fake_pct, real_pct, confidence, signals = await asyncio.to_thread(
                _run_zero_shot, text
            )
            source = "model"
        except Exception as exc:
            log.warning("Zero-shot inference error (%s) — falling back", exc)
            fake_pct, real_pct, confidence, signals = _heuristic_fallback(text)
            source = "heuristic-fallback"
    else:
        fake_pct, real_pct, confidence, signals = _heuristic_fallback(text)
        source = "heuristic"

    latency_ms = round((time.time() - t0) * 1000)

    return {
        "models": [
            {
                "modelId":      MODEL_ID,
                "modelName":    MODEL_NAME,
                "weight":       MODEL_WEIGHT,
                "fake_percent": fake_pct,
                "real_percent": real_pct,
                "confidence":   confidence,
                "score":        real_pct,   # trust-score: 100 = real/credible
                "signals":      signals,
                "latencyMs":    latency_ms,
                "source":       source,
            }
        ]
    }