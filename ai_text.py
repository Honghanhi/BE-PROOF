"""
AI Text Detection Service
══════════════════════════════════════════════════════════════
Primary model : Hello-SimpleAI/chatgpt-detector-roberta  (HuggingFace)
Fallback      : statistical heuristic (zero dependencies)

Label mapping
─────────────
  LABEL_0  →  FAKE  →  AI-generated   (low trust score)
  LABEL_1  →  REAL  →  Human-written  (high trust score)

The pipeline returns raw logits → softmax → probabilities.
We read the REAL (human) probability as the trust score and
AI probability as 1 − trust.

Chunking
────────
RoBERTa has a 512-token hard limit.  Long texts are split into
overlapping 480-token windows (32-token overlap) and results are
averaged with a length-weighted mean so every part of the text
contributes proportionally.

Model cache
───────────
The pipeline is loaded once at module import and held in
_PIPELINE.  Concurrent requests share the same loaded model;
asyncio.to_thread() keeps the blocking inference off the event
loop so the server stays responsive.

Exported surface
────────────────
  analyze_text(text)  →  { models: [ModelResult] }

ModelResult:
  modelId      str
  modelName    str
  ai_percent   float   0–100
  human_percent float  0–100
  confidence   float   0–1
  score        float   0–100   (= human_percent, trust-score convention)
  latencyMs    int
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Any

log = logging.getLogger("aiproof.ai_text")

# ── Model config ──────────────────────────────────────────────────────────────

MODEL_ID   = "Hello-SimpleAI/chatgpt-detector-roberta"
MODEL_NAME = "ChatGPT Detector RoBERTa (Hello-SimpleAI)"
MODEL_WEIGHT = 0.35          # consensus weight (higher = more trusted)

MAX_TOKENS   = 512
WINDOW_TOKENS = 480          # tokens per chunk (< MAX_TOKENS)
OVERLAP_TOKENS = 32          # overlap between consecutive chunks

# ── Pipeline singleton ────────────────────────────────────────────────────────

_pipeline = None             # transformers TextClassificationPipeline
_pipeline_error: str | None = None   # set if load failed

def _load_pipeline():
    """
    Load the HuggingFace pipeline once.  Called at import time in a
    try/except so a missing model or missing torch does not crash the server.
    """
    global _pipeline, _pipeline_error
    try:
        from transformers import pipeline as hf_pipeline
        log.info("Loading %s …", MODEL_ID)
        t0 = time.time()
        _pipeline = hf_pipeline(
            "text-classification",
            model=MODEL_ID,
            top_k=None,            # return both LABEL_0 and LABEL_1
            truncation=True,
            max_length=MAX_TOKENS,
        )
        log.info("%s loaded in %.1fs", MODEL_ID, time.time() - t0)
    except Exception as exc:
        _pipeline_error = str(exc)
        log.warning(
            "Could not load %s (%s) — heuristic fallback active",
            MODEL_ID, exc,
        )


_load_pipeline()


# ── Label normalisation ───────────────────────────────────────────────────────

def _parse_labels(raw: list[dict]) -> tuple[float, float]:
    """
    Map raw pipeline output → (ai_pct, human_pct).

    Handles all known label formats:
      LABEL_0 / LABEL_1
      Fake / Real
      AI / Human
      generated / original
    """
    mapping: dict[str, str] = {}
    for item in raw:
        lbl = item["label"].upper()
        if lbl in ("LABEL_0", "FAKE", "AI", "GENERATED", "MACHINE"):
            mapping["ai"]    = item["score"]
        elif lbl in ("LABEL_1", "REAL", "HUMAN", "ORIGINAL"):
            mapping["human"] = item["score"]

    # If only one label came back, derive the other
    if "ai" in mapping and "human" not in mapping:
        mapping["human"] = 1.0 - mapping["ai"]
    elif "human" in mapping and "ai" not in mapping:
        mapping["ai"] = 1.0 - mapping["human"]
    elif not mapping:
        # Totally unknown label — treat as uncertain
        mapping = {"ai": 0.5, "human": 0.5}

    ai_pct    = round(mapping["ai"]    * 100, 2)
    human_pct = round(mapping["human"] * 100, 2)
    return ai_pct, human_pct


# ── Chunked inference ─────────────────────────────────────────────────────────

def _chunk_text(text: str, tokenizer) -> list[str]:
    """
    Split text into overlapping token windows.
    Returns a list of decoded strings ready for the pipeline.
    """
    ids    = tokenizer.encode(text, add_special_tokens=False)
    stride = WINDOW_TOKENS - OVERLAP_TOKENS
    chunks = []
    start  = 0
    while start < len(ids):
        window = ids[start : start + WINDOW_TOKENS]
        chunks.append(tokenizer.decode(window, skip_special_tokens=True))
        start += stride
        if start + WINDOW_TOKENS >= len(ids):
            # Add the final tail and stop
            tail = ids[start:]
            if tail:
                chunks.append(tokenizer.decode(tail, skip_special_tokens=True))
            break
    return chunks or [text]


def _run_inference(text: str) -> tuple[float, float, float]:
    """
    Blocking inference — run via asyncio.to_thread().
    Returns (ai_pct, human_pct, confidence).
    """
    pipe = _pipeline
    tokenizer = pipe.tokenizer

    # Rough token estimate to decide whether chunking is needed
    approx_tokens = len(text.split()) * 1.3
    if approx_tokens <= WINDOW_TOKENS:
        raw    = pipe(text, truncation=True, max_length=MAX_TOKENS)[0]
        chunks_raw = [raw if isinstance(raw, list) else [raw]]
    else:
        chunks     = _chunk_text(text, tokenizer)
        chunks_raw = [
            (r if isinstance(r, list) else [r])
            for r in pipe(chunks, truncation=True, max_length=MAX_TOKENS)
        ]

    # Weighted average — longer chunks carry more weight
    total_weight = 0.0
    ai_sum       = 0.0
    human_sum    = 0.0
    conf_sum     = 0.0

    chunk_texts = _chunk_text(text, tokenizer) if approx_tokens > WINDOW_TOKENS else [text]

    for i, label_list in enumerate(chunks_raw):
        w         = len(chunk_texts[i].split()) if i < len(chunk_texts) else 1
        ai_p, hu_p = _parse_labels(label_list)
        conf      = max(ai_p, hu_p) / 100.0
        ai_sum    += ai_p * w
        human_sum += hu_p * w
        conf_sum  += conf * w
        total_weight += w

    if total_weight == 0:
        return 50.0, 50.0, 0.5

    ai_pct    = round(ai_sum    / total_weight, 2)
    human_pct = round(human_sum / total_weight, 2)
    confidence = round(conf_sum / total_weight, 4)

    return ai_pct, human_pct, confidence


# ── Heuristic fallback ────────────────────────────────────────────────────────

def _heuristic_fallback(text: str) -> tuple[float, float, float]:
    """
    Pure-Python statistical fallback when the transformer is unavailable.
    Uses type-token ratio, sentence-length variance, and AI-marker density.

    Returns (ai_pct, human_pct, confidence).
    Confidence is always low (≤ 0.55) to signal fallback mode.
    """
    import re as _re

    words   = text.split()
    if not words:
        return 50.0, 50.0, 0.0

    # Type-token ratio (low TTR → more repetitive → more AI-like)
    unique = len(set(w.lower().strip(".,!?;:\"'") for w in words))
    ttr    = unique / len(words)

    # Sentence length variance (AI tends to produce uniform sentence lengths)
    sentences = [s.strip() for s in _re.split(r"[.!?]+", text) if s.strip()]
    if len(sentences) >= 2:
        lengths  = [len(s.split()) for s in sentences]
        mean_l   = sum(lengths) / len(lengths)
        variance = sum((l - mean_l) ** 2 for l in lengths) / len(lengths)
        std_dev  = math.sqrt(variance)
    else:
        std_dev = 5.0   # neutral default

    # AI marker words
    AI_MARKERS = [
        "furthermore", "moreover", "additionally", "in conclusion",
        "it is worth noting", "it is important to note", "in summary",
        "as mentioned earlier", "delve", "underscore", "pivotal",
        "it should be noted", "with that said", "that being said",
    ]
    text_lower = text.lower()
    marker_hits = sum(1 for m in AI_MARKERS if m in text_lower)

    # Combine signals → ai_pct
    # Low TTR, low variance, many markers → higher AI probability
    ttr_signal      = max(0, (0.72 - ttr) * 60)          # 0–43
    var_signal      = max(0, (8.0 - std_dev) * 2.5)      # 0–20
    marker_signal   = min(30, marker_hits * 7)             # 0–30

    ai_raw   = ttr_signal + var_signal + marker_signal     # 0–93
    ai_pct   = round(min(92, max(8, ai_raw)), 2)
    human_pct = round(100 - ai_pct, 2)

    # Scale word-count → confidence (more text → more certain)
    wc_conf    = min(1.0, len(words) / 250)
    confidence = round(0.35 + wc_conf * 0.20, 4)          # 0.35–0.55

    return ai_pct, human_pct, confidence


# ── Public entry point ────────────────────────────────────────────────────────

async def analyze_text(text: str) -> dict[str, Any]:
    """
    Analyze text for AI-generation signals.

    Returns:
        {
          "models": [
            {
              "modelId":      "roberta-base-openai-detector",
              "modelName":    "ChatGPT Detector RoBERTa (Hello-SimpleAI)",
              "ai_percent":   float,   # 0–100
              "human_percent":float,   # 0–100
              "confidence":   float,   # 0–1
              "score":        float,   # = human_percent (trust-score convention)
              "latencyMs":    int,
            }
          ]
        }
    """
    t0 = time.time()

    if _pipeline is not None:
        # Real inference — offload blocking call to thread pool
        try:
            ai_pct, human_pct, confidence = await asyncio.to_thread(
                _run_inference, text
            )
            source = "model"
        except Exception as exc:
            log.warning("Inference error (%s) — falling back to heuristic", exc)
            ai_pct, human_pct, confidence = _heuristic_fallback(text)
            source = "heuristic-fallback"
    else:
        # Model never loaded — use heuristic
        ai_pct, human_pct, confidence = _heuristic_fallback(text)
        source = "heuristic"

    latency_ms = round((time.time() - t0) * 1000)

    return {
        "models": [
            {
                "modelId":       MODEL_ID,
                "modelName":     MODEL_NAME,
                "weight":        MODEL_WEIGHT,
                "ai_percent":    ai_pct,
                "human_percent": human_pct,
                "confidence":    confidence,
                "score":         human_pct,   # trust-score convention: 100 = human
                "latencyMs":     latency_ms,
                "source":        source,
            }
        ]
    }