"""
AI Text Detection Service
══════════════════════════════════════════════════════════════
Primary model : Hello-SimpleAI/chatgpt-detector-roberta  (ONNX Runtime)
Fallback      : statistical heuristic (zero dependencies)

Thay đổi so với bản gốc
────────────────────────
- Dùng optimum[onnxruntime] thay cho torch+transformers thuần
- RAM giảm từ ~500MB xuống ~200–280MB → chạy được trên Render free tier
- Tự động export sang ONNX lần đầu chạy (export=True)
- Mọi logic chunking, label parsing, heuristic fallback giữ nguyên

Label mapping
─────────────
  LABEL_0  →  FAKE  →  AI-generated   (low trust score)
  LABEL_1  →  REAL  →  Human-written  (high trust score)
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Any

log = logging.getLogger("aiproof.ai_text")

# ── Model config ──────────────────────────────────────────────────────────────

MODEL_ID      = "Hello-SimpleAI/chatgpt-detector-roberta"
MODEL_NAME    = "ChatGPT Detector RoBERTa (Hello-SimpleAI)"
MODEL_WEIGHT  = 0.35

MAX_TOKENS    = 512
WINDOW_TOKENS = 480
OVERLAP_TOKENS = 32

# ── Pipeline singleton ────────────────────────────────────────────────────────

_pipeline = None
_pipeline_error: str | None = None


def _load_pipeline():
    """
    Load model qua ONNX Runtime thuần — KHÔNG dùng torch.
    - Dùng ORTModelForSequenceClassification từ optimum
    - Dùng InferenceSession trực tiếp nếu optimum cũng kéo torch
    - RAM ~200–280MB, chạy tốt trên CPU Render free tier
    """
    global _pipeline, _pipeline_error

    # Chặn torch được import bởi bất kỳ sub-dependency nào
    import sys
    sys.modules.setdefault("torch", None)  # type: ignore

    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer, pipeline as hf_pipeline

        log.info("Loading %s via ONNX Runtime…", MODEL_ID)
        t0 = time.time()

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        model = ORTModelForSequenceClassification.from_pretrained(
            MODEL_ID,
            export=True,                        # convert HF → ONNX lần đầu
            provider="CPUExecutionProvider",    # CPU only, không cần GPU
        )

        _pipeline = hf_pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            top_k=None,
            truncation=True,
            max_length=MAX_TOKENS,
        )

        log.info("%s loaded via ONNX in %.1fs", MODEL_ID, time.time() - t0)

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
    Xử lý tất cả label format: LABEL_0/1, Fake/Real, AI/Human, generated/original
    """
    mapping: dict[str, float] = {}
    for item in raw:
        lbl = item["label"].upper()
        if lbl in ("LABEL_0", "FAKE", "AI", "GENERATED", "MACHINE"):
            mapping["ai"]    = item["score"]
        elif lbl in ("LABEL_1", "REAL", "HUMAN", "ORIGINAL"):
            mapping["human"] = item["score"]

    if "ai" in mapping and "human" not in mapping:
        mapping["human"] = 1.0 - mapping["ai"]
    elif "human" in mapping and "ai" not in mapping:
        mapping["ai"] = 1.0 - mapping["human"]
    elif not mapping:
        mapping = {"ai": 0.5, "human": 0.5}

    ai_pct    = round(mapping["ai"]    * 100, 2)
    human_pct = round(mapping["human"] * 100, 2)
    return ai_pct, human_pct


# ── Chunked inference ─────────────────────────────────────────────────────────

def _chunk_text(text: str, tokenizer) -> list[str]:
    """Chia text thành các cửa sổ token chồng lấp để xử lý văn bản dài."""
    ids    = tokenizer.encode(text, add_special_tokens=False)
    stride = WINDOW_TOKENS - OVERLAP_TOKENS
    chunks = []
    start  = 0
    while start < len(ids):
        window = ids[start : start + WINDOW_TOKENS]
        chunks.append(tokenizer.decode(window, skip_special_tokens=True))
        start += stride
        if start + WINDOW_TOKENS >= len(ids):
            tail = ids[start:]
            if tail:
                chunks.append(tokenizer.decode(tail, skip_special_tokens=True))
            break
    return chunks or [text]


def _run_inference(text: str) -> tuple[float, float, float]:
    """
    Blocking inference — được gọi qua asyncio.to_thread().
    Trả về (ai_pct, human_pct, confidence).
    """
    pipe      = _pipeline
    tokenizer = pipe.tokenizer

    approx_tokens = len(text.split()) * 1.3

    if approx_tokens <= WINDOW_TOKENS:
        raw        = pipe(text, truncation=True, max_length=MAX_TOKENS)[0]
        chunks_raw = [raw if isinstance(raw, list) else [raw]]
        chunk_texts = [text]
    else:
        chunk_texts = _chunk_text(text, tokenizer)
        chunks_raw  = [
            (r if isinstance(r, list) else [r])
            for r in pipe(chunk_texts, truncation=True, max_length=MAX_TOKENS)
        ]

    total_weight = 0.0
    ai_sum       = 0.0
    human_sum    = 0.0
    conf_sum     = 0.0

    for i, label_list in enumerate(chunks_raw):
        w          = len(chunk_texts[i].split()) if i < len(chunk_texts) else 1
        ai_p, hu_p = _parse_labels(label_list)
        conf       = max(ai_p, hu_p) / 100.0
        ai_sum    += ai_p * w
        human_sum += hu_p * w
        conf_sum  += conf * w
        total_weight += w

    if total_weight == 0:
        return 50.0, 50.0, 0.5

    ai_pct     = round(ai_sum    / total_weight, 2)
    human_pct  = round(human_sum / total_weight, 2)
    confidence = round(conf_sum  / total_weight, 4)

    return ai_pct, human_pct, confidence


# ── Heuristic fallback ────────────────────────────────────────────────────────

def _heuristic_fallback(text: str) -> tuple[float, float, float]:
    """
    Fallback thuần Python khi ONNX pipeline không load được.
    Confidence luôn thấp (≤ 0.55) để báo hiệu chế độ fallback.
    """
    import re as _re

    words = text.split()
    if not words:
        return 50.0, 50.0, 0.0

    unique = len(set(w.lower().strip(".,!?;:\"'") for w in words))
    ttr    = unique / len(words)

    sentences = [s.strip() for s in _re.split(r"[.!?]+", text) if s.strip()]
    if len(sentences) >= 2:
        lengths  = [len(s.split()) for s in sentences]
        mean_l   = sum(lengths) / len(lengths)
        variance = sum((l - mean_l) ** 2 for l in lengths) / len(lengths)
        std_dev  = math.sqrt(variance)
    else:
        std_dev = 5.0

    AI_MARKERS = [
        "furthermore", "moreover", "additionally", "in conclusion",
        "it is worth noting", "it is important to note", "in summary",
        "as mentioned earlier", "delve", "underscore", "pivotal",
        "it should be noted", "with that said", "that being said",
    ]
    text_lower  = text.lower()
    marker_hits = sum(1 for m in AI_MARKERS if m in text_lower)

    ttr_signal    = max(0, (0.72 - ttr) * 60)
    var_signal    = max(0, (8.0 - std_dev) * 2.5)
    marker_signal = min(30, marker_hits * 7)

    ai_raw    = ttr_signal + var_signal + marker_signal
    ai_pct    = round(min(92, max(8, ai_raw)), 2)
    human_pct = round(100 - ai_pct, 2)

    wc_conf    = min(1.0, len(words) / 250)
    confidence = round(0.35 + wc_conf * 0.20, 4)

    return ai_pct, human_pct, confidence


# ── Public entry point ────────────────────────────────────────────────────────

async def analyze_text(text: str) -> dict[str, Any]:
    """
    Phân tích text để phát hiện AI-generated content.

    Trả về:
        {
          "models": [
            {
              "modelId":       str,
              "modelName":     str,
              "ai_percent":    float,   # 0–100
              "human_percent": float,   # 0–100
              "confidence":    float,   # 0–1
              "score":         float,   # = human_percent
              "latencyMs":     int,
            }
          ]
        }
    """
    t0 = time.time()

    if _pipeline is not None:
        try:
            ai_pct, human_pct, confidence = await asyncio.to_thread(
                _run_inference, text
            )
            source = "onnx-model"
        except Exception as exc:
            log.warning("Inference error (%s) — falling back to heuristic", exc)
            ai_pct, human_pct, confidence = _heuristic_fallback(text)
            source = "heuristic-fallback"
    else:
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
                "score":         human_pct,
                "latencyMs":     latency_ms,
                "source":        source,
            }
        ]
    }


# ══════════════════════════════════════════════════════════
#  FastAPI wrapper  —  text-service
#  Exposes:  GET /health   POST /detect
# ══════════════════════════════════════════════════════════

import os
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        "text-service ready  model=%s  pipeline=%s  runtime=ONNX",
        MODEL_ID,
        "loaded" if _pipeline else "heuristic-only",
    )
    yield


app = FastAPI(title="Text Detection Service", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextRequest(BaseModel):
    content: str

    @field_validator("content")
    @classmethod
    def _check(cls, v: str) -> str:
        if not v or len(v.strip()) < 10:
            raise ValueError("content must be at least 10 characters")
        return v.strip()


@app.get("/health")
async def health():
    return {
        "status":   "ok",
        "service":  "text-detection",
        "model":    MODEL_ID,
        "runtime":  "onnx" if _pipeline else "heuristic",
        "pipeline": _pipeline is not None,
    }


@app.post("/detect")
async def detect(req: TextRequest):
    result = await analyze_text(req.content)
    model  = result["models"][0]
    return {
        "modelId":       model["modelId"],
        "modelName":     model["modelName"],
        "ai_percent":    model["ai_percent"],
        "human_percent": model["human_percent"],
        "confidence":    model["confidence"],
        "score":         model["score"],
        "latencyMs":     model["latencyMs"],
        "source":        model["source"],
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8001)),
        reload=False,
    )
