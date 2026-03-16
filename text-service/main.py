"""
AI Text Detection Service
══════════════════════════════════════════════════════════════
Runtime       : ONNX Runtime thuần CPU  (KHÔNG có torch)
Primary model : Hello-SimpleAI/chatgpt-detector-roberta
Fallback      : statistical heuristic

Dependencies (nhẹ, không kéo torch):
  onnxruntime      ~10 MB
  tokenizers       ~4  MB  (pure Rust)
  huggingface-hub  ~1  MB
  numpy            ~20 MB
  fastapi + uvicorn

Label mapping
─────────────
  LABEL_0 / Fake → AI-generated   (low trust)
  LABEL_1 / Real → Human-written  (high trust)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from typing import Any

import numpy as np

log = logging.getLogger("aiproof.ai_text")

# ── Model config ──────────────────────────────────────────────────────────────

MODEL_ID       = "Hello-SimpleAI/chatgpt-detector-roberta"
MODEL_NAME     = "ChatGPT Detector RoBERTa (Hello-SimpleAI)"
MODEL_WEIGHT   = 0.35

MAX_TOKENS     = 512
WINDOW_TOKENS  = 480
OVERLAP_TOKENS = 32

# ── Globals ───────────────────────────────────────────────────────────────────

_pipeline      = None        # True khi ORT session sẵn sàng
_pipeline_error: str | None = None

_ORT_SESSION   = None        # onnxruntime.InferenceSession
_TOKENIZER     = None        # tokenizers.Tokenizer
_ID2LABEL: dict[int, str] = {}

_ONNX_CANDIDATES = ["model.onnx", "onnx/model.onnx", "pytorch_model.onnx"]


# ── Load pipeline ─────────────────────────────────────────────────────────────

def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _load_pipeline():
    """
    Load model dùng ONNX Runtime + tokenizers (Rust).
    Hoàn toàn KHÔNG import torch / transformers / optimum ở runtime.
    RAM ~200–280 MB thay vì ~500 MB.
    """
    global _pipeline, _pipeline_error, _ORT_SESSION, _TOKENIZER, _ID2LABEL

    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer
        from huggingface_hub import hf_hub_download

        log.info("Downloading tokenizer + ONNX model from %s …", MODEL_ID)
        t0 = time.time()

        # 1. Tokenizer (pure Rust, không cần torch)
        tok_path   = hf_hub_download(MODEL_ID, "tokenizer.json")
        _TOKENIZER = Tokenizer.from_file(tok_path)
        _TOKENIZER.enable_truncation(max_length=MAX_TOKENS)
        _TOKENIZER.enable_padding()

        # 2. id2label từ config.json
        cfg_path = hf_hub_download(MODEL_ID, "config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        _ID2LABEL = {
            int(k): v
            for k, v in cfg.get("id2label", {"0": "LABEL_0", "1": "LABEL_1"}).items()
        }

        # 3. Tìm file .onnx
        onnx_path = None
        for candidate in _ONNX_CANDIDATES:
            try:
                onnx_path = hf_hub_download(MODEL_ID, candidate)
                log.info("Found ONNX: %s", candidate)
                break
            except Exception:
                continue

        # 4. Nếu không có .onnx → convert bằng optimum-cli (subprocess, không import torch)
        if onnx_path is None:
            log.info("No .onnx in repo — exporting with optimum-cli …")
            import subprocess, tempfile
            from pathlib import Path
            out_dir = tempfile.mkdtemp(prefix="onnx_export_")
            proc = subprocess.run(
                [
                    "optimum-cli", "export", "onnx",
                    "--model", MODEL_ID,
                    "--task",  "text-classification",
                    out_dir,
                ],
                capture_output=True, text=True, timeout=300,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"optimum-cli export failed:\n{proc.stderr}")
            onnx_path = str(Path(out_dir) / "model.onnx")
            log.info("ONNX export → %s", onnx_path)

        # 5. Tạo InferenceSession
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2    # giới hạn CPU cho Render free tier
        _ORT_SESSION = ort.InferenceSession(
            onnx_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        _pipeline = True   # flag "đã sẵn sàng"
        log.info("%s loaded via pure ONNX in %.1fs", MODEL_ID, time.time() - t0)

    except Exception as exc:
        _pipeline_error = str(exc)
        log.warning("Load failed (%s) — heuristic fallback active", exc)


_load_pipeline()


# ── ONNX predict ──────────────────────────────────────────────────────────────

def _ort_predict(texts: list[str]) -> list[list[dict]]:
    """Batch inference qua ORT. Trả về list[list[{label, score}]]."""
    enc            = _TOKENIZER.encode_batch(texts)
    input_ids      = np.array([e.ids for e in enc],            dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in enc], dtype=np.int64)

    feeds: dict[str, np.ndarray] = {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
    }
    input_names = [inp.name for inp in _ORT_SESSION.get_inputs()]
    if "token_type_ids" in input_names:
        feeds["token_type_ids"] = np.zeros_like(input_ids)

    logits = _ORT_SESSION.run(None, feeds)[0]   # (batch, num_labels)
    probs  = _softmax(logits)

    return [
        [{"label": _ID2LABEL.get(i, f"LABEL_{i}"), "score": float(row[i])}
         for i in range(len(row))]
        for row in probs
    ]


# ── Label normalisation ───────────────────────────────────────────────────────

def _parse_labels(raw: list[dict]) -> tuple[float, float]:
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

    return round(mapping["ai"] * 100, 2), round(mapping["human"] * 100, 2)


# ── Chunked inference ─────────────────────────────────────────────────────────

def _run_inference(text: str) -> tuple[float, float, float]:
    """
    Blocking inference qua ONNX Runtime (không có torch).
    Trả về (ai_pct, human_pct, confidence).
    """
    approx_tokens = len(text.split()) * 1.3

    if approx_tokens <= WINDOW_TOKENS:
        chunk_texts = [text]
    else:
        ids    = _TOKENIZER.encode(text).ids
        stride = WINDOW_TOKENS - OVERLAP_TOKENS
        chunks = []
        start  = 0
        while start < len(ids):
            window = ids[start : start + WINDOW_TOKENS]
            chunks.append(_TOKENIZER.decode(window))
            start += stride
            if start + WINDOW_TOKENS >= len(ids):
                tail = ids[start:]
                if tail:
                    chunks.append(_TOKENIZER.decode(tail))
                break
        chunk_texts = chunks or [text]

    chunks_raw = _ort_predict(chunk_texts)

    total_weight = ai_sum = human_sum = conf_sum = 0.0

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

    return (
        round(ai_sum    / total_weight, 2),
        round(human_sum / total_weight, 2),
        round(conf_sum  / total_weight, 4),
    )


# ── Heuristic fallback ────────────────────────────────────────────────────────

def _heuristic_fallback(text: str) -> tuple[float, float, float]:
    """Fallback thuần Python khi ONNX không load được. Confidence ≤ 0.55."""
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
    marker_hits = sum(1 for m in AI_MARKERS if m in text.lower())

    ai_raw    = max(0, (0.72 - ttr) * 60) + max(0, (8.0 - std_dev) * 2.5) + min(30, marker_hits * 7)
    ai_pct    = round(min(92, max(8, ai_raw)), 2)
    human_pct = round(100 - ai_pct, 2)
    confidence = round(0.35 + min(1.0, len(words) / 250) * 0.20, 4)

    return ai_pct, human_pct, confidence


# ── Public entry point ────────────────────────────────────────────────────────

async def analyze_text(text: str) -> dict[str, Any]:
    t0 = time.time()

    if _pipeline is not None:
        try:
            ai_pct, human_pct, confidence = await asyncio.to_thread(_run_inference, text)
            source = "onnx-model"
        except Exception as exc:
            log.warning("Inference error (%s) — heuristic fallback", exc)
            ai_pct, human_pct, confidence = _heuristic_fallback(text)
            source = "heuristic-fallback"
    else:
        ai_pct, human_pct, confidence = _heuristic_fallback(text)
        source = "heuristic"

    return {
        "models": [{
            "modelId":       MODEL_ID,
            "modelName":     MODEL_NAME,
            "weight":        MODEL_WEIGHT,
            "ai_percent":    ai_pct,
            "human_percent": human_pct,
            "confidence":    confidence,
            "score":         human_pct,
            "latencyMs":     round((time.time() - t0) * 1000),
            "source":        source,
        }]
    }


# ══════════════════════════════════════════════════════════
#  FastAPI  —  GET /health   POST /detect
# ══════════════════════════════════════════════════════════

import os
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("text-service ready  model=%s  runtime=%s",
             MODEL_ID, "onnx" if _pipeline else "heuristic-only")
    yield


app = FastAPI(title="Text Detection Service", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"])


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
    m = result["models"][0]
    return {
        "modelId":       m["modelId"],
        "modelName":     m["modelName"],
        "ai_percent":    m["ai_percent"],
        "human_percent": m["human_percent"],
        "confidence":    m["confidence"],
        "score":         m["score"],
        "latencyMs":     m["latencyMs"],
        "source":        m["source"],
    }


if __name__ == "__main__":
    uvicorn.run("main:app",
                host="0.0.0.0",
                port=int(os.getenv("PORT", 8001)),
                reload=False)
