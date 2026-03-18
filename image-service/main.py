"""
AI Image Detection Service
══════════════════════════════════════════════════════════════
Runtime       : CPU-only, không torch, không onnxruntime, không CLIP
Phân tích     : DCT, noise uniformity, colour channel correlation,
                edge consistency, compression artifacts
EXIF          : software markers, missing camera info, GPS
RAM           : ~60–80 MB (Pillow + numpy + FastAPI)

Thay đổi so với bản CLIP
─────────────────────────
- Bỏ hoàn toàn onnxruntime, tokenizers, huggingface_hub, CLIP
- Tăng cường pixel forensics: thêm edge consistency + JPEG artifact check
- Trọng số: pixel forensics 70%, EXIF 30%
- RAM ~60-80MB, không bao giờ OOM trên Render Free 512MB
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import math
import time
from typing import Any

import numpy as np

log = logging.getLogger("aiproof.ai_image")

# ── AI software markers ───────────────────────────────────────────────────────

_AI_SOFTWARE_MARKERS = [
    "stable diffusion", "midjourney", "dalle", "dall-e",
    "firefly", "gen-2", "runway", "comfy", "automatic1111",
    "novelai", "invoke", "dream", "artbreeder", "adobe firefly",
    "canva ai", "bing image creator", "ideogram", "leonardo",
]

# ── Image decoding ────────────────────────────────────────────────────────────

def _decode_image(image_b64: str):
    from PIL import Image
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    try:
        raw   = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        return image, raw
    except Exception as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc


# ── DCT helper ────────────────────────────────────────────────────────────────

def _dct1d(x: np.ndarray) -> np.ndarray:
    n      = len(x)
    result = np.zeros(n)
    for k in range(n):
        result[k] = sum(
            x[i] * math.cos(math.pi * k * (2 * i + 1) / (2 * n))
            for i in range(n)
        )
    return result


def _dct2(block: np.ndarray) -> np.ndarray:
    n      = block.shape[0]
    result = np.zeros_like(block)
    for i in range(n):
        result[i] = _dct1d(block[i])
    for j in range(n):
        result[:, j] = _dct1d(result[:, j])
    return result


# ── Pixel forensics ───────────────────────────────────────────────────────────

def _pixel_forensics(image) -> dict[str, Any]:
    img_arr = np.array(image, dtype=np.float32)
    h, w, _ = img_arr.shape
    grey     = img_arr.mean(axis=2)
    signals  = []
    scores   = []

    # ── 1. DCT high-frequency energy ─────────────────────────────────────────
    try:
        hf_ratios = []
        rng = np.random.default_rng(42)
        for _ in range(20):
            ry = int(rng.integers(0, max(1, h - 8)))
            rx = int(rng.integers(0, max(1, w - 8)))
            block = grey[ry:ry+8, rx:rx+8]
            dct   = _dct2(block)
            dc    = float(dct[0, 0] ** 2)
            ac    = float((dct ** 2).sum() - dc)
            hf_ratios.append(ac / (dc + 1e-6))
        avg_hf = float(np.mean(hf_ratios))
        if avg_hf < 0.10:
            scores.append(85.0)
            signals.append({"type": "dct-frequency", "label": "Unnaturally smooth frequency spectrum", "strength": 0.90})
        elif avg_hf < 0.20:
            scores.append(60.0)
            signals.append({"type": "dct-frequency", "label": "Low high-frequency energy", "strength": 0.60})
        elif avg_hf < 0.35:
            scores.append(30.0)
            signals.append({"type": "dct-frequency", "label": "Slightly low frequency variance", "strength": 0.30})
        else:
            scores.append(15.0)
            signals.append({"type": "dct-frequency", "label": "Natural camera noise pattern", "strength": 0.10})
    except Exception as e:
        log.debug("DCT failed: %s", e)

    # ── 2. Noise uniformity ───────────────────────────────────────────────────
    try:
        ch, cw = max(1, h // 4), max(1, w // 4)
        local_stds = [
            float(np.std(grey[i*ch:(i+1)*ch, j*cw:(j+1)*cw]))
            for i in range(4) for j in range(4)
        ]
        std_of_stds = float(np.std(local_stds))
        if std_of_stds < 2.5:
            scores.append(80.0)
            signals.append({"type": "noise-uniformity", "label": "Suspiciously uniform noise across regions", "strength": 0.85})
        elif std_of_stds < 7.0:
            scores.append(45.0)
            signals.append({"type": "noise-uniformity", "label": "Slightly uniform noise distribution", "strength": 0.45})
        else:
            scores.append(15.0)
            signals.append({"type": "noise-uniformity", "label": "Natural regional noise variation", "strength": 0.10})
    except Exception as e:
        log.debug("Noise failed: %s", e)

    # ── 3. Colour channel correlation ─────────────────────────────────────────
    try:
        r = img_arr[:, :, 0].flatten()
        g = img_arr[:, :, 1].flatten()
        b = img_arr[:, :, 2].flatten()
        rg = abs(float(np.corrcoef(r, g)[0, 1]))
        rb = abs(float(np.corrcoef(r, b)[0, 1]))
        avg_corr = (rg + rb) / 2
        if avg_corr > 0.97:
            scores.append(75.0)
            signals.append({"type": "colour-correlation", "label": "Unnaturally high channel correlation", "strength": 0.80})
        elif avg_corr > 0.90:
            scores.append(40.0)
            signals.append({"type": "colour-correlation", "label": "Elevated colour channel correlation", "strength": 0.40})
        else:
            scores.append(10.0)
    except Exception as e:
        log.debug("Colour failed: %s", e)

    # ── 4. Edge consistency (Laplacian variance) ──────────────────────────────
    try:
        # Laplacian kernel
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        # Sample 16 random 32x32 patches
        edge_vars = []
        rng2 = np.random.default_rng(7)
        for _ in range(16):
            py = int(rng2.integers(0, max(1, h - 32)))
            px = int(rng2.integers(0, max(1, w - 32)))
            patch = grey[py:py+32, px:px+32]
            # Manual 2D convolution using stride tricks (small patch, OK)
            from numpy.lib.stride_tricks import sliding_window_view
            windows = sliding_window_view(patch, (3, 3))
            lap = (windows * kernel).sum(axis=(-1, -2))
            edge_vars.append(float(np.var(lap)))
        avg_edge_var = float(np.mean(edge_vars))
        var_of_vars  = float(np.std(edge_vars))
        # AI images tend to have very consistent (low variance) edge patterns
        if var_of_vars < 50:
            scores.append(70.0)
            signals.append({"type": "edge-consistency", "label": "Unnaturally consistent edge sharpness", "strength": 0.75})
        elif var_of_vars < 150:
            scores.append(35.0)
            signals.append({"type": "edge-consistency", "label": "Slightly uniform edge patterns", "strength": 0.35})
        else:
            scores.append(10.0)
            signals.append({"type": "edge-consistency", "label": "Natural edge variation", "strength": 0.10})
    except Exception as e:
        log.debug("Edge failed: %s", e)

    # ── 5. JPEG compression artifact check ───────────────────────────────────
    try:
        # AI images resampled/saved often have unusually clean 8x8 block boundaries
        block_diffs = []
        for by in range(0, min(h - 1, 128), 8):
            row_above = grey[by,     max(0, w//4):min(w, 3*w//4)]
            row_below = grey[by + 1, max(0, w//4):min(w, 3*w//4)]
            block_diffs.append(float(np.mean(np.abs(row_above - row_below))))
        if len(block_diffs) > 2:
            bd_std = float(np.std(block_diffs))
            if bd_std < 1.5:
                scores.append(65.0)
                signals.append({"type": "compression-artifacts", "label": "Suspiciously clean block boundaries", "strength": 0.65})
            elif bd_std < 4.0:
                scores.append(30.0)
                signals.append({"type": "compression-artifacts", "label": "Slightly uniform block transitions", "strength": 0.30})
            else:
                scores.append(10.0)
    except Exception as e:
        log.debug("JPEG artifact check failed: %s", e)

    composite = round(sum(scores) / len(scores), 2) if scores else 50.0
    return {"score": composite, "signals": signals}


# ── EXIF analysis ─────────────────────────────────────────────────────────────

def _exif_analysis(raw_bytes: bytes) -> dict[str, Any]:
    signals  = []
    metadata = {}
    score    = 0.0

    try:
        from PIL import Image, ExifTags
        img      = Image.open(io.BytesIO(raw_bytes))
        exif_raw = img._getexif() if hasattr(img, "_getexif") else None
        exif     = {}
        if exif_raw:
            exif = {ExifTags.TAGS.get(k, k): v for k, v in exif_raw.items()}

        if not exif:
            score += 30
            signals.append({"type": "exif-absent", "label": "No EXIF metadata (common in AI images)", "strength": 0.65})
            metadata["hasExif"] = False
        else:
            metadata["hasExif"] = True

            # AI software marker
            software = str(exif.get("Software", "")).lower()
            metadata["software"] = exif.get("Software")
            for marker in _AI_SOFTWARE_MARKERS:
                if marker in software:
                    score += 70
                    signals.append({
                        "type": "exif-software",
                        "label": f"AI software detected: {exif.get('Software')}",
                        "strength": 0.98,
                    })
                    break

            # No camera make/model
            if not exif.get("Make") and not exif.get("Model"):
                score += 25
                signals.append({"type": "exif-no-camera", "label": "No camera make/model in metadata", "strength": 0.55})

            # No GPS (phones almost always have GPS)
            if "GPSInfo" not in exif:
                score += 10
                signals.append({"type": "exif-no-gps", "label": "No GPS data", "strength": 0.25})

            # Check for suspiciously round/zero shutter speed or ISO
            iso = exif.get("ISOSpeedRatings")
            if iso and iso in (0, 1):
                score += 15
                signals.append({"type": "exif-invalid-iso", "label": "Invalid ISO value in metadata", "strength": 0.50})

            metadata["cameraMake"]  = exif.get("Make")
            metadata["cameraModel"] = exif.get("Model")
            metadata["hasGPS"]      = "GPSInfo" in exif
            metadata["software"]    = exif.get("Software")

        metadata.update({
            "width":  img.width,
            "height": img.height,
            "format": img.format,
        })

    except Exception as exc:
        log.debug("EXIF error: %s", exc)

    return {"score": round(min(100, score), 2), "signals": signals, "metadata": metadata}


# ── Verdict helpers ───────────────────────────────────────────────────────────

VERDICT_THRESHOLDS = [
    (85, "AUTHENTIC",    "#00ff9d"),
    (70, "LIKELY REAL",  "#7aff6e"),
    (50, "UNCERTAIN",    "#ffb300"),
    (30, "SUSPICIOUS",   "#ff7a00"),
    ( 0, "AI-GENERATED", "#ff3d5a"),
]


def _verdict(score: int) -> dict:
    for threshold, label, color in VERDICT_THRESHOLDS:
        if score >= threshold:
            return {"label": label, "color": color}
    return {"label": "UNKNOWN", "color": "#00e5ff"}


def _ms(t0: float) -> int:
    return round((time.time() - t0) * 1000)


# ── Public entry point ────────────────────────────────────────────────────────

async def analyze_image(image_b64: str) -> dict[str, Any]:
    t0 = time.time()

    try:
        image, raw_bytes = _decode_image(image_b64)
    except ValueError as exc:
        return {
            "trustScore":   50,
            "ai_percent":   50.0,
            "real_percent": 50.0,
            "confidence":   0.0,
            "verdict":      _verdict(50),
            "models":       [],
            "signals":      [],
            "metadata":     {"error": str(exc)},
            "processingMs": _ms(t0),
            "source":       "error",
        }

    # Run both analyses concurrently
    pixel_result, exif_result = await asyncio.gather(
        asyncio.to_thread(_pixel_forensics, image),
        asyncio.to_thread(_exif_analysis,   raw_bytes),
    )

    pixel_ai = pixel_result["score"]
    exif_ai  = exif_result["score"]

    # Weighted combine: pixel 70%, EXIF 30%
    combined_ai  = pixel_ai * 0.70 + exif_ai * 0.30
    ai_pct       = round(min(98, max(2, combined_ai)), 2)
    real_pct     = round(100 - ai_pct, 2)

    pixel_conf   = round(abs(pixel_ai / 100 - 0.5) * 2, 4)
    exif_conf    = round(abs(exif_ai  / 100 - 0.5) * 2, 4)
    confidence   = round(pixel_conf * 0.7 + exif_conf * 0.3, 4)

    trust_score  = round(real_pct)

    return {
        "trustScore":   trust_score,
        "ai_percent":   ai_pct,
        "real_percent": real_pct,
        "confidence":   confidence,
        "verdict":      _verdict(trust_score),
        "models": [
            {
                "modelId":      "pixel-forensics",
                "modelName":    "Pixel Forensics",
                "weight":       0.70,
                "ai_percent":   pixel_ai,
                "real_percent": round(100 - pixel_ai, 2),
                "score":        round(100 - pixel_ai),
                "confidence":   pixel_conf,
                "source":       "local",
            },
            {
                "modelId":      "exif-analysis",
                "modelName":    "EXIF Analysis",
                "weight":       0.30,
                "ai_percent":   exif_ai,
                "real_percent": round(100 - exif_ai, 2),
                "score":        round(100 - exif_ai),
                "confidence":   exif_conf,
                "source":       "local",
            },
        ],
        "signals":      pixel_result.get("signals", []) + exif_result.get("signals", []),
        "metadata":     exif_result.get("metadata", {}),
        "processingMs": _ms(t0),
        "source":       "forensics-only",
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
    log.info("image-service ready  runtime=forensics-only  ram=~70MB")
    yield


app = FastAPI(
    title="Image Detection Service",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageRequest(BaseModel):
    image: str

    @field_validator("image")
    @classmethod
    def _check(cls, v: str) -> str:
        if not v or len(v.strip()) < 20:
            raise ValueError("image must be a non-empty base64 string")
        return v.strip()


@app.get("/health")
async def health():
    return {
        "status":   "ok",
        "service":  "image-detection",
        "runtime":  "forensics-only",
        "pipeline": True,
        "version":  "2.0.0",
    }


@app.post("/detect")
async def detect(req: ImageRequest):
    return await analyze_image(req.image)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8003)),
        reload=False,
    )
