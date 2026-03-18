"""
Fake News Detection Service
══════════════════════════════════════════════════════════════
Runtime       : Heuristic thuần — KHÔNG có torch, transformers,
                onnxruntime, hay bất kỳ ML dependency nào
RAM           : ~35 MB  (chỉ FastAPI + uvicorn + httpx)
Startup       : < 1s

Phương pháp
───────────
5 tầng phân tích độc lập, kết hợp có trọng số:

  1. Sensationalism score   — clickbait phrases, ALLCAPS, dấu !!!
  2. Credibility score      — citation markers, nguồn uy tín
  3. Linguistic score       — hedge words, absolute language, weasel words
  4. Structure score        — độ dài bài, tỉ lệ câu hỏi, đoạn văn
  5. Source score           — domain trong URL (nếu có)

Mỗi tầng trả về 0–100 (100 = chắc chắn fake).
Weighted average → fake_percent cuối cùng.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any
from urllib.parse import urlparse

log = logging.getLogger("aiproof.fake_news")

MODEL_ID     = "heuristic-advanced-v2"
MODEL_NAME   = "Advanced Heuristic Analyzer"
MODEL_WEIGHT = 1.0
MAX_CHARS    = 8_000

# ── Layer 1: Sensationalism ───────────────────────────────────────────────────

_SENSATIONAL_HIGH = [
    "you won't believe", "they don't want you to know", "hidden truth",
    "mainstream media lies", "wake up sheeple", "plandemic", "false flag",
    "deep state", "big pharma conspiracy", "doctors don't want",
    "miracle cure", "secret remedy", "banned by", "what they're hiding",
    "the truth about", "exposed!", "they lied", "cover-up revealed",
]

_SENSATIONAL_MED = [
    "breaking", "shocking", "bombshell", "explosive", "urgent",
    "exposed", "hoax", "cover-up", "banned", "censored",
    "wake up", "sheeple", "mainstream media", "crisis actors",
    "they don't want", "hidden agenda", "new world order",
    "mind control", "chemtrails", "microchip",
]

_CLICKBAIT_PATTERNS = [
    r"\b\d+\s+reasons?\b",
    r"\bthis is why\b",
    r"\byou need to (know|see|read)\b",
    r"\bwhat (they|he|she|no one) (won't|didn't|doesn't)\b",
    r"\bthe (real|shocking|hidden|dark) truth\b",
    r"\bthis will (shock|surprise|blow your mind)\b",
]

# ── Layer 2: Credibility ──────────────────────────────────────────────────────

_CREDIBILITY_STRONG = [
    "peer-reviewed", "published in", "according to the study",
    "clinical trial", "randomized controlled", "meta-analysis",
    "systematic review", "official statement", "press release",
    "confirmed by", "verified by", "fact-checked",
]

_CREDIBILITY_MED = [
    "according to", "researchers found", "study shows",
    "data shows", "report says", "spokesperson said",
    "experts say", "scientists found", "survey found",
    "statistics show", "evidence suggests", "analysis found",
]

_CREDIBLE_DOMAINS = [
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "theguardian.com", "nytimes.com", "washingtonpost.com",
    "nature.com", "science.org", "who.int", "cdc.gov",
    "nih.gov", "gov.uk", "europa.eu", "un.org",
]

_SUSPICIOUS_DOMAINS = [
    "infowars", "naturalnews", "beforeitsnews", "zerohedge",
    "breitbart", "thegatewaypundit", "worldnewsdailyreport",
    "empirenews", "realnewsrightnow", "abcnews.com.co",
]

# ── Layer 3: Linguistic ───────────────────────────────────────────────────────

_ABSOLUTE_WORDS = [
    "always", "never", "everyone", "nobody", "all", "none",
    "every single", "without exception", "100%", "definitely",
    "certainly", "undoubtedly", "absolutely", "proven fact",
    "irrefutable", "undeniable", "unquestionable",
]

_HEDGE_WORDS = [
    "allegedly", "reportedly", "according to", "it appears",
    "it seems", "may", "might", "could", "possibly", "perhaps",
    "some sources", "unconfirmed", "unclear",
]

_WEASEL_WORDS = [
    "many people say", "some people think", "experts believe",
    "sources say", "it is said", "rumor has it",
    "word is", "people are saying", "they say",
]

# ── Scoring functions ─────────────────────────────────────────────────────────

def _score_sensationalism(text: str, lower: str) -> tuple[float, list[dict]]:
    signals = []
    score   = 0.0

    for phrase in _SENSATIONAL_HIGH:
        if phrase in lower:
            score += 18
            signals.append({"type": "sensationalism", "phrase": phrase, "severity": "high"})

    med_hits = sum(1 for p in _SENSATIONAL_MED if p in lower)
    score += min(30, med_hits * 7)
    for p in _SENSATIONAL_MED:
        if p in lower:
            signals.append({"type": "sensationalism", "phrase": p, "severity": "medium"})

    for pattern in _CLICKBAIT_PATTERNS:
        if re.search(pattern, lower):
            score += 8
            signals.append({"type": "clickbait", "phrase": pattern, "severity": "medium"})

    allcaps = [w for w in re.findall(r'\b[A-Z]{4,}\b', text)
               if w not in ("NASA", "NATO", "IAEA", "UEFA", "FIFA", "HTTP", "HTML")]
    if len(allcaps) > 3:
        score += min(15, len(allcaps) * 2)
        signals.append({"type": "allcaps", "phrase": f"{len(allcaps)} ALL-CAPS words", "severity": "medium"})

    exclaim = text.count("!")
    if exclaim > 3:
        score += min(12, exclaim * 2)
        signals.append({"type": "punctuation", "phrase": f"{exclaim} exclamation marks", "severity": "low"})

    return min(100.0, score), signals


def _score_credibility(lower: str) -> tuple[float, list[dict]]:
    signals = []
    score   = 50.0

    for phrase in _CREDIBILITY_STRONG:
        if phrase in lower:
            score -= 12
            signals.append({"type": "credibility_marker", "phrase": phrase, "severity": "positive"})

    cred_hits = sum(1 for p in _CREDIBILITY_MED if p in lower)
    score -= min(20, cred_hits * 5)
    for p in _CREDIBILITY_MED:
        if p in lower:
            signals.append({"type": "credibility_marker", "phrase": p, "severity": "positive"})

    return max(0.0, min(100.0, score)), signals


def _score_linguistic(text: str, lower: str) -> tuple[float, list[dict]]:
    signals = []
    score   = 30.0

    abs_hits = sum(1 for w in _ABSOLUTE_WORDS if w in lower)
    score += min(25, abs_hits * 6)
    if abs_hits:
        signals.append({"type": "absolute-language", "phrase": f"{abs_hits} absolute claims", "severity": "medium"})

    weasel_hits = sum(1 for w in _WEASEL_WORDS if w in lower)
    score += min(20, weasel_hits * 7)
    if weasel_hits:
        signals.append({"type": "weasel-words", "phrase": f"{weasel_hits} unattributed claims", "severity": "medium"})

    hedge_hits = sum(1 for w in _HEDGE_WORDS if w in lower)
    score -= min(15, hedge_hits * 3)

    emotional = len(re.findall(
        r'\b(outrage|disgusting|evil|corrupt|traitor|tyranny|freedom|liberty|'
        r'communist|fascist|globalist|elite|puppet|slave|regime|coup)\b', lower
    ))
    if emotional > 2:
        score += min(15, emotional * 3)
        signals.append({"type": "emotional-language", "phrase": f"{emotional} emotionally charged words", "severity": "medium"})

    return max(0.0, min(100.0, score)), signals


def _score_structure(text: str) -> tuple[float, list[dict]]:
    signals    = []
    score      = 25.0
    words      = text.split()
    word_count = len(words)
    sentences  = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
    n_sent     = max(1, len(sentences))

    if word_count < 100:
        score += 20
        signals.append({"type": "structure", "phrase": f"Very short article ({word_count} words)", "severity": "medium"})
    elif word_count > 600:
        score -= 10

    avg_sent_len = word_count / n_sent
    if avg_sent_len < 8:
        score += 10
        signals.append({"type": "structure", "phrase": "Unusually short sentences", "severity": "low"})

    q_ratio = text.count("?") / n_sent
    if q_ratio > 0.3:
        score += 10
        signals.append({"type": "structure", "phrase": "High ratio of rhetorical questions", "severity": "low"})

    unique_ratio = len(set(w.lower() for w in words)) / max(1, word_count)
    if unique_ratio < 0.4:
        score += 8
        signals.append({"type": "structure", "phrase": "Low vocabulary diversity", "severity": "low"})

    return max(0.0, min(100.0, score)), signals


def _score_source(url: str | None) -> tuple[float, list[dict]]:
    if not url:
        return 50.0, []

    signals = []
    try:
        domain = urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return 50.0, []

    for d in _CREDIBLE_DOMAINS:
        if d in domain:
            signals.append({"type": "source", "phrase": f"Credible domain: {domain}", "severity": "positive"})
            return 10.0, signals

    for d in _SUSPICIOUS_DOMAINS:
        if d in domain:
            signals.append({"type": "source", "phrase": f"Known unreliable domain: {domain}", "severity": "high"})
            return 85.0, signals

    return 50.0, signals


# ── Main scorer ───────────────────────────────────────────────────────────────

def _run_heuristic(text: str, url: str | None = None) -> tuple[float, float, float, list[dict]]:
    lower = text.lower()

    s1, sig1 = _score_sensationalism(text, lower)
    s2, sig2 = _score_credibility(lower)
    s3, sig3 = _score_linguistic(text, lower)
    s4, sig4 = _score_structure(text)
    s5, sig5 = _score_source(url)

    fake_pct = (
        s1 * 0.30 +
        s2 * 0.25 +
        s3 * 0.20 +
        s4 * 0.10 +
        s5 * 0.15
    )
    fake_pct   = round(min(97, max(3, fake_pct)), 2)
    real_pct   = round(100 - fake_pct, 2)
    confidence = round(abs(fake_pct / 100 - 0.5) * 2, 4)

    all_signals = sig1 + sig2 + sig3 + sig4 + sig5
    severity_order = {"high": 0, "medium": 1, "low": 2, "positive": 3}
    all_signals.sort(key=lambda s: severity_order.get(s.get("severity", "low"), 2))

    return fake_pct, real_pct, confidence, all_signals[:12]


# ── URL fetcher ───────────────────────────────────────────────────────────────

async def fetch_url(url: str, timeout: float = 15.0) -> str:
    try:
        import httpx
    except ImportError:
        raise ValueError("httpx is not installed — cannot fetch URLs")

    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "AI-PROOF/2.2 fake-news-detector"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text
    except Exception as exc:
        raise ValueError(f"Could not fetch URL: {exc}") from exc

    return _clean_html(html)


def _clean_html(html: str) -> str:
    html = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", " ",
                  html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


# ── Public entry point ────────────────────────────────────────────────────────

async def detect_fake_news(text: str, url: str | None = None) -> dict[str, Any]:
    t0 = time.time()

    fake_pct, real_pct, confidence, signals = await asyncio.to_thread(
        _run_heuristic, text, url
    )

    return {
        "models": [{
            "modelId":      MODEL_ID,
            "modelName":    MODEL_NAME,
            "weight":       MODEL_WEIGHT,
            "fake_percent": fake_pct,
            "real_percent": real_pct,
            "confidence":   confidence,
            "score":        real_pct,
            "signals":      signals,
            "latencyMs":    round((time.time() - t0) * 1000),
            "source":       "heuristic",
        }]
    }


# ══════════════════════════════════════════════════════════
#  FastAPI  —  GET /health   POST /detect
# ══════════════════════════════════════════════════════════

import os
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("fakenews-service ready  runtime=heuristic-only  ram=~35MB")
    yield


app = FastAPI(title="Fake News Detection Service", version="3.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"])


class FakeNewsRequest(BaseModel):
    content: Optional[str] = None
    url:     Optional[str] = None

    @field_validator("url")
    @classmethod
    def _check_url(cls, v: str | None) -> str | None:
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("url must start with http:// or https://")
        return v


@app.get("/health")
async def health():
    return {"status": "ok", "service": "fakenews-detection",
            "runtime": "heuristic", "pipeline": True, "version": "3.0.0"}


@app.post("/detect")
async def detect(req: FakeNewsRequest):
    if req.url:
        try:
            text = await fetch_url(req.url)
        except Exception as exc:
            raise HTTPException(422, detail=str(exc))
        if not text or len(text.strip()) < 10:
            raise HTTPException(422, detail="Could not extract readable text from URL")
        url = req.url
    elif req.content and len(req.content.strip()) >= 10:
        text = req.content.strip()
        url  = None
    else:
        raise HTTPException(400, detail="Provide 'content' (min 10 chars) or a valid 'url'")

    result = await detect_fake_news(text, url)
    model  = result["models"][0]

    return {
        "modelId":      model["modelId"],
        "modelName":    model["modelName"],
        "fake_percent": model["fake_percent"],
        "real_percent": model["real_percent"],
        "confidence":   model["confidence"],
        "score":        model["score"],
        "signals":      model["signals"],
        "latencyMs":    model["latencyMs"],
        "source":       model["source"],
        "url":          req.url,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("main:app", host="0.0.0.0",
                port=int(os.getenv("PORT", 8002)), reload=False)
