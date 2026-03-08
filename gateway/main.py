"""
AI-PROOF  —  API Gateway
════════════════════════════════════════════════════════════
Single entry-point for the frontend.
Calls 3 microservices in parallel (asyncio.gather) and
aggregates results into a unified consensus verdict.

Routes
──────
GET  /health              → liveness (also pings all services)
POST /analyze/text        → text + fakenews in parallel → consensus
POST /analyze/url         → fakenews service fetches URL
POST /analyze/image       → image service
POST /detect-text         → text service only
POST /detect-fake-news    → fakenews service only
POST /detect-ai-image     → image service only

Env vars
────────
TEXT_SERVICE_URL      default http://localhost:8001
FAKENEWS_SERVICE_URL  default http://localhost:8002
IMAGE_SERVICE_URL     default http://localhost:8003
PORT                  default 8000
CORS_ORIGINS          comma-separated, default *
"""
from __future__ import annotations
import asyncio, logging, os, re, time
from contextlib import asynccontextmanager
from typing import Any, Optional
import httpx, uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gateway")

# ── Service URLs ──────────────────────────────────────────────────────────────

TEXT_URL     = os.getenv("TEXT_SERVICE_URL",    "http://localhost:8001")
FAKENEWS_URL = os.getenv("FAKENEWS_SERVICE_URL","http://localhost:8002")
IMAGE_URL    = os.getenv("IMAGE_SERVICE_URL",   "http://localhost:8003")
TIMEOUT      = httpx.Timeout(60.0, connect=5.0)

# ── HTTP client (shared, keep-alive) ─────────────────────────────────────────

_client: httpx.AsyncClient | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    _client = httpx.AsyncClient(timeout=TIMEOUT)
    log.info("Gateway ready  text=%s  fakenews=%s  image=%s",
             TEXT_URL, FAKENEWS_URL, IMAGE_URL)
    yield
    await _client.aclose()

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="AI-PROOF Gateway", version="2.0.0", lifespan=lifespan)

_origins = [o.strip() for o in os.getenv("CORS_ORIGINS","*").split(",")]
app.add_middleware(CORSMiddleware,
    allow_origins=_origins, allow_credentials=True,
    allow_methods=["GET","POST","OPTIONS"], allow_headers=["*"])

@app.exception_handler(Exception)
async def _err(request: Request, exc: Exception):
    log.exception("Unhandled error: %s", exc)
    return JSONResponse(500, content={"detail": str(exc)})

# ── Service caller ────────────────────────────────────────────────────────────

async def _call(url: str, body: dict, retries: int = 1) -> dict:
    """POST to a microservice. Retries once on network error."""
    for attempt in range(retries + 1):
        try:
            r = await _client.post(url + "/detect", json=body)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            detail = e.response.text[:200]
            raise HTTPException(e.response.status_code, detail=detail)
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            if attempt == retries:
                raise HTTPException(503, detail=f"Service unreachable: {url}  ({e})")
            await asyncio.sleep(0.5)

async def _health_service(url: str, name: str) -> dict:
    try:
        r = await _client.get(url + "/health", timeout=4)
        d = r.json()
        return {"service": name, "status": "ok", "pipeline": d.get("pipeline"), "model": d.get("model")}
    except Exception as e:
        return {"service": name, "status": "unreachable", "error": str(e)}

# ── Consensus ─────────────────────────────────────────────────────────────────

MODEL_WEIGHTS = {
    "Hello-SimpleAI/chatgpt-detector-roberta": 0.40,
    "cross-encoder/nli-MiniLM2-L6-H768":       0.35,
    "openai/clip-vit-base-patch16":             0.25,
}

def _consensus(results: list[dict]) -> dict:
    """Weighted average of trust scores from multiple service results."""
    w_sum = w_total = 0.0
    for r in results:
        mid    = r.get("modelId", "")
        weight = MODEL_WEIGHTS.get(mid, 0.25) * (r.get("confidence", 0.75))
        score  = r.get("score", 50)
        w_sum   += score * weight
        w_total += weight
    trust = round(w_sum / w_total) if w_total else 50
    scores = [r.get("score", 50) for r in results]
    mean   = sum(scores) / len(scores)
    var    = sum((s-mean)**2 for s in scores) / len(scores)
    agreement = max(0, min(100, round(100 - var**0.5)))
    return {"trustScore": trust, "agreement": agreement}

VERDICTS = [(85,"AUTHENTIC","badge-green","#00ff9d"),
            (70,"LIKELY REAL","badge-green","#7aff6e"),
            (50,"UNCERTAIN","badge-yellow","#ffb300"),
            (30,"SUSPICIOUS","badge-yellow","#ff7a00"),
            (0,"AI-GENERATED","badge-red","#ff3d5a")]

def _verdict(score: int) -> dict:
    for t,l,bc,c in VERDICTS:
        if score >= t: return {"label":l,"class":bc,"color":c}
    return {"label":"UNKNOWN","class":"badge-cyan","color":"#00e5ff"}

def _ms(t0): return round((time.time()-t0)*1000)

# ── Schemas ───────────────────────────────────────────────────────────────────

class TextReq(BaseModel):
    content: str
    @field_validator("content")
    @classmethod
    def _v(cls, v):
        if not v or len(v.strip()) < 10: raise ValueError("content must be ≥ 10 chars")
        return v.strip()

class URLReq(BaseModel):
    url: str
    @field_validator("url")
    @classmethod
    def _v(cls, v):
        if not v.startswith(("http://","https://")): raise ValueError("invalid url")
        return v.strip()

class ImageReq(BaseModel):
    image: str
    @field_validator("image")
    @classmethod
    def _v(cls, v):
        if not v or len(v.strip()) < 20: raise ValueError("invalid base64")
        return v.strip()

class FakeNewsReq(BaseModel):
    content: Optional[str] = None
    url:     Optional[str] = None

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
async def health():
    t0 = time.time()
    services = await asyncio.gather(
        _health_service(TEXT_URL,     "text-service"),
        _health_service(FAKENEWS_URL, "fakenews-service"),
        _health_service(IMAGE_URL,    "image-service"),
    )
    all_ok = all(s["status"] == "ok" for s in services)
    return {
        "status":    "ok" if all_ok else "degraded",
        "services":  list(services),
        "latencyMs": _ms(t0),
    }


@app.post("/analyze/text", tags=["analyze"])
async def analyze_text(req: TextReq):
    """Text + fake-news in parallel → consensus verdict."""
    t0 = time.time()
    text_res, fake_res = await asyncio.gather(
        _call(TEXT_URL,     {"content": req.content}),
        _call(FAKENEWS_URL, {"content": req.content}),
    )
    models  = [text_res, fake_res]
    con     = _consensus(models)
    signals = fake_res.get("signals", [])

    return {
        "trustScore":   con["trustScore"],
        "verdict":      _verdict(con["trustScore"]),
        "agreement":    con["agreement"],
        "models":       models,
        "signals":      signals,
        "explanation":  "",
        "processingMs": _ms(t0),
        "source":       "gateway",
    }


@app.post("/analyze/url", tags=["analyze"])
async def analyze_url(req: URLReq):
    """Fake-news service fetches + analyses the URL."""
    t0  = time.time()
    res = await _call(FAKENEWS_URL, {"url": req.url},
                      retries=0)   # URL fetch already has its own timeout
    trust = res.get("score", 50)
    return {
        "trustScore":   round(trust),
        "verdict":      _verdict(round(trust)),
        "models":       [res],
        "signals":      res.get("signals", []),
        "explanation":  "",
        "url":          req.url,
        "processingMs": _ms(t0),
        "source":       "gateway",
    }


@app.post("/analyze/image", tags=["analyze"])
async def analyze_image(req: ImageReq):
    """Full image pipeline via image-service."""
    t0  = time.time()
    res = await _call(IMAGE_URL, {"image": req.image})
    return {
        **res,
        "processingMs": _ms(t0),
        "source":       "gateway",
    }


# ── Thin pass-through endpoints (single-service) ─────────────────────────────

@app.post("/detect-text", tags=["detect"])
async def detect_text(req: TextReq):
    t0  = time.time()
    res = await _call(TEXT_URL, {"content": req.content})
    return {**res, "processingMs": _ms(t0)}


@app.post("/detect-fake-news", tags=["detect"])
async def detect_fake_news(req: FakeNewsReq):
    if not req.content and not req.url:
        raise HTTPException(400, detail="Provide 'content' or 'url'")
    t0   = time.time()
    body = {"url": req.url} if req.url else {"content": req.content}
    res  = await _call(FAKENEWS_URL, body)
    return {**res, "processingMs": _ms(t0)}


@app.post("/detect-ai-image", tags=["detect"])
async def detect_ai_image(req: ImageReq):
    t0  = time.time()
    res = await _call(IMAGE_URL, {"image": req.image})
    return {**res, "processingMs": _ms(t0)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0",
                port=int(os.getenv("PORT", 8000)), reload=False)