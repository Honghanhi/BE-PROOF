"""
AI-PROOF  —  Utility Service  v2.0
════════════════════════════════════════════════════════════════
Hợp nhất tất cả tính năng phụ trợ + proxy endpoints để
frontend gọi qua backend (tránh CORS).

Routes (gốc)
────────────
GET  /health
POST /consensus
POST /explain
POST /version/compare
POST /blockchain/verify

Routes (mới — proxy tránh CORS)
────────────────────────────────
POST /proxy/ai              → Proxy Claude/Gemini/Groq API
POST /proxy/urlscan         → URLScan.io search (có CORS header)
POST /proxy/virustotal/url  → VirusTotal URL scan
POST /proxy/virustotal/domain → VirusTotal domain info
POST /proxy/ipinfo          → IPInfo.io geo lookup
POST /proxy/dns             → DNS-over-HTTPS (Cloudflare)
POST /proxy/allorigins      → allorigins.win proxy (source scan)

Env vars
────────
ANTHROPIC_API_KEY     (Claude proxy)
GEMINI_API_KEY        (Gemini Flash proxy)
GROQ_API_KEY          (Groq proxy)
VIRUSTOTAL_API_KEY    (VirusTotal)
URLSCAN_API_KEY       (URLScan.io - optional)
IPINFO_TOKEN          (IPInfo.io - optional)
CORS_ORIGINS          comma-separated, default *
PORT                  default 8004
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from blockchain_verify import verify_on_chain
from consensus         import aggregate_consensus
from explainable_ai    import explain_prediction
from version_compare   import compare_versions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("utility-service")

# ── Env ───────────────────────────────────────────────────────────────────────
ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY",  "")
GEMINI_KEY     = os.getenv("GEMINI_API_KEY",      "")
GROQ_KEY       = os.getenv("GROQ_API_KEY",        "")
VT_KEY         = os.getenv("VIRUSTOTAL_API_KEY",  "")
URLSCAN_KEY    = os.getenv("URLSCAN_API_KEY",     "")
IPINFO_TOKEN   = os.getenv("IPINFO_TOKEN",        "")

TIMEOUT = httpx.Timeout(30.0, connect=5.0)

# ── Lifespan ──────────────────────────────────────────────────────────────────
_http: httpx.AsyncClient | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http
    _http = httpx.AsyncClient(
        timeout=TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": "AI-PROOF-Utility/2.0"},
    )
    log.info("utility-service v2 ready — AI:%s VT:%s URLScan:%s IPInfo:%s",
             bool(ANTHROPIC_KEY or GEMINI_KEY or GROQ_KEY),
             bool(VT_KEY), bool(URLSCAN_KEY), bool(IPINFO_TOKEN))
    yield
    await _http.aclose()

# ── App ───────────────────────────────────────────────────────────────────────
_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
app = FastAPI(title="AI-PROOF Utility Service", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def _err(req: Request, exc: Exception):
    log.exception("Unhandled: %s", exc)
    return JSONResponse(500, content={"detail": str(exc)})

# ── Schemas ───────────────────────────────────────────────────────────────────

class ConsensusRequest(BaseModel):
    models: list[dict]

class ExplainRequest(BaseModel):
    text:        str
    trust_score: int
    models:      list[dict] = []

class VersionCompareRequest(BaseModel):
    text_a: str
    text_b: str

class BlockchainVerifyRequest(BaseModel):
    content_hash: str
    block_id:     Optional[int] = None

# Proxy schemas
class AIProxyRequest(BaseModel):
    messages:   list[dict]
    max_tokens: int = 1000
    provider:   str = "auto"   # "claude" | "gemini" | "groq" | "auto"

class URLScanRequest(BaseModel):
    domain: str
    submit: bool = False
    url:    Optional[str] = None

class VTUrlRequest(BaseModel):
    url: str

class VTDomainRequest(BaseModel):
    domain: str

class IPInfoRequest(BaseModel):
    domain: str

class DNSRequest(BaseModel):
    domain: str
    type:   str = "A"

class OriginsRequest(BaseModel):
    url: str

# ═══════════════════════════════════════════════════════════════
#  ORIGINAL ROUTES (giữ nguyên)
# ═══════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {
        "status":   "ok",
        "service":  "utility",
        "version":  "2.0",
        "features": [
            "consensus", "explainable_ai", "version_compare", "blockchain_verify",
            "proxy/ai", "proxy/urlscan", "proxy/virustotal", "proxy/ipinfo", "proxy/dns",
        ],
        "keys": {
            "ai":         bool(ANTHROPIC_KEY or GEMINI_KEY or GROQ_KEY),
            "virustotal": bool(VT_KEY),
            "urlscan":    bool(URLSCAN_KEY),
            "ipinfo":     bool(IPINFO_TOKEN),
        },
    }

@app.post("/consensus")
async def consensus(req: ConsensusRequest):
    if not req.models:
        raise HTTPException(400, detail="models list is empty")
    return aggregate_consensus(req.models)

@app.post("/explain")
async def explain(req: ExplainRequest):
    if not req.text or len(req.text.strip()) < 5:
        raise HTTPException(400, detail="text must be at least 5 characters")
    return await explain_prediction(req.text, req.trust_score, req.models)

@app.post("/version/compare")
async def version_compare(req: VersionCompareRequest):
    if not req.text_a or not req.text_b:
        raise HTTPException(400, detail="text_a and text_b are required")
    return compare_versions(req.text_a, req.text_b)

@app.post("/blockchain/verify")
async def blockchain_verify(req: BlockchainVerifyRequest):
    if not req.content_hash:
        raise HTTPException(400, detail="content_hash is required")
    return await verify_on_chain(req.content_hash, req.block_id)

# ═══════════════════════════════════════════════════════════════
#  PROXY ROUTES (mới)
# ═══════════════════════════════════════════════════════════════

# ── /proxy/ai ─────────────────────────────────────────────────
@app.post("/proxy/ai")
async def proxy_ai(req: AIProxyRequest):
    """
    Proxy AI request — thử Claude → Gemini → Groq theo thứ tự.
    Frontend gửi messages[], backend gắn API key và forward.
    """
    if req.max_tokens > 1500:
        raise HTTPException(400, detail="max_tokens capped at 1500")

    errors = []

    # ── Claude ──
    if req.provider in ("auto", "claude") and ANTHROPIC_KEY:
        try:
            r = await _http.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key":         ANTHROPIC_KEY,
                    "anthropic-version": "2023-06-01",
                    "Content-Type":      "application/json",
                },
                json={
                    "model":      "claude-sonnet-4-20250514",
                    "max_tokens": req.max_tokens,
                    "messages":   req.messages,
                },
                timeout=25,
            )
            if r.is_success:
                data = r.json()
                return {
                    "provider": "claude",
                    "text":     data["content"][0]["text"],
                    "usage":    data.get("usage", {}),
                }
            errors.append(f"Claude HTTP {r.status_code}")
        except Exception as e:
            errors.append(f"Claude: {e}")

    # ── Gemini ──
    if req.provider in ("auto", "gemini") and GEMINI_KEY:
        try:
            # Convert messages to Gemini format
            parts = [{"text": m["content"] if isinstance(m["content"], str)
                              else m["content"][0].get("text", "")}
                     for m in req.messages if m.get("role") == "user"]
            r = await _http.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-2.0-flash:generateContent?key={GEMINI_KEY}",
                json={
                    "contents": [{"parts": parts}],
                    "generationConfig": {
                        "maxOutputTokens": req.max_tokens,
                        "temperature":     0.2,
                    },
                },
                timeout=25,
            )
            if r.is_success:
                data = r.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return {"provider": "gemini", "text": text}
            errors.append(f"Gemini HTTP {r.status_code}")
        except Exception as e:
            errors.append(f"Gemini: {e}")

    # ── Groq ──
    if req.provider in ("auto", "groq") and GROQ_KEY:
        try:
            r = await _http.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_KEY}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":       "llama-3.1-8b-instant",
                    "messages":    req.messages,
                    "max_tokens":  req.max_tokens,
                    "temperature": 0.2,
                },
                timeout=20,
            )
            if r.is_success:
                data = r.json()
                text = data["choices"][0]["message"]["content"]
                return {"provider": "groq", "text": text}
            errors.append(f"Groq HTTP {r.status_code}")
        except Exception as e:
            errors.append(f"Groq: {e}")

    raise HTTPException(503, detail=f"All AI providers failed: {'; '.join(errors)}")


# ── /proxy/dns ────────────────────────────────────────────────
@app.post("/proxy/dns")
async def proxy_dns(req: DNSRequest):
    """DNS-over-HTTPS via Cloudflare — hỗ trợ mọi TLD kể cả .vn"""
    domain = req.domain.lower().strip().lstrip("www.")
    dns_type = req.type.upper()
    errors = []

    for provider_url in [
        f"https://cloudflare-dns.com/dns-query?name={domain}&type={dns_type}",
        f"https://dns.google/resolve?name={domain}&type={dns_type}",
    ]:
        try:
            r = await _http.get(
                provider_url,
                headers={"Accept": "application/dns-json"},
                timeout=7,
            )
            if r.is_success:
                data = r.json()
                return {
                    "ok":      data.get("Status") == 0,
                    "status":  data.get("Status"),
                    "records": [a["data"] for a in data.get("Answer", [])],
                    "type":    dns_type,
                    "domain":  domain,
                }
        except Exception as e:
            errors.append(str(e))

    return {"ok": False, "records": [], "type": dns_type, "domain": domain, "errors": errors}


# ── /proxy/ipinfo ─────────────────────────────────────────────
@app.post("/proxy/ipinfo")
async def proxy_ipinfo(req: IPInfoRequest):
    """Resolve domain IP rồi gọi IPInfo — hỗ trợ .vn"""
    domain = req.domain.lower().strip().lstrip("www.")

    # Resolve IP qua DNS
    dns_resp = await proxy_dns(DNSRequest(domain=domain, type="A"))
    if not dns_resp.get("ok") or not dns_resp.get("records"):
        return {"ok": False, "domain": domain, "error": "DNS resolve failed"}

    ip = dns_resp["records"][0]
    url = f"https://ipinfo.io/{ip}/json"
    if IPINFO_TOKEN:
        url += f"?token={IPINFO_TOKEN}"

    try:
        r = await _http.get(url, timeout=8)
        if r.is_success:
            data = r.json()
            return {
                "ok":       True,
                "ip":       ip,
                "hostname": data.get("hostname", domain),
                "city":     data.get("city", ""),
                "region":   data.get("region", ""),
                "country":  data.get("country", ""),
                "org":      data.get("org", ""),
                "timezone": data.get("timezone", ""),
                "isVPN":    data.get("privacy", {}).get("vpn",   False),
                "isProxy":  data.get("privacy", {}).get("proxy", False),
                "isTor":    data.get("privacy", {}).get("tor",   False),
            }
    except Exception as e:
        log.warning("IPInfo failed: %s", e)

    return {"ok": False, "ip": ip, "domain": domain}


# ── /proxy/virustotal/url ─────────────────────────────────────
@app.post("/proxy/virustotal/url")
async def proxy_vt_url(req: VTUrlRequest):
    """Submit URL scan to VirusTotal và lấy kết quả."""
    if not VT_KEY:
        raise HTTPException(503, detail="VIRUSTOTAL_API_KEY not configured")

    hdrs = {"x-apikey": VT_KEY}
    try:
        # Submit
        r = await _http.post(
            "https://www.virustotal.com/api/v3/urls",
            headers={**hdrs, "Content-Type": "application/x-www-form-urlencoded"},
            content=f"url={req.url}".encode(),
            timeout=15,
        )
        if not r.is_success:
            raise HTTPException(r.status_code, detail=f"VT submit: {r.text[:200]}")

        analysis_id = r.json()["data"]["id"]

        # Poll kết quả (tối đa 3 lần, cách 3s)
        for _ in range(3):
            await asyncio.sleep(3)
            r2 = await _http.get(
                f"https://www.virustotal.com/api/v3/analyses/{analysis_id}",
                headers=hdrs, timeout=10,
            )
            if r2.is_success:
                stats = r2.json()["data"]["attributes"].get("stats", {})
                total = sum(stats.values())
                malicious  = stats.get("malicious",  0)
                suspicious = stats.get("suspicious", 0)
                return {
                    "ok":          True,
                    "malicious":   malicious,
                    "suspicious":  suspicious,
                    "harmless":    stats.get("harmless",   0),
                    "undetected":  stats.get("undetected", 0),
                    "totalEngines": total,
                    "threatScore": round((malicious + suspicious * 0.5) / max(1, total) * 100),
                    "source":      "VirusTotal",
                }
    except HTTPException:
        raise
    except Exception as e:
        log.warning("VT URL scan failed: %s", e)

    return {"ok": False, "error": "Scan did not complete"}


# ── /proxy/virustotal/domain ──────────────────────────────────
@app.post("/proxy/virustotal/domain")
async def proxy_vt_domain(req: VTDomainRequest):
    """Lấy thông tin domain từ VirusTotal — hỗ trợ .vn."""
    if not VT_KEY:
        raise HTTPException(503, detail="VIRUSTOTAL_API_KEY not configured")

    # Lấy registrable domain (baotintuc.vn, không phải subdomain)
    parts = req.domain.lower().strip().split(".")
    registrable = ".".join(parts[-2:]) if len(parts) >= 2 else req.domain

    try:
        r = await _http.get(
            f"https://www.virustotal.com/api/v3/domains/{registrable}",
            headers={"x-apikey": VT_KEY},
            timeout=12,
        )
        if r.is_success:
            attrs = r.json()["data"]["attributes"]
            stats = attrs.get("last_analysis_stats", {})
            cd    = attrs.get("creation_date")
            return {
                "ok":           True,
                "domain":       registrable,
                "reputation":   attrs.get("reputation", 0),
                "categories":   ", ".join(attrs.get("categories", {}).values()),
                "malicious":    stats.get("malicious",  0),
                "harmless":     stats.get("harmless",   0),
                "suspicious":   stats.get("suspicious", 0),
                "creationDate": _ts(cd) if cd else None,
                "registrar":    attrs.get("registrar", ""),
                "country":      attrs.get("country",   ""),
                "source":       "VirusTotal",
            }
        if r.status_code == 404:
            return {"ok": False, "domain": registrable, "error": "Domain not in VT database"}
        raise HTTPException(r.status_code, detail=f"VT domain: {r.text[:200]}")
    except HTTPException:
        raise
    except Exception as e:
        log.warning("VT domain failed: %s", e)
        return {"ok": False, "domain": registrable, "error": str(e)}


def _ts(unix_ts) -> str:
    """Convert Unix timestamp → ISO date string."""
    try:
        import datetime
        return datetime.datetime.utcfromtimestamp(int(unix_ts)).strftime("%Y-%m-%d")
    except Exception:
        return str(unix_ts)


# ── /proxy/urlscan ────────────────────────────────────────────
@app.post("/proxy/urlscan")
async def proxy_urlscan(req: URLScanRequest):
    """
    Search URLScan.io cho domain — không cần API key để search.
    Nếu submit=True và URLSCAN_KEY có sẵn → submit scan mới.
    """
    domain = req.domain.lower().strip().lstrip("www.")
    parts  = domain.split(".")
    registrable = ".".join(parts[-2:]) if len(parts) >= 2 else domain

    hdrs = {}
    if URLSCAN_KEY:
        hdrs["API-Key"] = URLSCAN_KEY

    # Submit scan mới nếu yêu cầu
    if req.submit and req.url and URLSCAN_KEY:
        try:
            r = await _http.post(
                "https://urlscan.io/api/v1/scan/",
                headers={**hdrs, "Content-Type": "application/json"},
                json={"url": req.url, "visibility": "public"},
                timeout=15,
            )
            if r.is_success:
                uuid = r.json().get("uuid")
                # Đợi và lấy kết quả (30s)
                await asyncio.sleep(25)
                r2 = await _http.get(
                    f"https://urlscan.io/api/v1/result/{uuid}/",
                    headers=hdrs, timeout=15,
                )
                if r2.is_success:
                    d = r2.json()
                    v = d.get("verdicts", {}).get("overall", {})
                    return {
                        "ok":          True,
                        "domain":      registrable,
                        "malicious":   v.get("malicious", False),
                        "score":       v.get("score", 0),
                        "tags":        v.get("tags", []),
                        "screenshot":  d.get("task", {}).get("screenshotURL"),
                        "reportURL":   f"https://urlscan.io/result/{uuid}/",
                        "server":      d.get("page", {}).get("server", ""),
                        "country":     d.get("page", {}).get("country", ""),
                        "source":      "URLScan.io (submit)",
                    }
        except Exception as e:
            log.warning("URLScan submit failed: %s", e)

    # Search lịch sử scan
    try:
        r = await _http.get(
            f"https://urlscan.io/api/v1/search/?q=domain:{registrable}&size=3&sort=date",
            headers=hdrs,
            timeout=10,
        )
        if r.is_success:
            results = r.json().get("results", [])
            if not results:
                return {"ok": True, "domain": registrable, "found": False, "results": []}
            latest = results[0]
            v = latest.get("verdicts", {}).get("overall", {})
            return {
                "ok":         True,
                "domain":     registrable,
                "found":      True,
                "malicious":  v.get("malicious",    False),
                "score":      v.get("score",         0),
                "tags":       v.get("tags",           []),
                "screenshot": latest.get("screenshot"),
                "reportURL":  f"https://urlscan.io/result/{latest['task']['uuid']}/",
                "server":     latest.get("page", {}).get("server",  ""),
                "country":    latest.get("page", {}).get("country", ""),
                "lastScanned":latest.get("task", {}).get("time"),
                "source":     "URLScan.io",
            }
        # 403 = Rate limited hoặc blocked
        if r.status_code in (403, 429):
            log.warning("URLScan rate limited: %s", r.status_code)
            return {"ok": False, "domain": registrable, "error": f"Rate limited ({r.status_code})"}
    except Exception as e:
        log.warning("URLScan search failed: %s", e)

    return {"ok": False, "domain": registrable, "error": "URLScan unavailable"}


# ── /proxy/allorigins ─────────────────────────────────────────
@app.post("/proxy/allorigins")
async def proxy_allorigins(req: OriginsRequest):
    """
    Fetch HTML source của URL qua allorigins.win.
    Trả về html content để frontend scan (gambling, ads, malware).
    """
    try:
        r = await _http.get(
            f"https://api.allorigins.win/get?url={req.url}",
            timeout=14,
        )
        if r.is_success:
            data = r.json()
            html = data.get("contents", "")
            return {
                "ok":     True,
                "html":   html[:80000],   # giới hạn 80KB
                "length": len(html),
            }
    except Exception as e:
        log.warning("allorigins failed for %s: %s", req.url, e)

    return {"ok": False, "html": "", "error": "Could not fetch source"}


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8004)),
        reload=False,
    )
