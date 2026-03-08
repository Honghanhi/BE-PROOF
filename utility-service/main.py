"""
Utility Service — FastAPI wrapper
Import 4 file gốc, expose thành HTTP routes.
"""
import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from blockchain_verify import verify_on_chain
from consensus         import aggregate_consensus
from explainable_ai    import explain_prediction
from version_compare   import compare_versions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("utility-service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("utility-service ready")
    yield


app = FastAPI(title="Utility Service", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"])


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


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":   "ok",
        "service":  "utility",
        "features": ["consensus", "explainable_ai", "version_compare", "blockchain_verify"],
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


if __name__ == "__main__":
    uvicorn.run("main:app",
                host="0.0.0.0",
                port=int(os.getenv("PORT", 8004)),
                reload=False)