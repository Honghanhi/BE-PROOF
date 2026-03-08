"""
Blockchain Verification Service
Verifies content hashes against the stored proof chain.
"""

import asyncio
import hashlib
import json
from typing import Any, Optional


async def verify_on_chain(content_hash: str, block_id: Optional[int] = None) -> dict[str, Any]:
    """
    Verify that a content hash exists in the blockchain.
    In production: query a local SQLite/PostgreSQL ledger or external chain.
    """
    await asyncio.sleep(0.05)

    # Mock: treat any well-formed sha256 hash as verifiable
    is_valid_hash = len(content_hash) == 64 and all(c in "0123456789abcdef" for c in content_hash.lower())

    if not is_valid_hash:
        return {
            "verified": False,
            "reason":   "Invalid hash format",
            "blockId":  None,
        }

    # Simulate block lookup
    found = True  # In production: DB.query(hash == content_hash)

    return {
        "verified":    found,
        "blockId":     block_id or 1,
        "contentHash": content_hash,
        "merkleProof": _mock_merkle_proof(content_hash),
        "timestamp":   "2024-01-01T00:00:00Z",
    }


def _mock_merkle_proof(leaf: str) -> list[str]:
    """Generate a mock Merkle proof path."""
    h = leaf
    proof = []
    for i in range(4):
        sibling = hashlib.sha256(f"{h}{i}".encode()).hexdigest()
        proof.append(sibling)
        h = hashlib.sha256((h + sibling).encode()).hexdigest()
    return proof