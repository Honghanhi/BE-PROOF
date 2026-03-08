"""
Consensus Aggregation Service
Computes weighted multi-model trust score.
"""

from typing import Any

MODEL_WEIGHTS = {
    "gpt-detector":   0.25,
    "roberta-base":   0.20,
    "radar":          0.20,
    "fake-news-bert": 0.20,
    "grover":         0.15,
}


def aggregate_consensus(models: list[dict]) -> dict[str, Any]:
    """
    Compute weighted consensus trust score from model results.
    """
    if not models:
        return {"trust_score": 50, "agreement": 0, "weighted_scores": []}

    weighted_sum  = 0.0
    total_weight  = 0.0
    weighted_list = []

    for m in models:
        model_id = m.get("modelId", "")
        weight   = MODEL_WEIGHTS.get(model_id, 0.20)
        conf     = m.get("confidence", 0.80)
        eff_w    = weight * conf
        weighted_sum  += m["score"] * eff_w
        total_weight  += eff_w
        weighted_list.append({**m, "effectiveWeight": round(eff_w, 4)})

    trust_score = round(weighted_sum / total_weight) if total_weight else 50

    # Agreement metric: inverse of std deviation
    mean      = sum(m["score"] for m in models) / len(models)
    variance  = sum((m["score"] - mean) ** 2 for m in models) / len(models)
    std_dev   = variance ** 0.5
    agreement = max(0, min(100, round(100 - std_dev)))

    return {
        "trust_score":     trust_score,
        "agreement":       agreement,
        "weighted_scores": weighted_list,
    }