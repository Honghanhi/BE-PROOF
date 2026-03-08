"""
Version Compare Service
Computes a semantic diff between two text versions.
"""

import difflib
from typing import Any


def compare_versions(text_a: str, text_b: str) -> dict[str, Any]:
    """
    Compute word-level diff between two text versions.
    Returns structured diff with addition/deletion spans.
    """
    words_a = text_a.split()
    words_b = text_b.split()

    matcher = difflib.SequenceMatcher(None, words_a, words_b)
    opcodes = matcher.get_opcodes()

    spans      = []
    additions  = 0
    deletions  = 0
    similarity = matcher.ratio()

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            spans.append({
                "type":  "equal",
                "words": words_a[i1:i2],
            })
        elif tag == "replace":
            spans.append({
                "type":    "delete",
                "words":   words_a[i1:i2],
            })
            spans.append({
                "type":    "insert",
                "words":   words_b[j1:j2],
            })
            deletions  += i2 - i1
            additions  += j2 - j1
        elif tag == "delete":
            spans.append({
                "type":  "delete",
                "words": words_a[i1:i2],
            })
            deletions += i2 - i1
        elif tag == "insert":
            spans.append({
                "type":  "insert",
                "words": words_b[j1:j2],
            })
            additions += j2 - j1

    return {
        "spans":      spans,
        "additions":  additions,
        "deletions":  deletions,
        "similarity": round(similarity * 100, 1),
        "changed":    additions > 0 or deletions > 0,
    }