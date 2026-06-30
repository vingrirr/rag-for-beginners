"""Maximum Marginal Relevance.

Greedy selection that trades off relevance (each chunk's prior score) against
diversity (max cosine similarity to chunks already chosen). Lambda controls
the mix: 1.0 → pure relevance, 0.0 → max diversity. Defaults to 0.5.
"""

import math
from typing import Sequence

from app.models import RetrievedChunk


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def mmr_rerank(
    chunks: list[RetrievedChunk],
    k: int,
    lambda_: float = 0.5,
) -> list[RetrievedChunk]:
    """Return up to k chunks ordered by MMR.

    Uses chunk.score as the relevance signal (normalized to [0, 1] across the
    candidate set so the term is comparable to cosine on the diversity side).
    Chunks missing an embedding are passed through in score order without
    diversity penalty.
    """
    if k <= 0 or not chunks:
        return []

    with_emb = [c for c in chunks if c.embedding]
    without_emb = [c for c in chunks if not c.embedding]

    if not with_emb:
        return sorted(chunks, key=lambda c: c.score, reverse=True)[:k]

    scores = [c.score for c in with_emb]
    lo, hi = min(scores), max(scores)
    span = (hi - lo) or 1.0
    relevance = {id(c): (c.score - lo) / span for c in with_emb}

    selected: list[RetrievedChunk] = []
    remaining = list(with_emb)

    while remaining and len(selected) < k:
        best = remaining[0]
        best_score = -math.inf
        for c in remaining:
            div = max(
                (_cosine(c.embedding or [], s.embedding or []) for s in selected),
                default=0.0,
            )
            mmr_score = lambda_ * relevance[id(c)] - (1.0 - lambda_) * div
            if mmr_score > best_score:
                best_score = mmr_score
                best = c
        selected.append(best)
        remaining.remove(best)

    if len(selected) < k:
        selected.extend(sorted(without_emb, key=lambda c: c.score, reverse=True)[: k - len(selected)])
    return selected
