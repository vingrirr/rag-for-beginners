from app.mmr import mmr_rerank
from app.models import RetrievedChunk


def _chunk(post_id: str, score: float, embedding: list[float]) -> RetrievedChunk:
    return RetrievedChunk(
        post_id=post_id,
        content=f"content of {post_id}",
        kind="text",
        title=post_id,
        url=None,
        score=score,
        embedding=embedding,
    )


def test_mmr_empty():
    assert mmr_rerank([], k=5) == []


def test_mmr_k_zero():
    chunks = [_chunk("a", 1.0, [1.0, 0.0])]
    assert mmr_rerank(chunks, k=0) == []


def test_mmr_lambda_one_keeps_relevance_order():
    chunks = [
        _chunk("a", 0.9, [1.0, 0.0]),
        _chunk("b", 0.5, [1.0, 0.0]),   # identical embedding to a
        _chunk("c", 0.7, [0.0, 1.0]),
    ]
    out = mmr_rerank(chunks, k=3, lambda_=1.0)
    assert [c.post_id for c in out] == ["a", "c", "b"]


def test_mmr_lambda_zero_maximizes_diversity():
    chunks = [
        _chunk("a", 0.9, [1.0, 0.0]),
        _chunk("b", 0.85, [1.0, 0.0]),  # near-duplicate of a
        _chunk("c", 0.4, [0.0, 1.0]),   # orthogonal
    ]
    out = mmr_rerank(chunks, k=2, lambda_=0.0)
    assert [c.post_id for c in out[:2]] == ["a", "c"]


def test_mmr_falls_back_when_no_embeddings():
    chunks = [
        RetrievedChunk(post_id="a", content="x", kind="text", title="a", url=None, score=0.3),
        RetrievedChunk(post_id="b", content="y", kind="text", title="b", url=None, score=0.9),
    ]
    out = mmr_rerank(chunks, k=2, lambda_=0.5)
    assert [c.post_id for c in out] == ["b", "a"]
