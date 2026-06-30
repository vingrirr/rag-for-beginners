from typing import Protocol

from app.models import RetrievedChunk


class Reranker(Protocol):
    async def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]: ...


class CrossEncoderReranker:
    """Local cross-encoder reranker (BAAI/bge-reranker-base by default).

    Loaded lazily so the import cost is paid only when USE_RERANKER=true.
    """

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        return self._model

    async def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return chunks
        model = self._load()
        pairs = [(query, c.content) for c in chunks]
        scores = model.predict(pairs)
        scored = sorted(
            (c.model_copy(update={"score": float(s)}) for c, s in zip(chunks, scores)),
            key=lambda c: c.score,
            reverse=True,
        )
        return scored
