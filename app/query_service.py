import structlog

from app.config import AppSettings
from app.models import QueryRequest, QueryResponse, RetrievedChunk, SourceRef
from app.ollama_client import OllamaClient
from app.reranker import Reranker
from app.vector_store import VectorStore


log = structlog.get_logger()


class QueryService:
    """Answer a question through a composable retrieval pipeline:
      embed -> retrieve (hybrid or vector) -> optional rerank -> generate.

    Each stage is a single method so a /debug/search endpoint can stop after
    retrieval and a future change can swap in a different reranker."""

    def __init__(
        self,
        ollama: OllamaClient,
        store: VectorStore,
        settings: AppSettings,
        reranker: Reranker | None = None,
    ):
        self._ollama = ollama
        self._store = store
        self._settings = settings
        self._reranker = reranker

    async def retrieve(self, question: str, top_k: int) -> list[RetrievedChunk]:
        embedding = await self._ollama.embed(question)
        retrieve_k = max(self._settings.retrieve_k, top_k)

        if self._settings.use_hybrid:
            candidates = await self._store.hybrid_search(embedding, question, retrieve_k)
        else:
            candidates = await self._store.vector_search(embedding, retrieve_k)

        if self._reranker is not None:
            candidates = await self._reranker.rerank(question, candidates)

        log.info(
            "query.retrieve",
            hybrid=self._settings.use_hybrid,
            reranked=self._reranker is not None,
            candidates=len(candidates),
        )
        return candidates[:top_k]

    async def ask(self, req: QueryRequest) -> QueryResponse:
        top_k = req.top_k or self._settings.top_k
        chunks = await self.retrieve(req.question, top_k)

        context_blocks = []
        sources: list[SourceRef] = []
        for chunk in chunks:
            context_blocks.append(f"[Source: {chunk.title}]\n{chunk.content}\n\n---\n")
            sources.append(SourceRef(title=chunk.title, url=chunk.url, score=round(chunk.score, 3)))

        prompt = (
            "You are answering questions about Cestrian Capital Research articles. "
            "Use ONLY the sources below. If the answer isn't in them, say so. "
            "Cite the article title(s) you used.\n\n"
            f"SOURCES:\n{''.join(context_blocks)}\n\n"
            f"QUESTION: {req.question}\n\nANSWER:"
        )

        answer = await self._ollama.chat(prompt)
        return QueryResponse(answer=answer, sources=sources)
