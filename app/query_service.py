from collections import defaultdict

import structlog

from app.config import AppSettings
from app.mmr import mmr_rerank
from app.models import QueryRequest, QueryResponse, RetrievedChunk, SourceRef
from app.ollama_client import OllamaClient
from app.reranker import Reranker
from app.vector_store import VectorStore


log = structlog.get_logger()


class QueryService:
    """Composable retrieval pipeline:

        embed -> [multi-query expand] -> hybrid/vector retrieve
              -> [min-score filter is applied inside vector_search]
              -> [cross-encoder rerank]
              -> [MMR diversity]
              -> top-k -> generate
    """

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
        retrieve_k = max(self._settings.retrieve_k, top_k)

        queries = [question]
        if self._settings.use_multi_query:
            try:
                queries += await self._expand_queries(question, self._settings.multi_query_count)
            except Exception as exc:
                log.warning("query.multi_query_failed", error=str(exc))

        result_lists: list[list[RetrievedChunk]] = []
        for q in queries:
            embedding = await self._ollama.embed(q)
            if self._settings.use_hybrid:
                hits = await self._store.hybrid_search(
                    embedding, q, retrieve_k, min_score=self._settings.min_score
                )
            else:
                hits = await self._store.vector_search(
                    embedding, retrieve_k, min_score=self._settings.min_score
                )
            result_lists.append(hits)

        candidates = (
            result_lists[0] if len(result_lists) == 1 else _rrf_fuse(result_lists, retrieve_k)
        )

        if self._reranker is not None:
            candidates = await self._reranker.rerank(question, candidates)

        if self._settings.use_mmr:
            candidates = mmr_rerank(candidates, top_k, self._settings.mmr_lambda)

        log.info(
            "query.retrieve",
            queries=len(queries),
            hybrid=self._settings.use_hybrid,
            reranked=self._reranker is not None,
            mmr=self._settings.use_mmr,
            min_score=self._settings.min_score,
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

    async def _expand_queries(self, original: str, count: int) -> list[str]:
        prompt = (
            f"Rewrite the question below as {count} alternative search queries that "
            "approach the same information need from different angles. Use different "
            "phrasings, synonyms, and entity names where helpful.\n\n"
            'Return ONLY a JSON object of the form: {"queries": ["...", "...", ...]} '
            f"with exactly {count} strings.\n\n"
            f"Question: {original}"
        )
        body = await self._ollama.chat_json(prompt)
        raw = body.get("queries") if isinstance(body, dict) else None
        if not isinstance(raw, list):
            return []
        return [q for q in raw if isinstance(q, str) and q.strip()][:count]


def _rrf_fuse(result_lists: list[list[RetrievedChunk]], k: int, rrf_k: int = 60) -> list[RetrievedChunk]:
    """RRF over multiple ranked lists. Keeps the first encountered chunk
    object (with its embedding) and overwrites the score with the fused one."""
    scores: dict[tuple[str, str], float] = defaultdict(float)
    sample: dict[tuple[str, str], RetrievedChunk] = {}
    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            key = (chunk.post_id, chunk.content)
            scores[key] += 1.0 / (rrf_k + rank)
            sample.setdefault(key, chunk)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [sample[key].model_copy(update={"score": score}) for key, score in ordered[:k]]
