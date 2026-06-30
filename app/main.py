from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException

from app.config import AppSettings
from app.ghost_client import GhostClient
from app.ingest_service import IngestService
from app.models import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
)
from app.ollama_client import OllamaClient
from app.query_service import QueryService
from app.reranker import CrossEncoderReranker
from app.text_processor import TextProcessor
from app.vector_store import VectorStore


log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = AppSettings()
    store = VectorStore(settings)
    await store.open()
    ollama = OllamaClient(settings)
    ghost = GhostClient(settings)
    text = TextProcessor(settings)
    reranker = CrossEncoderReranker(settings.reranker_model) if settings.use_reranker else None

    app.state.settings = settings
    app.state.ingest = IngestService(ghost, text, ollama, store)
    app.state.query = QueryService(ollama, store, settings, reranker)
    app.state.store = store
    app.state.ollama = ollama
    app.state.ghost = ghost
    try:
        yield
    finally:
        await store.close()
        await ollama.aclose()
        await ghost.aclose()


app = FastAPI(title="Cestrian RAG (Python)", lifespan=lifespan)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest) -> IngestResponse:
    return await app.state.ingest.run(req)


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    return await app.state.query.ask(req)


@app.get("/debug/search")
async def debug_search(q: str, k: int | None = None) -> list[RetrievedChunk]:
    top_k = k or app.state.settings.top_k
    return await app.state.query.retrieve(q, top_k)


@app.get("/debug/chunks/{post_id}")
async def debug_chunks(post_id: str) -> list[dict]:
    rows = await app.state.store.chunks_for_post(post_id)
    if not rows:
        raise HTTPException(status_code=404, detail="post not found or has no chunks")
    return [{"chunk_index": i, "kind": kind, "content": content} for i, content, kind in rows]


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True}
