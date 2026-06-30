from datetime import datetime

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    limit: int | None = None
    force: bool = False


class ProcessedPost(BaseModel):
    slug: str
    chunks: int


class IngestResponse(BaseModel):
    processed: list[ProcessedPost]
    skipped: list[str]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = None


class SourceRef(BaseModel):
    title: str
    url: str | None
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceRef]


class GhostPost(BaseModel):
    id: str
    slug: str
    title: str
    url: str | None = None
    html: str = ""
    published_at: datetime | None = None
    updated_at: datetime | None = None


class RetrievedChunk(BaseModel):
    post_id: str
    content: str
    kind: str
    title: str
    url: str | None
    score: float
    # Carried through retrieval so MMR (and any other diversity / clustering
    # step) can run without a second round-trip to the DB or Ollama.
    embedding: list[float] | None = None
