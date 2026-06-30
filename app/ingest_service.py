import structlog

from app.ghost_client import GhostClient
from app.models import IngestRequest, IngestResponse, ProcessedPost
from app.ollama_client import OllamaClient
from app.text_processor import TextProcessor
from app.vector_store import VectorStore


log = structlog.get_logger()


class IngestService:
    """Orchestrates ingest: fetch from Ghost, skip unchanged posts, extract
    text, chunk, embed, store. Mirrors the C# IngestService body line-for-line."""

    def __init__(
        self,
        ghost: GhostClient,
        text: TextProcessor,
        ollama: OllamaClient,
        store: VectorStore,
    ):
        self._ghost = ghost
        self._text = text
        self._ollama = ollama
        self._store = store

    async def run(self, req: IngestRequest) -> IngestResponse:
        posts = await self._ghost.fetch_posts(req.limit)
        log.info("ingest.fetched", count=len(posts))

        processed: list[ProcessedPost] = []
        skipped: list[str] = []

        for post in posts:
            if not req.force and not await self._store.post_needs_ingest(post.id, post.updated_at):
                skipped.append(post.slug)
                continue

            plain = self._text.html_to_text(post.html)
            chunks = self._text.chunk_text(plain)

            await self._store.upsert_post(post)

            for i, chunk in enumerate(chunks):
                embedding = await self._ollama.embed(chunk)
                await self._store.insert_chunk(post.id, i, chunk, "text", embedding)

            log.info("ingest.post", slug=post.slug, chunks=len(chunks))
            processed.append(ProcessedPost(slug=post.slug, chunks=len(chunks)))

        return IngestResponse(processed=processed, skipped=skipped)
