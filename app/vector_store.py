from collections import defaultdict
from datetime import datetime

import psycopg
from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool

from app.config import AppSettings
from app.models import GhostPost, RetrievedChunk


class VectorStore:
    """pgvector data access, mirroring the C# VectorStore. Holds one async
    connection pool. SearchAsync supports vector, keyword, or RRF-fused hybrid."""

    def __init__(self, settings: AppSettings):
        self._settings = settings
        self._pool: AsyncConnectionPool | None = None

    async def open(self) -> None:
        if self._pool is not None:
            return

        async def _configure(conn: psycopg.AsyncConnection) -> None:
            await register_vector_async(conn)

        self._pool = AsyncConnectionPool(
            self._settings.database_url,
            min_size=1,
            max_size=4,
            kwargs={"autocommit": False},
            configure=_configure,
            open=False,
        )
        await self._pool.open()

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def post_needs_ingest(self, post_id: str, updated_at: datetime | None) -> bool:
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute("SELECT updated_at FROM posts WHERE id = %s", (post_id,))
            row = await cur.fetchone()
        if row is None or row[0] is None:
            return True
        return updated_at is not None and updated_at > row[0]

    async def upsert_post(self, post: GhostPost) -> None:
        async with self._connection() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM chunks WHERE post_id = %s", (post.id,))
                await conn.execute(
                    """
                    INSERT INTO posts (id, slug, title, url, published_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        slug = EXCLUDED.slug,
                        title = EXCLUDED.title,
                        url = EXCLUDED.url,
                        published_at = EXCLUDED.published_at,
                        updated_at = EXCLUDED.updated_at,
                        ingested_at = now()
                    """,
                    (post.id, post.slug, post.title, post.url, post.published_at, post.updated_at),
                )

    async def insert_chunk(
        self,
        post_id: str,
        index: int,
        content: str,
        kind: str,
        embedding: list[float],
    ) -> None:
        async with self._connection() as conn:
            await conn.execute(
                """
                INSERT INTO chunks (post_id, chunk_index, content, kind, embedding)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (post_id, index, content, kind, embedding),
            )
            await conn.commit()

    async def vector_search(
        self,
        query_embedding: list[float],
        k: int,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        sql = """
            SELECT c.post_id, c.content, c.kind, p.title, p.url,
                   1 - (c.embedding <=> %s::vector) AS score,
                   c.embedding
            FROM chunks c
            JOIN posts p ON p.id = c.post_id
            WHERE 1 - (c.embedding <=> %s::vector) >= %s
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        """
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                sql, (query_embedding, query_embedding, min_score, query_embedding, k)
            )
            rows = await cur.fetchall()
        return [_row_to_chunk(r, embedding_idx=6) for r in rows]

    async def keyword_search(self, query_text: str, k: int) -> list[RetrievedChunk]:
        sql = """
            SELECT c.post_id, c.content, c.kind, p.title, p.url,
                   ts_rank_cd(c.content_tsv, plainto_tsquery('english', %s)) AS score,
                   c.embedding
            FROM chunks c
            JOIN posts p ON p.id = c.post_id
            WHERE c.content_tsv @@ plainto_tsquery('english', %s)
            ORDER BY score DESC
            LIMIT %s
        """
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(sql, (query_text, query_text, k))
            rows = await cur.fetchall()
        return [_row_to_chunk(r, embedding_idx=6) for r in rows]

    async def hybrid_search(
        self,
        query_embedding: list[float],
        query_text: str,
        k: int,
        rrf_k: int = 60,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        """Vector + BM25 fused with Reciprocal Rank Fusion: score = sum(1/(rrf_k+rank))."""
        vector_hits = await self.vector_search(query_embedding, k, min_score=min_score)
        keyword_hits = await self.keyword_search(query_text, k)

        scores: dict[tuple[str, str], float] = defaultdict(float)
        sample: dict[tuple[str, str], RetrievedChunk] = {}

        for results in (vector_hits, keyword_hits):
            for rank, chunk in enumerate(results, start=1):
                key = (chunk.post_id, chunk.content)
                scores[key] += 1.0 / (rrf_k + rank)
                sample.setdefault(key, chunk)

        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [sample[key].model_copy(update={"score": score}) for key, score in ordered[:k]]

    async def chunks_for_post(self, post_id: str) -> list[tuple[int, str, str]]:
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT chunk_index, content, kind FROM chunks WHERE post_id = %s ORDER BY chunk_index",
                (post_id,),
            )
            return [(r[0], r[1], r[2]) for r in await cur.fetchall()]

    def _connection(self):
        assert self._pool is not None, "VectorStore.open() must be called first"
        return self._pool.connection()


def _row_to_chunk(row, *, embedding_idx: int) -> RetrievedChunk:
    """Build a RetrievedChunk from a SELECT row. The embedding column comes
    back as a numpy array (via pgvector); coerce to plain list[float] so the
    object round-trips cleanly through Pydantic and JSON."""
    raw_embedding = row[embedding_idx]
    embedding = [float(x) for x in raw_embedding] if raw_embedding is not None else None
    return RetrievedChunk(
        post_id=row[0],
        content=row[1],
        kind=row[2],
        title=row[3],
        url=row[4],
        score=float(row[5]),
        embedding=embedding,
    )
