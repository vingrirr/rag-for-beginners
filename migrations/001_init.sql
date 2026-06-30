-- pgvector extension + schema for the Ghost RAG app.
--
-- IMPORTANT: the vector(...) dimension below MUST match the output dimension
-- of EMBED_MODEL. Defaults:
--   nomic-embed-text   -> 768
--   mxbai-embed-large  -> 1024
-- If you change the embed model, re-run this migration against an empty DB.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS posts (
    id           TEXT PRIMARY KEY,
    slug         TEXT NOT NULL,
    title        TEXT NOT NULL,
    url          TEXT,
    published_at TIMESTAMPTZ,
    updated_at   TIMESTAMPTZ,
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    post_id     TEXT NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    chunk_index INT  NOT NULL,
    content     TEXT NOT NULL,
    kind        TEXT NOT NULL DEFAULT 'text',
    embedding   vector(768) NOT NULL,
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_post_id_idx ON chunks (post_id);

-- HNSW index for fast cosine similarity. Build params are conservative defaults.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- GIN index for BM25-style full-text retrieval (hybrid search).
CREATE INDEX IF NOT EXISTS chunks_content_tsv_gin
    ON chunks USING gin (content_tsv);
