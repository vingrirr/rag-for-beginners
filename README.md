# rag-for-beginners

A Python RAG service that ingests posts from a Ghost site, embeds them with a
local Ollama model, and stores chunks in Postgres + pgvector. Mirrors the
architecture of the parallel C# `CestrianRag` app so the two stay debuggable
side-by-side.

The original numbered tutorial scripts (1–13) now live under `learning/` for
reference.

## Layout

```
app/                    FastAPI service mirroring the C# services
  config.py             pydantic-settings AppSettings
  ollama_client.py      /api/embeddings + /api/generate
  ghost_client.py       Ghost Admin API (JWT-signed)
  text_processor.py     HTML -> block text + word-based chunking with overlap
  vector_store.py       psycopg + pgvector; vector / keyword / hybrid search
  reranker.py           Optional local cross-encoder rerank
  ingest_service.py     fetch -> skip-unchanged -> chunk -> embed -> store
  query_service.py      embed -> retrieve -> (rerank) -> generate
  main.py               FastAPI endpoints
scripts/                CLI wrappers around the services (set breakpoints here)
migrations/             SQL run by Postgres on first boot
tests/                  pytest (start with text_processor)
learning/               Original numbered tutorial scripts
```

## Run it locally

```bash
cp .env.example .env                # fill GHOST_URL + GHOST_ADMIN_API_KEY

docker compose up -d                # pgvector + ollama (schema auto-applies)

docker exec -it $(docker ps -qf name=ollama) \
    ollama pull nomic-embed-text llama3.1:8b

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000
```

Then:

```bash
# Ingest the 10 most recently updated posts.
curl -X POST localhost:8000/ingest -H 'content-type: application/json' \
    -d '{"limit": 10}'

# Ask a question.
curl -X POST localhost:8000/query -H 'content-type: application/json' \
    -d '{"question": "What did NVDA report in Q3?"}'

# Inspect retrieval without the LLM.
curl 'localhost:8000/debug/search?q=NVDA+Q3+revenue&k=5'
```

## Debugging without HTTP

```bash
python -m scripts.ingest --limit 5 --force
python -m scripts.ask "What did NVDA report in Q3?" --debug    # chunks + scores only
python -m scripts.ask "What did NVDA report in Q3?"            # full answer
```

`--debug` skips the LLM step entirely so you can tell whether a bad answer is
a retrieval problem or a generation problem.

## Retrieval pipeline

`QueryService.retrieve` is a composable pipeline; toggles in `.env`:

| Setting | What it does |
| --- | --- |
| `USE_HYBRID=true` | Vector + BM25 (Postgres full-text) fused with RRF |
| `USE_HYBRID=false` | Pure cosine-similarity vector search |
| `USE_RERANKER=true` | Add a local cross-encoder rerank after retrieval |
| `RETRIEVE_K=20` | Candidates fetched before rerank |
| `TOP_K=5` | Chunks finally passed to the LLM |

For "wrong articles" retrieval bugs, hybrid + reranker is the biggest lever.

## Vector dimension

The `chunks.embedding` column is `vector(768)` to match `nomic-embed-text`.
If you switch to `mxbai-embed-large` (1024-d) or any other embed model, edit
`migrations/001_init.sql` and re-run against an empty database.

## Tests

```bash
pytest tests/
```

`tests/test_text_processor.py` covers HTML extraction and chunking, the
module most likely to drift from the C# port.
