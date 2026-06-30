"""One-shot ingest, no HTTP. Useful for breakpoints and inspection.

Usage:
    python -m scripts.ingest --limit 5 --force
"""

import argparse
import asyncio

from app.config import AppSettings
from app.ghost_client import GhostClient
from app.ingest_service import IngestService
from app.models import IngestRequest
from app.ollama_client import OllamaClient
from app.text_processor import TextProcessor
from app.vector_store import VectorStore


async def _run(limit: int | None, force: bool) -> None:
    settings = AppSettings()
    store = VectorStore(settings)
    await store.open()
    ollama = OllamaClient(settings)
    ghost = GhostClient(settings)
    try:
        service = IngestService(ghost, TextProcessor(settings), ollama, store)
        result = await service.run(IngestRequest(limit=limit, force=force))
        print(f"Processed {len(result.processed)} posts, skipped {len(result.skipped)}.")
        for p in result.processed:
            print(f"  + {p.slug}  ({p.chunks} chunks)")
        for s in result.skipped:
            print(f"  = {s}  (unchanged)")
    finally:
        await ghost.aclose()
        await ollama.aclose()
        await store.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    asyncio.run(_run(args.limit, args.force))


if __name__ == "__main__":
    main()
