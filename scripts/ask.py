"""One-shot question, no HTTP. Useful for breakpoints and tuning retrieval.

Usage:
    python -m scripts.ask "What did NVDA report in Q3?" --top-k 5 --debug
"""

import argparse
import asyncio

from app.config import AppSettings
from app.models import QueryRequest
from app.ollama_client import OllamaClient
from app.query_service import QueryService
from app.reranker import CrossEncoderReranker
from app.vector_store import VectorStore


async def _run(question: str, top_k: int | None, debug: bool) -> None:
    settings = AppSettings()
    store = VectorStore(settings)
    await store.open()
    ollama = OllamaClient(settings)
    reranker = CrossEncoderReranker(settings.reranker_model) if settings.use_reranker else None
    try:
        service = QueryService(ollama, store, settings, reranker)

        if debug:
            chunks = await service.retrieve(question, top_k or settings.top_k)
            print(f"\n--- Top {len(chunks)} chunks ---")
            for i, c in enumerate(chunks, 1):
                print(f"{i}. [{c.score:.3f}] {c.title}")
                print(f"   {c.content[:200].strip()}...\n")
            return

        result = await service.ask(QueryRequest(question=question, top_k=top_k))
        print("\n--- Answer ---")
        print(result.answer)
        print("\n--- Sources ---")
        for s in result.sources:
            print(f"  [{s.score}] {s.title}  {s.url or ''}")
    finally:
        await ollama.aclose()
        await store.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print retrieved chunks + scores, skip the LLM step.",
    )
    args = parser.parse_args()
    asyncio.run(_run(args.question, args.top_k, args.debug))


if __name__ == "__main__":
    main()
