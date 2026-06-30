import json

import httpx

from app.config import AppSettings


class OllamaClient:
    """Thin wrapper over the Ollama HTTP API. Mirrors the C# OllamaClient:
    /api/embeddings for chunks + queries, /api/generate for grounded answers."""

    def __init__(self, settings: AppSettings, client: httpx.AsyncClient | None = None):
        self._settings = settings
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(120.0))

    async def embed(self, text: str) -> list[float]:
        resp = await self._client.post(
            f"{self._settings.ollama_base_url}/api/embeddings",
            json={"model": self._settings.embed_model, "prompt": text},
        )
        resp.raise_for_status()
        body = resp.json()
        embedding = body.get("embedding")
        if not embedding:
            raise RuntimeError("Ollama returned no embedding")
        return embedding

    async def chat(self, prompt: str) -> str:
        resp = await self._client.post(
            f"{self._settings.ollama_base_url}/api/generate",
            json={
                "model": self._settings.chat_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1},
            },
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    async def chat_json(self, prompt: str) -> dict:
        """Generate with Ollama's JSON mode; returns a parsed dict.
        The caller must include the schema/keys it expects in the prompt."""
        resp = await self._client.post(
            f"{self._settings.ollama_base_url}/api/generate",
            json={
                "model": self._settings.chat_model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1},
            },
        )
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        return json.loads(text) if text else {}

    async def aclose(self) -> None:
        await self._client.aclose()
