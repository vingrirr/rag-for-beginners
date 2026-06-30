from datetime import datetime, timedelta, timezone

import httpx
import jwt

from app.config import AppSettings
from app.models import GhostPost


class GhostClient:
    """Ghost Admin API client. Signs short-lived JWTs from an `id:secret`
    Admin API key and pages through /admin/posts/ to fetch full HTML."""

    def __init__(self, settings: AppSettings, client: httpx.AsyncClient | None = None):
        self._settings = settings
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(30.0))

    def _make_token(self) -> str:
        key_id, secret = self._settings.ghost_admin_api_key.split(":", 1)
        now = datetime.now(tz=timezone.utc)
        payload = {
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=5)).timestamp()),
            "aud": "/admin/",
        }
        return jwt.encode(
            payload,
            bytes.fromhex(secret),
            algorithm="HS256",
            headers={"kid": key_id, "alg": "HS256", "typ": "JWT"},
        )

    async def fetch_posts(self, limit: int | None = None) -> list[GhostPost]:
        if not self._settings.ghost_url or not self._settings.ghost_admin_api_key:
            raise RuntimeError("GHOST_URL and GHOST_ADMIN_API_KEY must be set")

        token = self._make_token()
        headers = {"Authorization": f"Ghost {token}"}
        page_size = 100 if limit is None else min(limit, 100)

        out: list[GhostPost] = []
        page = 1
        while True:
            params = {
                "limit": page_size,
                "page": page,
                "formats": "html",
                "include": "tags,authors",
                "filter": "status:[published,draft,scheduled]",
            }
            resp = await self._client.get(
                f"{self._settings.ghost_url}/ghost/api/admin/posts/",
                params=params,
                headers=headers,
            )
            resp.raise_for_status()
            body = resp.json()
            for raw in body.get("posts", []):
                out.append(
                    GhostPost(
                        id=raw["id"],
                        slug=raw["slug"],
                        title=raw.get("title") or raw["slug"],
                        url=raw.get("url"),
                        html=raw.get("html") or "",
                        published_at=raw.get("published_at"),
                        updated_at=raw.get("updated_at"),
                    )
                )
                if limit is not None and len(out) >= limit:
                    return out

            pagination = body.get("meta", {}).get("pagination", {})
            if not pagination.get("next"):
                break
            page = pagination["next"]
        return out

    async def aclose(self) -> None:
        await self._client.aclose()
