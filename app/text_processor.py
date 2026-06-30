import re

from bs4 import BeautifulSoup

from app.config import AppSettings


BLOCK_TAGS = ("p", "h1", "h2", "h3", "h4", "li", "blockquote")
_WS = re.compile(r"\s+")


class TextProcessor:
    """Ports the C# TextProcessor: HTML -> block-tag plain text, then word-based
    chunking with overlap on paragraph boundaries. Images and other tags are
    ignored on purpose."""

    def __init__(self, settings: AppSettings):
        self._settings = settings

    def html_to_text(self, html: str) -> str:
        if not html or not html.strip():
            return ""
        soup = BeautifulSoup(html, "html.parser")
        parts: list[str] = []
        for node in soup.find_all(BLOCK_TAGS):
            text = node.get_text(separator=" ", strip=True)
            if not text:
                continue
            parts.append(_WS.sub(" ", text))
        return "\n\n".join(parts)

    def chunk_text(self, text: str) -> list[str]:
        chunk_size = self._settings.chunk_size
        overlap = self._settings.chunk_overlap

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: list[str] = []
        current: list[str] = []

        for para in paragraphs:
            words = [w for w in para.split(" ") if w]
            if len(current) + len(words) > chunk_size and current:
                chunks.append(" ".join(current))
                if 0 < overlap < len(current):
                    current = current[-overlap:]
                else:
                    current = list(current)
            current.extend(words)

        if current:
            chunks.append(" ".join(current))
        return chunks
