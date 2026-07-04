"""Embedding generation for code chunks.

Uses local sentence-transformers by default so indexing works out of the box
for a solo developer with no paid API key required (Master Document 7.1 lists
this as the offline option). Swappable behind Embedder for an API-based
embedder (Voyage/OpenAI) later without touching the rest of the pipeline.
"""

from __future__ import annotations

from typing import Protocol

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Embedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class LocalEmbedder:
    """Local, offline embedder backed by sentence-transformers."""

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        # imported lazily so importing this module doesn't force-load torch
        # for callers who only need chunking/symbol indexing.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return vectors.tolist()


_default_embedder: LocalEmbedder | None = None


def get_default_embedder() -> LocalEmbedder:
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = LocalEmbedder()
    return _default_embedder
