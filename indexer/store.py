"""Local vector store for code chunks, backed by Chroma.

Chroma is chosen over a flat FAISS index because it persists vectors +
metadata (file path, symbol, line range) together with a simple upsert/query
API, so no separate metadata-persistence layer is needed. At MVP scale
(thousands of chunks) FAISS's raw speed advantage isn't relevant.
"""

from __future__ import annotations

from pathlib import Path

import chromadb

from indexer.chunker import Chunk

_COLLECTION_NAME = "solvix_chunks"


def _chunk_id(chunk: Chunk) -> str:
    return f"{chunk.file_path}::{chunk.symbol}::{chunk.start_line}-{chunk.end_line}"


class VectorStore:
    """Wraps a persistent Chroma collection of code-chunk embeddings."""

    def __init__(self, persist_dir: Path) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(_COLLECTION_NAME)

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[_chunk_id(c) for c in chunks],
            embeddings=embeddings,
            documents=[c.code for c in chunks],
            metadatas=[
                {
                    "file_path": c.file_path,
                    "symbol": c.symbol,
                    "kind": c.kind,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                }
                for c in chunks
            ],
        )

    def query(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        hits = []
        ids = result.get("ids", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        documents = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for i in range(len(ids)):
            hits.append(
                {
                    "id": ids[i],
                    "metadata": metadatas[i],
                    "code": documents[i],
                    "distance": distances[i],
                }
            )
        return hits

    def count(self) -> int:
        return self._collection.count()
