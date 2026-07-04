"""Single entry point that wires chunking, symbol indexing, embedding, and
vector storage together: index_repo(repo_path).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from indexer.chunker import Chunk, chunk_repo
from indexer.embedder import Embedder, get_default_embedder
from indexer.store import VectorStore
from indexer.symbol_index import SymbolIndex, build_symbol_index

_INDEX_DIRNAME = ".solvix"
_CHROMA_SUBDIR = "chroma"


@dataclass
class IndexResult:
    repo_root: Path
    num_files_indexed: int
    num_chunks: int
    symbol_index: SymbolIndex
    vector_store: VectorStore


def index_repo(
    repo_path: str | Path,
    embedder: Embedder | None = None,
) -> IndexResult:
    """Build a searchable index (chunks + symbol map + embeddings) for repo_path.

    Re-runs chunking fresh each time; symbol index and vector store are both
    derived from the same chunk list so they stay consistent with each other.
    """
    repo_root = Path(repo_path).resolve()
    if not repo_root.is_dir():
        raise ValueError(f"repo_path does not exist or is not a directory: {repo_root}")

    chunks: list[Chunk] = chunk_repo(repo_root)

    symbol_index = build_symbol_index(repo_root, chunks=chunks)

    embedder = embedder or get_default_embedder()
    embeddings = embedder.embed_texts([c.code for c in chunks])

    store_dir = repo_root / _INDEX_DIRNAME / _CHROMA_SUBDIR
    vector_store = VectorStore(store_dir)
    vector_store.upsert_chunks(chunks, embeddings)

    num_files = len({c.file_path for c in chunks})

    return IndexResult(
        repo_root=repo_root,
        num_files_indexed=num_files,
        num_chunks=len(chunks),
        symbol_index=symbol_index,
        vector_store=vector_store,
    )
