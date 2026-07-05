"""Single entry point that wires chunking, symbol indexing, embedding, and
vector storage together: index_repo(repo_path).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from indexer.chunker import Chunk, chunk_file, iter_source_files
from indexer.embedder import Embedder, get_default_embedder
from indexer.manifest import chunks_to_entry, entry_to_chunks, load_manifest, save_manifest
from indexer.store import VectorStore
from indexer.symbol_index import SymbolIndex, build_symbol_index

_INDEX_DIRNAME = ".solvix"
_CHROMA_SUBDIR = "chroma"

logger = logging.getLogger(__name__)


@dataclass
class IndexResult:
    repo_root: Path
    num_files_indexed: int
    num_chunks: int
    symbol_index: SymbolIndex
    vector_store: VectorStore


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index_repo(
    repo_path: str | Path,
    embedder: Embedder | None = None,
) -> IndexResult:
    """Build a searchable index (chunks + symbol map + embeddings) for repo_path.

    Incremental: a manifest at .solvix/index_manifest.json records each file's
    content hash and chunks from the last run. A file whose hash is unchanged
    reuses its cached chunks (no re-chunk, no re-embed, no vector-store write).
    Only new/changed files are re-chunked and re-embedded, and files removed
    from disk since the last run have their vectors dropped. The symbol index
    is rebuilt fresh each call from the full (reused + updated) chunk list, so
    it can never drift out of sync with the vector store.
    """
    repo_root = Path(repo_path).resolve()
    if not repo_root.is_dir():
        raise ValueError(f"repo_path does not exist or is not a directory: {repo_root}")

    store_dir = repo_root / _INDEX_DIRNAME / _CHROMA_SUBDIR
    vector_store = VectorStore(store_dir)

    manifest = load_manifest(repo_root)
    current_paths = list(iter_source_files(repo_root))
    current_rel_paths = {str(p.relative_to(repo_root)) for p in current_paths}

    # deleted files: present in the manifest from a previous run, gone now.
    for rel_path in list(manifest.keys()):
        if rel_path not in current_rel_paths:
            vector_store.delete_by_file_path(rel_path)
            del manifest[rel_path]

    embedder = embedder or get_default_embedder()

    all_chunks: list[Chunk] = []
    num_unchanged = 0
    num_reindexed = 0

    for path in current_paths:
        rel_path = str(path.relative_to(repo_root))
        content_hash = _hash_file(path)
        cached = manifest.get(rel_path)

        if cached is not None and cached["hash"] == content_hash:
            chunks = entry_to_chunks(cached)
            all_chunks.extend(chunks)
            num_unchanged += 1
            continue

        try:
            chunks = chunk_file(path, repo_root)
        except UnicodeDecodeError as error:
            logger.warning("skipping %s: not valid UTF-8 (%s)", rel_path, error)
            if cached is not None:
                vector_store.delete_by_file_path(rel_path)
                del manifest[rel_path]
            continue

        if cached is not None:
            # existing chunk IDs may not match new ones (e.g. shifted line
            # ranges), so drop the old vectors before upserting fresh ones.
            vector_store.delete_by_file_path(rel_path)
        if chunks:
            embeddings = embedder.embed_texts([c.code for c in chunks])
            vector_store.upsert_chunks(chunks, embeddings)
        manifest[rel_path] = chunks_to_entry(content_hash, chunks)
        all_chunks.extend(chunks)
        num_reindexed += 1

    logger.info(
        "index_repo: %d file(s) unchanged (skipped), %d file(s) re-indexed",
        num_unchanged,
        num_reindexed,
    )

    save_manifest(repo_root, manifest)

    symbol_index = build_symbol_index(repo_root, chunks=all_chunks)

    num_files = len({c.file_path for c in all_chunks})

    return IndexResult(
        repo_root=repo_root,
        num_files_indexed=num_files,
        num_chunks=len(all_chunks),
        symbol_index=symbol_index,
        vector_store=vector_store,
    )
