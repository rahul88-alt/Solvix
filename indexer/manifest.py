"""Persistent record of file_path -> content_hash -> chunks, so index_repo can
skip re-chunking and re-embedding files that haven't changed since last run.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from indexer.chunker import Chunk

_MANIFEST_FILENAME = "index_manifest.json"


def manifest_path(repo_root: Path) -> Path:
    return repo_root / ".solvix" / _MANIFEST_FILENAME


def load_manifest(repo_root: Path) -> dict[str, dict]:
    path = manifest_path(repo_root)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_manifest(repo_root: Path, manifest: dict[str, dict]) -> None:
    path = manifest_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def chunks_to_entry(content_hash: str, chunks: list[Chunk]) -> dict:
    return {"hash": content_hash, "chunks": [asdict(c) for c in chunks]}


def entry_to_chunks(entry: dict) -> list[Chunk]:
    return [Chunk(**c) for c in entry["chunks"]]
