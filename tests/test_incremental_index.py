"""Tests for SLX-A4: incremental re-indexing.

On a second index_repo() call, unchanged files should be skipped entirely
(no re-chunk, no re-embed), while changed/added/deleted files are handled
correctly and the symbol index stays consistent with the vector store.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from indexer.pipeline import index_repo

SAMPLE_REPO = Path(__file__).parent.parent / "sample_repo"


class SpyEmbedder:
    """Deterministic embedder that records every text it was asked to embed."""

    def __init__(self) -> None:
        self.call_count = 0
        self.embedded_texts: list[str] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        self.embedded_texts.extend(texts)
        return [[float(len(t)), float(t.count("\n"))] for t in texts]


def _copy_sample_repo(tmp_path: Path) -> Path:
    repo_copy = tmp_path / "sample_repo"
    shutil.copytree(SAMPLE_REPO, repo_copy)
    return repo_copy


def test_first_index_is_full_baseline(tmp_path):
    repo = _copy_sample_repo(tmp_path)
    spy = SpyEmbedder()

    result = index_repo(repo, embedder=spy)

    assert result.num_files_indexed == 4
    assert result.num_chunks == 15
    assert result.vector_store.count() == 15
    assert spy.call_count > 0


def test_second_call_with_no_changes_does_not_reembed(tmp_path):
    repo = _copy_sample_repo(tmp_path)
    index_repo(repo, embedder=SpyEmbedder())

    spy = SpyEmbedder()
    result = index_repo(repo, embedder=spy)

    assert spy.call_count == 0
    assert result.num_chunks == 15
    assert result.vector_store.count() == 15
    # retrieval still works against the reused vectors
    hits = result.vector_store.query([0.0, 0.0], top_k=15)
    assert len(hits) == 15
    assert result.symbol_index.lookup("Calculator.add")


def test_modifying_one_file_only_reembeds_that_file(tmp_path):
    repo = _copy_sample_repo(tmp_path)
    index_repo(repo, embedder=SpyEmbedder())

    calculator = repo / "calculator.py"
    original = calculator.read_text()
    calculator.write_text(original + "\n\ndef multiply(a, b):\n    return a * b\n")

    spy = SpyEmbedder()
    result = index_repo(repo, embedder=spy)

    # calculator.py now has 7 chunks (6 original + multiply); only its
    # chunks should have been sent to the embedder.
    assert spy.call_count == 1
    assert len(spy.embedded_texts) == 7
    assert any("multiply" in t for t in spy.embedded_texts)

    # other files' chunks are untouched, total count reflects the addition
    assert result.num_chunks == 16
    assert result.vector_store.count() == 16
    assert result.symbol_index.lookup("multiply")
    # unrelated symbols from untouched files are still present
    assert result.symbol_index.lookup("slugify")


def test_deleting_a_file_removes_its_vectors_and_symbols(tmp_path):
    repo = _copy_sample_repo(tmp_path)
    index_repo(repo, embedder=SpyEmbedder())

    (repo / "report.py").unlink()

    spy = SpyEmbedder()
    result = index_repo(repo, embedder=spy)

    assert spy.call_count == 0  # nothing new to embed, just a deletion
    assert result.num_chunks == 14
    assert result.vector_store.count() == 14
    assert not result.symbol_index.lookup("format_summary")

    hits = result.vector_store.query([0.0, 0.0], top_k=14)
    assert all(hit["metadata"]["file_path"] != "report.py" for hit in hits)


def test_adding_a_new_file_gets_indexed_and_is_retrievable(tmp_path):
    repo = _copy_sample_repo(tmp_path)
    index_repo(repo, embedder=SpyEmbedder())

    new_file = repo / "greeter.py"
    new_file.write_text("def greet(name):\n    return f'hello {name}'\n")

    spy = SpyEmbedder()
    result = index_repo(repo, embedder=spy)

    assert spy.call_count == 1
    assert len(spy.embedded_texts) == 1
    assert "greet" in spy.embedded_texts[0]

    assert result.num_files_indexed == 5
    assert result.num_chunks == 16
    assert result.symbol_index.lookup("greet")

    hits = result.vector_store.query([0.0, 0.0], top_k=16)
    assert any(hit["metadata"]["file_path"] == "greeter.py" for hit in hits)
