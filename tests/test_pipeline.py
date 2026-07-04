import shutil
from pathlib import Path

from indexer.pipeline import index_repo

SAMPLE_REPO = Path(__file__).parent.parent / "sample_repo"


class FakeEmbedder:
    """Deterministic, network-free embedder for testing pipeline wiring."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), float(text.count("\n"))] for t, text in zip(texts, texts)]


def test_index_repo_wires_chunking_symbols_and_vector_store(tmp_path):
    # copy into tmp_path so the pipeline's .solvix/ index dir doesn't get
    # written into the checked-in sample_repo fixture.
    repo_copy = tmp_path / "sample_repo"
    shutil.copytree(SAMPLE_REPO, repo_copy)

    result = index_repo(repo_copy, embedder=FakeEmbedder())

    assert result.num_files_indexed == 2
    assert result.num_chunks == 8
    assert result.symbol_index.lookup("Calculator.add")
    assert result.vector_store.count() == 8


def test_index_repo_raises_for_missing_path():
    import pytest

    with pytest.raises(ValueError):
        index_repo("/no/such/path/exists")
