"""Session-wide fixtures for Solvix's own test suite (not the sample repo's).

A manual `solvix run` against the checked-in sample_repo/ fixture (e.g. for
a real end-to-end smoke test) leaves a `.solvix/` index directory behind --
gitignored, but if left in place it gets swept up by
tests/test_pipeline.py's `shutil.copytree(SAMPLE_REPO, ...)` and can break
that test with a stale, differently-dimensioned vector store. Cleaning it
up automatically here means that's no longer something a developer has to
remember to do by hand after a manual smoke test.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

_SAMPLE_REPO_SOLVIX_DIR = Path(__file__).parent.parent / "sample_repo" / ".solvix"


@pytest.fixture(autouse=True, scope="session")
def _clean_sample_repo_solvix_dir():
    shutil.rmtree(_SAMPLE_REPO_SOLVIX_DIR, ignore_errors=True)
    yield
    shutil.rmtree(_SAMPLE_REPO_SOLVIX_DIR, ignore_errors=True)
