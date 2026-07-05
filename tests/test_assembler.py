"""Labeled retrieval-accuracy test set for SLX-A2 (Master Document 7.3 /
Epic A2 acceptance criteria): given a task description, the top-N retrieved
files must include the file(s) that actually require changes.

Uses the real local embedder (cached locally, no network needed) since
semantic similarity is exactly what's under test — a fake embedder would
make the assertions meaningless.
"""

import shutil
from pathlib import Path

import pytest

from context.assembler import build_import_graph, retrieve_relevant_files
from indexer.embedder import get_default_embedder
from indexer.pipeline import index_repo

SAMPLE_REPO = Path(__file__).parent.parent / "sample_repo"

# (task description, expected file(s) that should appear in the top-N)
LABELED_TASKS = [
    (
        "Fix the subtract function, it returns the wrong result for negative numbers",
        {"calculator.py"},
    ),
    (
        "The slugify function doesn't handle unicode characters correctly, fix it",
        {"utils/strings.py"},
    ),
    (
        "Add input validation to the Calculator class's add method",
        {"calculator.py"},
    ),
    (
        "truncate is cutting words off in the middle instead of at a word boundary",
        {"utils/strings.py"},
    ),
    (
        "format_summary in the report module shows the wrong difference between two numbers",
        {"report.py"},
    ),
]


@pytest.fixture(scope="module")
def indexed_repo(tmp_path_factory):
    repo_copy = tmp_path_factory.mktemp("assembler_repo") / "sample_repo"
    shutil.copytree(SAMPLE_REPO, repo_copy)
    result = index_repo(repo_copy)
    return result


@pytest.mark.parametrize("task,expected_files", LABELED_TASKS)
def test_top_n_retrieval_includes_expected_file(indexed_repo, task, expected_files):
    result = retrieve_relevant_files(
        task, indexed_repo, embedder=get_default_embedder(), top_n=3
    )
    retrieved = set(result.file_paths)
    assert expected_files & retrieved, (
        f"expected one of {expected_files} in top-N, got {result.file_paths} "
        f"(scores: {[(f.file_path, f.score, f.reasons) for f in result.files]})"
    )


def test_exact_symbol_match_outranks_pure_embedding_similarity(indexed_repo):
    result = retrieve_relevant_files(
        "there's a bug in subtract", indexed_repo, embedder=get_default_embedder(), top_n=3
    )
    assert result.files[0].file_path == "calculator.py"
    assert any(r.startswith("symbol_match:") for r in result.files[0].reasons)


def test_one_hop_related_files_follow_import_edges(indexed_repo):
    result = retrieve_relevant_files(
        "fix format_summary in report.py", indexed_repo, embedder=get_default_embedder(), top_n=1
    )
    assert result.files[0].file_path == "report.py"
    related_paths = {f.file_path for f in result.related_files}
    # report.py imports from both calculator.py and utils/strings.py
    assert {"calculator.py", "utils/strings.py"} <= related_paths


def test_build_import_graph_resolves_intra_repo_imports():
    graph = build_import_graph(SAMPLE_REPO)
    assert graph["report.py"] == {"calculator.py", "utils/strings.py"}
    assert graph["calculator.py"] == set()


def test_build_import_graph_excludes_vendored_dependencies(tmp_path):
    """Epic A6: build_import_graph must never walk into `.venv`/etc -- a
    real run against Solvix's own repo previously crashed here because a
    `.venv`-vendored dependency's test fixture wasn't valid UTF-8, and more
    generally a repo's own import graph has no business parsing third-party
    dependency source at all.
    """
    (tmp_path / "main.py").write_text("import helper\n")
    (tmp_path / "helper.py").write_text("x = 1\n")
    venv_pkg = tmp_path / ".venv" / "lib" / "somepkg"
    venv_pkg.mkdir(parents=True)
    (venv_pkg / "__init__.py").write_text("import os\n")

    graph = build_import_graph(tmp_path)

    assert set(graph.keys()) == {"main.py", "helper.py"}
    assert graph["main.py"] == {"helper.py"}


def test_build_import_graph_skips_non_utf8_file_with_warning(tmp_path):
    (tmp_path / "main.py").write_text("import helper\n")
    (tmp_path / "helper.py").write_text("x = 1\n")
    (tmp_path / "bad.py").write_bytes(b"# -*- coding: big5 -*-\nx = '\xa4@'\n")

    with pytest.warns(UserWarning, match="bad.py"):
        graph = build_import_graph(tmp_path)

    # the bad file is still present as a graph node (it exists in the repo),
    # just never parsed for its own imports -- and every other file is
    # unaffected
    assert graph["bad.py"] == set()
    assert graph["main.py"] == {"helper.py"}
