"""Labeled retrieval-accuracy test set for SLX-A2 (Master Document 7.3 /
Epic A2 acceptance criteria): given a task description, the top-N retrieved
files must include the file(s) that actually require changes.

Uses the real local embedder (cached locally, no network needed) since
semantic similarity is exactly what's under test — a fake embedder would
make the assertions meaningless.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from context.assembler import DEFAULT_TOKEN_BUDGET, build_import_graph, estimate_tokens, retrieve_relevant_files
from indexer.embedder import get_default_embedder
from indexer.pipeline import index_repo
from indexer.symbol_index import SymbolLocation

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


# --- SLX-A5: token-budget truncation -----------------------------------


def test_estimate_tokens_is_a_length_based_approximation():
    assert estimate_tokens("") == 1  # never zero, even for empty text
    assert estimate_tokens("ab") == 1
    assert estimate_tokens("a" * 40) == 10  # ~4 chars/token


def test_default_budget_does_not_truncate_within_budget_retrieval(indexed_repo):
    """Regression check: sample_repo's retrieval result is tiny relative to
    the default budget, so nothing should be truncated -- same files/related
    files as before A5, just with the new estimated_tokens field populated.
    """
    result = retrieve_relevant_files(
        "fix format_summary in report.py", indexed_repo, embedder=get_default_embedder(), top_n=1
    )
    assert result.files[0].file_path == "report.py"
    assert {"calculator.py", "utils/strings.py"} <= {f.file_path for f in result.related_files}
    assert 0 < result.estimated_tokens <= DEFAULT_TOKEN_BUDGET


class _FakeVectorStore:
    def __init__(self, hits):
        self._hits = hits

    def query(self, query_embedding, top_k=5):
        return self._hits[:top_k]


class _FakeSymbolIndex:
    def __init__(self, locations: dict[str, list[SymbolLocation]], symbols: list[str]):
        self._locations = locations
        self._symbols = symbols

    def all_symbols(self):
        return self._symbols

    def lookup(self, symbol):
        return self._locations.get(symbol, [])


class _FakeEmbedder:
    def embed_texts(self, texts):
        return [[0.0] for _ in texts]


@dataclass
class _FakeIndexResult:
    repo_root: Path
    symbol_index: _FakeSymbolIndex
    vector_store: _FakeVectorStore


def _budget_test_repo(tmp_path: Path) -> _FakeIndexResult:
    """A tiny synthetic repo + hand-built symbol index/vector-store hits, so
    exact chunk content (and therefore exact token counts) are fully known --
    unlike the real sample_repo/embedder, which don't give deterministic
    control over evidence ordering.

    - "foo" (a.py) is named explicitly in the task text -> symbol-match tier.
    - "bar" (b.py) is only surfaced via a fake embedding-similarity hit ->
      embedding-chunk tier, lower priority than the symbol match.
    - c.py imports a.py, making it a's one-hop related file -> lowest tier.
    """
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n")
    (tmp_path / "b.py").write_text("def bar():\n    return 2\n")
    (tmp_path / "c.py").write_text("import a\n")

    foo_location = SymbolLocation(file_path="a.py", start_line=1, end_line=2, kind="function")
    bar_code = "def bar():\n    return 2"

    symbol_index = _FakeSymbolIndex(
        locations={"foo": [foo_location]},
        symbols=["foo", "bar"],
    )
    vector_store = _FakeVectorStore(
        hits=[
            {
                "id": "b.py::bar::1-2",
                "metadata": {"file_path": "b.py", "symbol": "bar", "kind": "function", "start_line": 1, "end_line": 2},
                "code": bar_code,
                "distance": 0.0,
            }
        ]
    )
    return _FakeIndexResult(repo_root=tmp_path, symbol_index=symbol_index, vector_store=vector_store)


def test_synthetic_truncation_keeps_symbol_match_over_embedding_chunk(tmp_path):
    index_result = _budget_test_repo(tmp_path)
    foo_tokens = estimate_tokens("def foo():\n    return 1")
    bar_tokens = estimate_tokens("def bar():\n    return 2")

    # budget fits exactly the symbol-match chunk (foo) and nothing more
    result = retrieve_relevant_files(
        "fix foo", index_result, embedder=_FakeEmbedder(), top_n=2, token_budget=foo_tokens
    )

    assert result.file_paths == ["a.py"]
    assert result.related_files == []
    assert result.estimated_tokens == foo_tokens
    assert result.estimated_tokens <= foo_tokens

    # a bigger budget (foo + bar, still nothing left for the one-hop ref)
    result2 = retrieve_relevant_files(
        "fix foo", index_result, embedder=_FakeEmbedder(), top_n=2, token_budget=foo_tokens + bar_tokens
    )
    assert set(result2.file_paths) == {"a.py", "b.py"}
    assert result2.related_files == []
    assert result2.estimated_tokens == foo_tokens + bar_tokens

    # a generous budget includes the one-hop related file too
    result3 = retrieve_relevant_files(
        "fix foo", index_result, embedder=_FakeEmbedder(), top_n=2, token_budget=10_000
    )
    assert set(result3.file_paths) == {"a.py", "b.py"}
    assert [f.file_path for f in result3.related_files] == ["c.py"]
    assert result3.estimated_tokens == foo_tokens + bar_tokens + estimate_tokens("c.py")


def test_synthetic_truncation_never_exceeds_budget_once_past_the_floor_item(tmp_path):
    """The budget invariant only holds once the budget is at least large
    enough for the single highest-priority item -- see
    test_synthetic_tiny_budget_still_returns_top_item for the floor case,
    where estimated_tokens is allowed to exceed a too-small budget.
    """
    index_result = _budget_test_repo(tmp_path)
    foo_tokens = estimate_tokens("def foo():\n    return 1")

    for budget in (foo_tokens, foo_tokens + 5, foo_tokens + 20, 1000):
        result = retrieve_relevant_files(
            "fix foo", index_result, embedder=_FakeEmbedder(), top_n=2, token_budget=budget
        )
        assert result.estimated_tokens <= budget


def test_synthetic_tiny_budget_still_returns_top_item(tmp_path):
    """A budget too small even for the single highest-priority item (e.g. a
    misconfigured .solvix.yml) must still return that one item rather than
    an empty result -- a plan grounded in one slightly-over-budget symbol
    match beats a plan built on no context at all.
    """
    index_result = _budget_test_repo(tmp_path)
    foo_tokens = estimate_tokens("def foo():\n    return 1")

    for budget in (0, 1, foo_tokens - 1):
        result = retrieve_relevant_files(
            "fix foo", index_result, embedder=_FakeEmbedder(), top_n=2, token_budget=budget
        )
        assert result.file_paths == ["a.py"]
        assert result.related_files == []
        assert result.estimated_tokens == foo_tokens
        assert result.estimated_tokens > budget  # the floor guarantee overshoots on purpose
