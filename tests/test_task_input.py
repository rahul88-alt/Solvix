import shutil
from pathlib import Path

import pytest

from indexer.embedder import get_default_embedder
from indexer.pipeline import index_repo
from reasoning.task_input import InvalidTaskInputError, build_task_context

SAMPLE_REPO = Path(__file__).parent.parent / "sample_repo"


@pytest.fixture(scope="module")
def indexed_repo(tmp_path_factory):
    repo_copy = tmp_path_factory.mktemp("task_input_repo") / "sample_repo"
    shutil.copytree(SAMPLE_REPO, repo_copy)
    return index_repo(repo_copy)


@pytest.mark.parametrize("raw_task", ["", "   ", "\n\t  \n"])
def test_empty_or_whitespace_task_is_rejected(indexed_repo, raw_task):
    with pytest.raises(InvalidTaskInputError, match="empty"):
        build_task_context(raw_task, indexed_repo, embedder=get_default_embedder())


def test_overly_long_task_is_rejected(indexed_repo):
    raw_task = "fix the bug " * 1000
    with pytest.raises(InvalidTaskInputError, match="too long"):
        build_task_context(raw_task, indexed_repo, embedder=get_default_embedder())


def test_normal_task_flows_through_to_retrieval(indexed_repo):
    raw_task = "  Fix the subtract function, it returns the wrong result  "
    context = build_task_context(raw_task, indexed_repo, embedder=get_default_embedder())

    assert context.task == raw_task.strip()
    assert "calculator.py" in context.retrieval.file_paths


def test_task_context_bundles_raw_task_and_retrieval_result(indexed_repo):
    context = build_task_context(
        "The slugify function mishandles unicode", indexed_repo, embedder=get_default_embedder()
    )

    assert isinstance(context.task, str) and context.task
    assert context.retrieval is not None
    assert context.retrieval.files
    assert hasattr(context.retrieval, "related_files")
