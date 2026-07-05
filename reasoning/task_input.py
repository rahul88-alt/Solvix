"""Accepts a raw free-text task description and turns it into structured
input the reasoning pipeline can use (Master Document 7.2/7.3, Epic B1).

This module does not plan or reason about the task (that's the Planner,
SLX-B3) — it only validates/normalizes the raw text and bundles it with the
retrieved context from context.assembler, so downstream stages always
receive the same shape regardless of how free-form the original input was.
"""

from __future__ import annotations

from dataclasses import dataclass

from context.assembler import DEFAULT_TOKEN_BUDGET, RetrievalResult, retrieve_relevant_files
from indexer.embedder import Embedder
from indexer.pipeline import IndexResult

_MIN_LENGTH = 1
_MAX_LENGTH = 4000


class InvalidTaskInputError(ValueError):
    """Raised when the raw task description fails validation."""


@dataclass(frozen=True)
class TaskContext:
    """Bundles the original task text with the context retrieved for it."""

    task: str
    retrieval: RetrievalResult


def _normalize(raw_task: str) -> str:
    if not isinstance(raw_task, str):
        raise InvalidTaskInputError(f"task must be a string, got {type(raw_task).__name__}")

    task = raw_task.strip()
    if len(task) < _MIN_LENGTH:
        raise InvalidTaskInputError("task description cannot be empty or whitespace-only")
    if len(task) > _MAX_LENGTH:
        raise InvalidTaskInputError(
            f"task description is too long ({len(task)} chars, max {_MAX_LENGTH})"
        )
    return task


def build_task_context(
    raw_task: str,
    index_result: IndexResult,
    embedder: Embedder,
    top_n: int = 3,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> TaskContext:
    """Validate/normalize a raw task string and retrieve its relevant context.

    Raises InvalidTaskInputError if the raw text fails validation.
    """
    task = _normalize(raw_task)
    retrieval = retrieve_relevant_files(task, index_result, embedder, top_n=top_n, token_budget=token_budget)
    return TaskContext(task=task, retrieval=retrieval)
