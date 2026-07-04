"""Tracks task outcomes across runs (Master Document 7.2/7.3, Epic D4): a
SQLite file per target repo at <repo>/.solvix/task_history.db, the same
hidden-directory convention indexer.pipeline already uses for its Chroma
store (<repo>/.solvix/chroma). Before this, every `solvix run`/`solvix
revise` was a fresh, memoryless process -- nothing survived past the
CLI printing its own output, so a manager asking "how reliable is this
thing" had no answer but re-reading terminal scrollback.

This is a task-level log -- one row per `run`/`revise` invocation -- not
the more granular per-attempt table the architecture doc's Memory/State
section describes ("tracking attempt number, diff proposed, verification
result, and error summary"); execution.orchestrator.StepResult/TaskResult
already carry that per-attempt detail for the PR body (review.pr_builder),
and the D4 story's own schema is explicitly task-level (task text,
start/end, outcome, total attempts, PR URL, clarification/revision flags),
so that's what's implemented here.

cli.py owns the row's lifecycle: start_task() is called at the very start
of `run`/`revise` (outcome=in_progress), and finish_task() is called once
the pipeline actually finishes, in every code path (success, needs_human_help,
a declined guardrail, an unhandled exception) via a try/finally -- so a
process that crashes or is killed mid-task leaves its row sitting at
in_progress rather than silently missing, which is the whole point of
logging at the start rather than only at the end.

Outcome has four states, not the three the story names explicitly
(success / needs_human_help / in_progress): a plan-approval or
dangerous-ops confirmation the user declines is success=False,
needs_human_help=False on TaskResult -- a deliberate stop, not "the agent
exhausted its retries and is stuck" (see execution.orchestrator.StepResult's
docstring). Folding that into needs_human_help would inflate the
retry-exhaustion rate with cases that were never a retry problem, so it
gets its own `failed` bucket instead.
"""

from __future__ import annotations

import sqlite3
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_DB_DIRNAME = ".solvix"
_DB_FILENAME = "task_history.db"

OUTCOME_IN_PROGRESS = "in_progress"
OUTCOME_SUCCESS = "success"
OUTCOME_NEEDS_HUMAN_HELP = "needs_human_help"
OUTCOME_FAILED = "failed"

_FINISHED_OUTCOMES = (OUTCOME_SUCCESS, OUTCOME_NEEDS_HUMAN_HELP, OUTCOME_FAILED)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS task_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_text TEXT NOT NULL,
    is_revision INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    outcome TEXT NOT NULL,
    total_attempts INTEGER NOT NULL DEFAULT 0,
    pr_url TEXT,
    had_clarification INTEGER NOT NULL DEFAULT 0
)
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


@dataclass(frozen=True)
class TaskRecord:
    id: int
    task_text: str
    is_revision: bool
    started_at: datetime
    ended_at: datetime | None
    outcome: str
    total_attempts: int
    pr_url: str | None
    had_clarification: bool

    @property
    def duration_seconds(self) -> float | None:
        """Wall-clock time from start_task() to finish_task(); None for a
        record that's still in_progress (never finished, or the process
        that owned it crashed before finishing).
        """
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()


@dataclass(frozen=True)
class TaskStats:
    total: int
    in_progress: int
    success: int
    needs_human_help: int
    failed: int
    resolution_rate: float | None
    retry_exhaustion_rate: float | None
    median_time_to_resolution_seconds: float | None

    @property
    def finished(self) -> int:
        return self.success + self.needs_human_help + self.failed


def db_path(repo_root: str | Path) -> Path:
    return Path(repo_root) / _DB_DIRNAME / _DB_FILENAME


def _row_to_record(row: tuple) -> TaskRecord:
    (
        id_,
        task_text,
        is_revision,
        started_at,
        ended_at,
        outcome,
        total_attempts,
        pr_url,
        had_clarification,
    ) = row
    return TaskRecord(
        id=id_,
        task_text=task_text,
        is_revision=bool(is_revision),
        started_at=_parse_iso(started_at),
        ended_at=_parse_iso(ended_at) if ended_at is not None else None,
        outcome=outcome,
        total_attempts=total_attempts,
        pr_url=pr_url,
        had_clarification=bool(had_clarification),
    )


class TaskStateStore:
    """One instance per repo_root, backed by <repo_root>/.solvix/task_history.db."""

    _COLUMNS = "id, task_text, is_revision, started_at, ended_at, outcome, total_attempts, pr_url, had_clarification"

    def __init__(self, repo_root: str | Path):
        self._path = db_path(repo_root)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "TaskStateStore":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def start_task(self, task_text: str, is_revision: bool, pr_url: str | None = None) -> int:
        """Insert a new in_progress row and return its id. pr_url is
        accepted here (not just at finish_task) because `revise` already
        knows which PR it's revising before the pipeline even runs --
        unlike `run`, which only learns a PR's URL after a successful
        delivery.
        """
        cursor = self._conn.execute(
            "INSERT INTO task_history "
            "(task_text, is_revision, started_at, outcome, total_attempts, pr_url, had_clarification) "
            "VALUES (?, ?, ?, ?, 0, ?, 0)",
            (task_text, int(is_revision), _now_iso(), OUTCOME_IN_PROGRESS, pr_url),
        )
        self._conn.commit()
        return cursor.lastrowid

    def finish_task(
        self,
        task_id: int,
        outcome: str,
        total_attempts: int = 0,
        pr_url: str | None = None,
        had_clarification: bool = False,
    ) -> None:
        if outcome not in _FINISHED_OUTCOMES:
            raise ValueError(f"finish_task requires a terminal outcome, got {outcome!r}")

        # COALESCE keeps a pr_url set at start_task() (revise's existing PR)
        # from being wiped out by a finish_task() call that doesn't pass one
        # (e.g. a failed revision never reaches the point of learning a new
        # URL, but the PR it was revising is still the relevant one to show).
        self._conn.execute(
            "UPDATE task_history "
            "SET ended_at = ?, outcome = ?, total_attempts = ?, pr_url = COALESCE(?, pr_url), had_clarification = ? "
            "WHERE id = ?",
            (_now_iso(), outcome, total_attempts, pr_url, int(had_clarification), task_id),
        )
        self._conn.commit()

    def list_tasks(self, limit: int = 20) -> list[TaskRecord]:
        rows = self._conn.execute(
            f"SELECT {self._COLUMNS} FROM task_history ORDER BY started_at DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_record(row) for row in rows]

    def stats(self) -> TaskStats:
        rows = self._conn.execute(f"SELECT {self._COLUMNS} FROM task_history").fetchall()
        records = [_row_to_record(row) for row in rows]

        in_progress = sum(1 for r in records if r.outcome == OUTCOME_IN_PROGRESS)
        success_records = [r for r in records if r.outcome == OUTCOME_SUCCESS]
        needs_human_help = sum(1 for r in records if r.outcome == OUTCOME_NEEDS_HUMAN_HELP)
        failed = sum(1 for r in records if r.outcome == OUTCOME_FAILED)
        finished = len(success_records) + needs_human_help + failed

        durations = [r.duration_seconds for r in success_records if r.duration_seconds is not None]

        return TaskStats(
            total=len(records),
            in_progress=in_progress,
            success=len(success_records),
            needs_human_help=needs_human_help,
            failed=failed,
            resolution_rate=(len(success_records) / finished) if finished else None,
            retry_exhaustion_rate=(needs_human_help / finished) if finished else None,
            median_time_to_resolution_seconds=(statistics.median(durations) if durations else None),
        )
