import time

from memory.task_state import (
    OUTCOME_FAILED,
    OUTCOME_IN_PROGRESS,
    OUTCOME_NEEDS_HUMAN_HELP,
    OUTCOME_SUCCESS,
    TaskStateStore,
    db_path,
)


def test_db_path_is_under_dot_solvix(tmp_path):
    assert db_path(tmp_path) == tmp_path / ".solvix" / "task_history.db"


def test_start_task_creates_in_progress_record(tmp_path):
    store = TaskStateStore(tmp_path)
    task_id = store.start_task("add a palindrome check", is_revision=False)

    records = store.list_tasks()
    assert len(records) == 1
    record = records[0]
    assert record.id == task_id
    assert record.task_text == "add a palindrome check"
    assert record.is_revision is False
    assert record.outcome == OUTCOME_IN_PROGRESS
    assert record.ended_at is None
    assert record.duration_seconds is None
    assert record.pr_url is None
    assert record.had_clarification is False


def test_db_file_actually_created_on_disk(tmp_path):
    TaskStateStore(tmp_path).start_task("x", is_revision=False)
    assert db_path(tmp_path).exists()


def test_finish_task_updates_record(tmp_path):
    store = TaskStateStore(tmp_path)
    task_id = store.start_task("add rate limiting", is_revision=False)
    time.sleep(0.01)

    store.finish_task(
        task_id,
        outcome=OUTCOME_SUCCESS,
        total_attempts=2,
        pr_url="https://github.com/acme/repo/pull/7",
        had_clarification=True,
    )

    record = store.list_tasks()[0]
    assert record.outcome == OUTCOME_SUCCESS
    assert record.total_attempts == 2
    assert record.pr_url == "https://github.com/acme/repo/pull/7"
    assert record.had_clarification is True
    assert record.ended_at is not None
    assert record.duration_seconds is not None
    assert record.duration_seconds >= 0


def test_finish_task_rejects_in_progress_as_terminal_outcome(tmp_path):
    store = TaskStateStore(tmp_path)
    task_id = store.start_task("x", is_revision=False)
    try:
        store.finish_task(task_id, outcome=OUTCOME_IN_PROGRESS)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_finish_task_pr_url_coalesces_with_start_time_value(tmp_path):
    """revise already knows the PR it's revising at start_task() time;
    finish_task() must not wipe that out when it doesn't pass a new url."""
    store = TaskStateStore(tmp_path)
    task_id = store.start_task(
        "please fix the edge case", is_revision=True, pr_url="https://github.com/acme/repo/pull/9"
    )

    store.finish_task(task_id, outcome=OUTCOME_NEEDS_HUMAN_HELP, total_attempts=3)

    record = store.list_tasks()[0]
    assert record.pr_url == "https://github.com/acme/repo/pull/9"
    assert record.outcome == OUTCOME_NEEDS_HUMAN_HELP


def test_never_finished_in_progress_record_is_queryable(tmp_path):
    """A process that crashes mid-task leaves its row at in_progress
    forever -- this must still show up in history, not vanish."""
    store = TaskStateStore(tmp_path)
    store.start_task("a task that never finishes", is_revision=False)

    records = store.list_tasks()
    assert len(records) == 1
    assert records[0].outcome == OUTCOME_IN_PROGRESS
    assert records[0].duration_seconds is None


def test_list_tasks_orders_most_recent_first_and_respects_limit(tmp_path):
    store = TaskStateStore(tmp_path)
    ids = [store.start_task(f"task {i}", is_revision=False) for i in range(5)]
    for task_id in ids:
        store.finish_task(task_id, outcome=OUTCOME_SUCCESS, total_attempts=1)

    records = store.list_tasks(limit=3)
    assert len(records) == 3
    assert [r.id for r in records] == list(reversed(ids))[:3]


def test_stats_on_empty_store(tmp_path):
    store = TaskStateStore(tmp_path)
    stats = store.stats()

    assert stats.total == 0
    assert stats.finished == 0
    assert stats.resolution_rate is None
    assert stats.retry_exhaustion_rate is None
    assert stats.median_time_to_resolution_seconds is None


def test_stats_computes_resolution_and_retry_exhaustion_rates(tmp_path):
    store = TaskStateStore(tmp_path)

    for _ in range(3):
        task_id = store.start_task("succeeds", is_revision=False)
        store.finish_task(task_id, outcome=OUTCOME_SUCCESS, total_attempts=1)

    task_id = store.start_task("needs help", is_revision=False)
    store.finish_task(task_id, outcome=OUTCOME_NEEDS_HUMAN_HELP, total_attempts=3)

    task_id = store.start_task("declined", is_revision=False)
    store.finish_task(task_id, outcome=OUTCOME_FAILED, total_attempts=0)

    store.start_task("still running", is_revision=False)  # in_progress, excluded from rates

    stats = store.stats()

    assert stats.total == 6
    assert stats.in_progress == 1
    assert stats.finished == 5
    assert stats.success == 3
    assert stats.needs_human_help == 1
    assert stats.failed == 1
    assert stats.resolution_rate == 3 / 5
    assert stats.retry_exhaustion_rate == 1 / 5


def test_stats_median_time_to_resolution_only_uses_successful_tasks(tmp_path):
    store = TaskStateStore(tmp_path)

    fast_id = store.start_task("fast success", is_revision=False)
    store.finish_task(fast_id, outcome=OUTCOME_SUCCESS, total_attempts=1)

    slow_id = store.start_task("needs help, irrelevant to median", is_revision=False)
    time.sleep(0.05)
    store.finish_task(slow_id, outcome=OUTCOME_NEEDS_HUMAN_HELP, total_attempts=3)

    stats = store.stats()

    assert stats.median_time_to_resolution_seconds is not None
    # only the successful task's (near-zero) duration should count, not the
    # slower needs_human_help one
    assert stats.median_time_to_resolution_seconds < 0.05


def test_reopening_store_reads_previously_written_data(tmp_path):
    store1 = TaskStateStore(tmp_path)
    task_id = store1.start_task("persisted task", is_revision=False)
    store1.finish_task(task_id, outcome=OUTCOME_SUCCESS, total_attempts=1)
    store1.close()

    store2 = TaskStateStore(tmp_path)
    records = store2.list_tasks()
    assert len(records) == 1
    assert records[0].task_text == "persisted task"
