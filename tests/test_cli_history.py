from click.testing import CliRunner

import cli
from memory.task_state import OUTCOME_FAILED, OUTCOME_NEEDS_HUMAN_HELP, OUTCOME_SUCCESS, TaskStateStore


def test_history_with_no_records(tmp_path):
    result = CliRunner().invoke(cli.cli, ["history", "--repo", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "No task history yet" in result.output


def test_history_shows_seeded_records_most_recent_first(tmp_path):
    store = TaskStateStore(tmp_path)
    older_id = store.start_task("add a palindrome check", is_revision=False)
    store.finish_task(older_id, outcome=OUTCOME_SUCCESS, total_attempts=1, pr_url="https://github.com/acme/repo/pull/1")

    newer_id = store.start_task(
        "please also handle unicode input in the palindrome checker this time around",
        is_revision=True,
        pr_url="https://github.com/acme/repo/pull/1",
    )
    store.finish_task(newer_id, outcome=OUTCOME_NEEDS_HUMAN_HELP, total_attempts=3)
    store.close()

    result = CliRunner().invoke(cli.cli, ["history", "--repo", str(tmp_path)])

    assert result.exit_code == 0, result.output
    output = result.output
    assert "add a palindrome check" in output
    assert "please also handle unicode" in output
    assert "…" in output  # long task text got truncated
    assert "success" in output
    assert "needs_human_help" in output
    assert "revise" in output
    assert "run" in output
    assert "https://github.com/acme/repo/pull/1" in output
    # most recent (the revision) listed before the older run
    assert output.index("please also handle unicode") < output.index("add a palindrome check")


def test_history_respects_limit(tmp_path):
    store = TaskStateStore(tmp_path)
    for i in range(5):
        task_id = store.start_task(f"task number {i}", is_revision=False)
        store.finish_task(task_id, outcome=OUTCOME_SUCCESS, total_attempts=1)
    store.close()

    result = CliRunner().invoke(cli.cli, ["history", "--repo", str(tmp_path), "--limit", "2"])

    assert result.exit_code == 0, result.output
    shown = sum(1 for i in range(5) if f"task number {i}" in result.output)
    assert shown == 2
    assert "task number 4" in result.output
    assert "task number 3" in result.output


def test_history_shows_in_progress_record_with_no_time_or_pr(tmp_path):
    store = TaskStateStore(tmp_path)
    store.start_task("a task that crashed mid-run", is_revision=False)
    store.close()

    result = CliRunner().invoke(cli.cli, ["history", "--repo", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "a task that crashed mid-run" in result.output
    assert "in_progress" in result.output


def test_stats_with_no_records(tmp_path):
    result = CliRunner().invoke(cli.cli, ["stats", "--repo", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "No task history yet" in result.output


def test_stats_computes_rates_from_seeded_data(tmp_path):
    store = TaskStateStore(tmp_path)

    for _ in range(3):
        task_id = store.start_task("succeeds", is_revision=False)
        store.finish_task(task_id, outcome=OUTCOME_SUCCESS, total_attempts=1)

    task_id = store.start_task("needs help", is_revision=False)
    store.finish_task(task_id, outcome=OUTCOME_NEEDS_HUMAN_HELP, total_attempts=3)

    task_id = store.start_task("declined", is_revision=False)
    store.finish_task(task_id, outcome=OUTCOME_FAILED, total_attempts=0)
    store.close()

    result = CliRunner().invoke(cli.cli, ["stats", "--repo", str(tmp_path)])

    assert result.exit_code == 0, result.output
    output = result.output
    assert "Tasks recorded: 5" in output
    assert "success:           3" in output
    assert "needs_human_help:  1" in output
    assert "failed:            1" in output
    assert "60.0%" in output  # resolution rate: 3/5
    assert "20.0%" in output  # retry-exhaustion rate: 1/5
