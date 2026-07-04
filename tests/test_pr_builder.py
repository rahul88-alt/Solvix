from unittest.mock import patch

import pytest

from context.assembler import RetrievalResult
from execution.orchestrator import StepResult
from execution.test_runner import TestResult
from reasoning.editor import Diff
from reasoning.planner import Plan, PlanStep
from reasoning.task_input import TaskContext
from review.pr_builder import PRBuildError, build_body, build_pr, build_title


def _completed(returncode=0, stdout="", stderr=""):
    class _Result:
        def __init__(self):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    return _Result()


def _task_context(task="add a widget"):
    return TaskContext(task=task, retrieval=RetrievalResult(files=[], related_files=[]))


def _plan():
    return Plan(
        steps=[PlanStep(file="widget.py", description="add widget function")],
        requires_approval=False,
        approval_reasons=(),
    )


def _step_results(passed=True):
    diff = Diff(target_file="widget.py", diff_text="diff --git a/widget.py b/widget.py\n", is_new_file=False)
    test_result = TestResult(passed=passed, output="1 passed" if passed else "1 failed", exit_code=0 if passed else 1)
    return (
        StepResult(
            success=passed,
            diff=diff,
            test_result=test_result,
            attempts=1,
            needs_human_help=False,
        ),
    )


def test_build_title_truncates_long_task():
    task_context = _task_context("a" * 100)
    title = build_title(task_context)
    assert title.startswith("solvix: ")
    assert len(title) <= 72


def test_build_body_includes_plan_files_and_test_results():
    body = build_body(_task_context(), _plan(), _step_results())
    assert "widget.py" in body
    assert "add widget function" in body
    assert "passed" in body


def test_build_pr_pushes_then_creates_pr_and_returns_url(tmp_path):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[:2] == ["git", "push"]:
            return _completed(0)
        if args[:3] == ["gh", "pr", "create"]:
            return _completed(0, stdout="https://github.com/acme/repo/pull/42\n")
        raise AssertionError(f"unexpected call: {args}")

    with patch("review.pr_builder.subprocess.run", side_effect=fake_run):
        result = build_pr(tmp_path, "solvix/my-task", _task_context(), _plan(), _step_results())

    assert result.url == "https://github.com/acme/repo/pull/42"
    assert calls[0] == ["git", "push", "-u", "origin", "solvix/my-task"]
    assert calls[1][:3] == ["gh", "pr", "create"]
    assert "--head" in calls[1]
    assert "solvix/my-task" in calls[1]


def test_build_pr_raises_when_push_fails(tmp_path):
    with patch("review.pr_builder.subprocess.run", return_value=_completed(1, stderr="rejected")):
        with pytest.raises(PRBuildError):
            build_pr(tmp_path, "solvix/my-task", _task_context(), _plan(), _step_results())


def test_build_pr_raises_when_gh_create_fails(tmp_path):
    def fake_run(args, **kwargs):
        if args[:2] == ["git", "push"]:
            return _completed(0)
        return _completed(1, stderr="gh not authenticated")

    with patch("review.pr_builder.subprocess.run", side_effect=fake_run):
        with pytest.raises(PRBuildError):
            build_pr(tmp_path, "solvix/my-task", _task_context(), _plan(), _step_results())
