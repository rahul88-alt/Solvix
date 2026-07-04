from unittest.mock import patch

import pytest

from context.assembler import RetrievalResult
from execution.orchestrator import StepResult, TaskResult
from execution.test_runner import TestResult
from reasoning.editor import Diff
from reasoning.planner import Clarification, Plan, PlanStep
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


def _plan(requires_approval=False, approval_reasons=()):
    return Plan(
        steps=[PlanStep(file="widget.py", description="add widget function")],
        requires_approval=requires_approval,
        approval_reasons=approval_reasons,
    )


def _diff(target_file="widget.py"):
    return Diff(target_file=target_file, diff_text=f"diff --git a/{target_file} b/{target_file}\n", is_new_file=False)


def _clean_task_result():
    """A single step, one attempt, no confirmations, no issues."""
    step_result = StepResult(
        success=True,
        diff=_diff(),
        test_result=TestResult(passed=True, output="1 passed", exit_code=0),
        attempts=1,
        needs_human_help=False,
    )
    return TaskResult(
        success=True,
        needs_human_help=False,
        step_results=(step_result,),
        total_attempts=1,
    )


def _retried_and_confirmed_task_result():
    """A single step that took 2 attempts and had a dangerous-ops
    confirmation approved along the way."""
    step_result = StepResult(
        success=True,
        diff=_diff(),
        test_result=TestResult(passed=True, output="2 passed", exit_code=0),
        attempts=2,
        needs_human_help=False,
        dangerous_ops_confirmed=True,
        dangerous_ops_confirmed_reasons=("diff matches dangerous-ops pattern: rm -rf",),
        attempt_failures=("attempt 1: AssertionError: expected 2, got 1",),
    )
    return TaskResult(
        success=True,
        needs_human_help=False,
        step_results=(step_result,),
        total_attempts=2,
    )


def test_build_title_truncates_long_task():
    task_context = _task_context("a" * 100)
    title = build_title(task_context)
    assert title.startswith("solvix: ")
    assert len(title) <= 72


def test_build_body_includes_plan_files_and_test_results():
    body = build_body(_task_context(), _plan(), _clean_task_result())
    assert "widget.py" in body
    assert "add widget function" in body
    assert "passed" in body


def test_build_body_clean_case_shows_no_flags():
    task_context = _task_context("add a widget")
    body = build_body(task_context, _plan(), _clean_task_result())

    assert "## Task" in body
    assert "add a widget" in body
    assert "## Plan" in body
    assert "## Key decisions" in body
    assert "not flagged for approval" in body.lower()
    assert "## Test results" in body
    assert "1 attempt" in body
    assert "## Needs attention" not in body
    assert "declined" not in body.lower()
    assert "approved" not in body.lower()


def test_build_body_includes_clarification_in_key_decisions_when_present():
    clarification = Clarification(
        question="Which aspect of the calculator should be improved?",
        answer="add support for exponentiation",
    )
    body = build_body(_task_context(), _plan(), _clean_task_result(), clarification)

    assert "## Key decisions" in body
    assert "Clarification requested" in body
    assert "Which aspect of the calculator should be improved?" in body
    assert "add support for exponentiation" in body


def test_build_body_omits_clarification_line_when_none():
    body = build_body(_task_context(), _plan(), _clean_task_result())

    assert "Clarification requested" not in body


def test_build_body_retried_and_confirmed_case_shows_both():
    plan = _plan(requires_approval=True, approval_reasons=("touches sensitive path(s): widget.py",))
    body = build_body(_task_context(), plan, _retried_and_confirmed_task_result())

    assert "flagged for approval" in body.lower()
    assert "touches sensitive path(s)" in body

    assert "dangerous-ops confirmation was required and approved" in body
    assert "rm -rf" in body

    assert "needed 2 attempts" in body
    assert "AssertionError" in body


def test_build_body_surfaces_needs_human_help_step_even_if_task_succeeded():
    flagged_step = StepResult(
        success=False,
        diff=_diff("other.py"),
        test_result=TestResult(passed=False, output="1 failed", exit_code=1),
        attempts=3,
        needs_human_help=True,
        failure_reason="exhausted its own per-step retry budget",
    )
    ok_step = StepResult(
        success=True,
        diff=_diff("widget.py"),
        test_result=TestResult(passed=True, output="1 passed", exit_code=0),
        attempts=1,
        needs_human_help=False,
    )
    task_result = TaskResult(
        success=True,
        needs_human_help=False,
        step_results=(flagged_step, ok_step),
        total_attempts=4,
    )

    body = build_body(_task_context(), _plan(), task_result)

    assert "## Needs attention" in body
    assert "other.py" in body
    assert "exhausted its own per-step retry budget" in body


def test_build_body_surfaces_assertion_gaming_even_on_a_successful_task():
    gamed_step = StepResult(
        success=True,
        diff=_diff("test_calc.py"),
        test_result=TestResult(passed=True, output="1 passed", exit_code=0),
        attempts=2,
        needs_human_help=False,
        assertion_gaming_suspected=True,
        assertion_gaming_details=(
            "attempt 2: assertion's expected value was changed from '0.3' to "
            "'0.30000000000000004' -- '0.30000000000000004' matches the previous "
            "attempt's actual (failing) output"
        ),
    )
    task_result = TaskResult(
        success=True,
        needs_human_help=False,
        step_results=(gamed_step,),
        total_attempts=2,
    )

    body = build_body(_task_context(), _plan(), task_result)

    assert "## Needs attention" in body
    assert "suspected assertion-gaming" in body
    assert "test_calc.py" in body
    assert "0.3" in body
    assert "0.30000000000000004" in body


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
        result = build_pr(tmp_path, "solvix/my-task", _task_context(), _plan(), _clean_task_result())

    assert result.url == "https://github.com/acme/repo/pull/42"
    assert calls[0] == ["git", "push", "-u", "origin", "solvix/my-task"]
    assert calls[1][:3] == ["gh", "pr", "create"]
    assert "--head" in calls[1]
    assert "solvix/my-task" in calls[1]


def test_build_pr_raises_when_push_fails(tmp_path):
    with patch("review.pr_builder.subprocess.run", return_value=_completed(1, stderr="rejected")):
        with pytest.raises(PRBuildError):
            build_pr(tmp_path, "solvix/my-task", _task_context(), _plan(), _clean_task_result())


def test_build_pr_raises_when_gh_create_fails(tmp_path):
    def fake_run(args, **kwargs):
        if args[:2] == ["git", "push"]:
            return _completed(0)
        return _completed(1, stderr="gh not authenticated")

    with patch("review.pr_builder.subprocess.run", side_effect=fake_run):
        with pytest.raises(PRBuildError):
            build_pr(tmp_path, "solvix/my-task", _task_context(), _plan(), _clean_task_result())
