import json
from unittest.mock import patch

import pytest

from execution.orchestrator import StepResult, TaskResult
from execution.test_runner import TestResult
from reasoning.editor import Diff
from reasoning.planner import Plan, PlanStep
from review.pr_builder import SOLVIX_COMMENT_MARKER
from review.pr_feedback import (
    PRFeedback,
    PRFeedbackError,
    build_revision_comment,
    fetch_pr_feedback,
    post_pr_comment,
)


def _completed(returncode=0, stdout="", stderr=""):
    class _Result:
        def __init__(self):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    return _Result()


def _plan():
    return Plan(
        steps=[PlanStep(file="widget.py", description="tweak widget per feedback")],
        requires_approval=False,
        approval_reasons=(),
    )


def _diff(target_file="widget.py"):
    return Diff(target_file=target_file, diff_text=f"diff --git a/{target_file} b/{target_file}\n", is_new_file=False)


def _successful_task_result():
    step_result = StepResult(
        success=True,
        diff=_diff(),
        test_result=TestResult(passed=True, output="1 passed", exit_code=0),
        attempts=1,
        needs_human_help=False,
    )
    return TaskResult(success=True, needs_human_help=False, step_results=(step_result,), total_attempts=1)


def _failed_task_result():
    step_result = StepResult(
        success=False,
        diff=_diff(),
        test_result=TestResult(passed=False, output="1 failed", exit_code=1),
        attempts=3,
        needs_human_help=True,
        failure_reason="exhausted its own per-step retry budget",
    )
    return TaskResult(
        success=False,
        needs_human_help=True,
        step_results=(step_result,),
        total_attempts=3,
        culprit_step=PlanStep(file="widget.py", description="tweak widget per feedback"),
        reason="exhausted its own per-step retry budget",
    )


def _gh_pr_view_stdout(branch="solvix/my-task", comments=None, url="https://github.com/acme/repo/pull/42"):
    return json.dumps(
        {"headRefName": branch, "comments": comments if comments is not None else [], "url": url}
    )


def test_fetch_pr_feedback_returns_branch_and_latest_human_comment(tmp_path):
    comments = [
        {"body": "looks good overall", "author": {"login": "reviewer1"}},
        {"body": "please also handle the empty-string case", "author": {"login": "reviewer1"}},
    ]

    def fake_run(args, **kwargs):
        assert args == ["gh", "pr", "view", "42", "--json", "headRefName,comments,url"]
        return _completed(0, stdout=_gh_pr_view_stdout(comments=comments))

    with patch("review.pr_feedback.subprocess.run", side_effect=fake_run):
        feedback = fetch_pr_feedback(tmp_path, 42)

    assert feedback.branch == "solvix/my-task"
    assert feedback.comment_body == "please also handle the empty-string case"


def test_fetch_pr_feedback_skips_solvix_own_comments(tmp_path):
    comments = [
        {"body": "please also handle the empty-string case", "author": {"login": "reviewer1"}},
        {"body": f"## Outcome\nDone.\n\n---\n{SOLVIX_COMMENT_MARKER}\n", "author": {"login": "reviewer1"}},
    ]

    def fake_run(args, **kwargs):
        return _completed(0, stdout=_gh_pr_view_stdout(comments=comments))

    with patch("review.pr_feedback.subprocess.run", side_effect=fake_run):
        feedback = fetch_pr_feedback(tmp_path, 42)

    assert feedback.comment_body == "please also handle the empty-string case"


def test_fetch_pr_feedback_raises_when_no_human_comment(tmp_path):
    comments = [{"body": f"round 1 done\n\n---\n{SOLVIX_COMMENT_MARKER}\n", "author": {"login": "reviewer1"}}]

    def fake_run(args, **kwargs):
        return _completed(0, stdout=_gh_pr_view_stdout(comments=comments))

    with patch("review.pr_feedback.subprocess.run", side_effect=fake_run):
        with pytest.raises(PRFeedbackError):
            fetch_pr_feedback(tmp_path, 42)


def test_fetch_pr_feedback_raises_when_gh_fails(tmp_path):
    with patch("review.pr_feedback.subprocess.run", return_value=_completed(1, stderr="not found")):
        with pytest.raises(PRFeedbackError):
            fetch_pr_feedback(tmp_path, 42)


def test_fetch_pr_feedback_raises_on_bad_json(tmp_path):
    with patch("review.pr_feedback.subprocess.run", return_value=_completed(0, stdout="not json")):
        with pytest.raises(PRFeedbackError):
            fetch_pr_feedback(tmp_path, 42)


def test_post_pr_comment_invokes_gh_pr_comment(tmp_path):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return _completed(0)

    with patch("review.pr_feedback.subprocess.run", side_effect=fake_run):
        post_pr_comment(tmp_path, 42, "hello")

    assert calls == [["gh", "pr", "comment", "42", "--body", "hello"]]


def test_post_pr_comment_raises_when_gh_fails(tmp_path):
    with patch("review.pr_feedback.subprocess.run", return_value=_completed(1, stderr="boom")):
        with pytest.raises(PRFeedbackError):
            post_pr_comment(tmp_path, 42, "hello")


def test_build_revision_comment_success_case():
    body = build_revision_comment(_plan(), _successful_task_result(), "please handle the empty-string case")

    assert "## Feedback addressed in this revision" in body
    assert "please handle the empty-string case" in body
    assert "## Outcome" in body
    assert "pushed as a new commit" in body
    assert "widget.py" in body
    assert "## Needs attention" not in body
    assert body.strip().endswith(SOLVIX_COMMENT_MARKER)


def test_build_revision_comment_failure_case():
    body = build_revision_comment(_plan(), _failed_task_result(), "please handle the empty-string case")

    assert "## Outcome" in body
    assert "needs human help" in body.lower()
    assert "exhausted its own per-step retry budget" in body
    assert SOLVIX_COMMENT_MARKER in body
