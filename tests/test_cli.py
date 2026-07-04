from click.testing import CliRunner

import cli
from config import SolvixConfig
from context.assembler import FileScore, RetrievalResult
from execution.orchestrator import DangerousOpsCheck, StepResult, TaskResult
from execution.test_runner import TestResult
from indexer.pipeline import IndexResult
from reasoning.editor import Diff
from reasoning.planner import Plan, PlanStep
from reasoning.task_input import TaskContext
from review.pr_builder import PullRequestResult


def _task_context():
    retrieval = RetrievalResult(
        files=[FileScore(file_path="utils/strings.py", score=1.0, reasons=("test",))],
        related_files=[],
    )
    return TaskContext(task="add a palindrome check", retrieval=retrieval)


def _plan(requires_approval=False, approval_reasons=()):
    return Plan(
        steps=[PlanStep(file="utils/strings.py", description="add is_palindrome")],
        requires_approval=requires_approval,
        approval_reasons=approval_reasons,
    )


def _diff():
    return Diff(
        target_file="utils/strings.py",
        diff_text="--- a/utils/strings.py\n+++ b/utils/strings.py\n@@ -1,1 +1,2 @@\n def slugify(text):\n+    pass\n",
        is_new_file=False,
    )


def _successful_task_result():
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


def _patch_common(monkeypatch, *, plan=None, run_task_fn=None, apply_fn=None, build_pr_fn=None):
    monkeypatch.setattr(cli, "load_config", lambda repo_root: SolvixConfig())
    monkeypatch.setattr(cli, "ensure_docker_available", lambda: None)
    monkeypatch.setattr(cli, "reap_orphans", lambda: [])
    monkeypatch.setattr(
        cli,
        "index_repo",
        lambda repo_path: IndexResult(
            repo_root=repo_path, num_files_indexed=3, num_chunks=10, symbol_index=None, vector_store=None
        ),
    )
    monkeypatch.setattr(cli, "get_default_embedder", lambda: object())
    monkeypatch.setattr(cli, "build_task_context", lambda *a, **k: _task_context())
    monkeypatch.setattr(cli, "generate_plan", lambda *a, **k: plan or _plan())
    if run_task_fn is not None:
        monkeypatch.setattr(cli, "run_task", run_task_fn)
    monkeypatch.setattr(
        cli,
        "apply_to_new_branch",
        apply_fn or (lambda repo_root, diff, branch_name, commit_message: branch_name),
    )
    monkeypatch.setattr(
        cli,
        "build_pr",
        build_pr_fn
        or (
            lambda repo_root, branch_name, task_context, plan, step_results: PullRequestResult(
                url="https://github.com/acme/repo/pull/1", title="t", body="b"
            )
        ),
    )


def test_run_normal_no_approval_needed(monkeypatch, tmp_path):
    calls = []

    def fake_run_task(plan, context, repo_root=None, config=None, confirm_dangerous_ops=None):
        calls.append((plan, confirm_dangerous_ops))
        return _successful_task_result()

    _patch_common(monkeypatch, run_task_fn=fake_run_task)

    result = CliRunner().invoke(cli.cli, ["run", "add a palindrome check", "--repo", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "Task completed successfully." in result.output
    assert len(calls) == 1


def test_run_plan_requires_approval_confirmed_yes(monkeypatch, tmp_path):
    calls = []

    def fake_run_task(plan, context, repo_root=None, config=None, confirm_dangerous_ops=None):
        calls.append(plan)
        return _successful_task_result()

    _patch_common(
        monkeypatch,
        plan=_plan(requires_approval=True, approval_reasons=("touches sensitive path(s): auth/login.py",)),
        run_task_fn=fake_run_task,
    )

    result = CliRunner().invoke(
        cli.cli, ["run", "add a palindrome check", "--repo", str(tmp_path)], input="y\n"
    )

    assert result.exit_code == 0, result.output
    assert "requires approval" in result.output
    assert "touches sensitive path(s): auth/login.py" in result.output
    assert "Task completed successfully." in result.output
    assert len(calls) == 1


def test_run_plan_requires_approval_declined_no_aborts_cleanly(monkeypatch, tmp_path):
    calls = []

    def fake_run_task(*a, **k):
        calls.append(True)
        return _successful_task_result()

    _patch_common(
        monkeypatch,
        plan=_plan(requires_approval=True, approval_reasons=("touches 5 files (limit 3)",)),
        run_task_fn=fake_run_task,
    )

    result = CliRunner().invoke(
        cli.cli, ["run", "add a palindrome check", "--repo", str(tmp_path)], input="n\n"
    )

    assert result.exit_code == 1
    assert "Aborted: plan was not approved." in result.output
    assert calls == []  # run_task must never be invoked


def test_run_dangerous_ops_confirmed_yes_proceeds(monkeypatch, tmp_path):
    check = DangerousOpsCheck(
        requires_confirmation=True,
        reasons=("diff matches dangerous-ops pattern: git\\s+push\\b.*--force\\b",),
    )

    def fake_run_task(plan, context, repo_root=None, config=None, confirm_dangerous_ops=None):
        approved = confirm_dangerous_ops(check)
        assert approved is True
        return _successful_task_result()

    _patch_common(monkeypatch, run_task_fn=fake_run_task)

    result = CliRunner().invoke(
        cli.cli, ["run", "add a palindrome check", "--repo", str(tmp_path)], input="y\n"
    )

    assert result.exit_code == 0, result.output
    assert "flagged as a dangerous operation" in result.output
    assert "git\\s+push" in result.output
    assert "Task completed successfully." in result.output


def test_run_dangerous_ops_declined_no_stops_step(monkeypatch, tmp_path):
    check = DangerousOpsCheck(
        requires_confirmation=True,
        reasons=("diff matches dangerous-ops pattern: DROP TABLE",),
    )

    def fake_run_task(plan, context, repo_root=None, config=None, confirm_dangerous_ops=None):
        approved = confirm_dangerous_ops(check)
        assert approved is False
        step_result = StepResult(
            success=False,
            diff=_diff(),
            test_result=None,
            attempts=1,
            needs_human_help=False,
            requires_confirmation=True,
            confirmation_reasons=check.reasons,
        )
        return TaskResult(
            success=False,
            needs_human_help=False,
            step_results=(step_result,),
            total_attempts=1,
            culprit_step=plan.steps[0],
            reason="step 'utils/strings.py' was not applied because dangerous-ops confirmation was declined",
        )

    _patch_common(monkeypatch, run_task_fn=fake_run_task)

    result = CliRunner().invoke(
        cli.cli, ["run", "add a palindrome check", "--repo", str(tmp_path)], input="n\n"
    )

    assert result.exit_code == 1
    assert "flagged as a dangerous operation" in result.output
    assert "Task did not complete successfully." in result.output
    assert "confirmation was declined" in result.output


def test_run_unhandled_pipeline_exception_reported_cleanly_not_as_traceback(monkeypatch, tmp_path):
    def fake_run_task(*a, **k):
        raise RuntimeError("diff did not apply cleanly against file content: boom")

    _patch_common(monkeypatch, run_task_fn=fake_run_task)

    result = CliRunner().invoke(cli.cli, ["run", "add a palindrome check", "--repo", str(tmp_path)])

    assert result.exit_code == 1
    assert result.exc_info[0] is SystemExit  # not an unhandled RuntimeError
    assert "This task needs human help." in result.output
    assert "unhandled error while executing the plan" in result.output
    assert "boom" in result.output


def test_run_needs_human_help_prints_reason_and_step_results(monkeypatch, tmp_path):
    step_result = StepResult(
        success=False,
        diff=_diff(),
        test_result=TestResult(passed=False, output="1 failed", exit_code=1),
        attempts=3,
        needs_human_help=True,
    )
    task_result = TaskResult(
        success=False,
        needs_human_help=True,
        step_results=(step_result,),
        total_attempts=3,
        culprit_step=PlanStep(file="utils/strings.py", description="add is_palindrome"),
        reason="step 'utils/strings.py' exhausted its own per-step retry budget (3 attempt(s)) and needs human help",
    )

    _patch_common(monkeypatch, run_task_fn=lambda *a, **k: task_result)

    result = CliRunner().invoke(cli.cli, ["run", "add a palindrome check", "--repo", str(tmp_path)])

    assert result.exit_code == 1
    assert "Task did not complete successfully." in result.output
    assert "This task needs human help." in result.output
    assert "exhausted its own per-step retry budget" in result.output
    assert "Culprit step: utils/strings.py" in result.output
    assert "1 failed" in result.output


def test_run_success_delivers_pull_request(monkeypatch, tmp_path):
    apply_calls = []
    pr_calls = []

    def fake_apply(repo_root, diff, branch_name, commit_message):
        apply_calls.append((repo_root, diff, branch_name, commit_message))
        return branch_name

    def fake_build_pr(repo_root, branch_name, task_context, plan, step_results):
        pr_calls.append((repo_root, branch_name, task_context, plan, step_results))
        return PullRequestResult(url="https://github.com/acme/repo/pull/7", title="t", body="b")

    _patch_common(
        monkeypatch,
        run_task_fn=lambda *a, **k: _successful_task_result(),
        apply_fn=fake_apply,
        build_pr_fn=fake_build_pr,
    )

    result = CliRunner().invoke(cli.cli, ["run", "add a palindrome check", "--repo", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "Opened pull request: https://github.com/acme/repo/pull/7" in result.output
    assert len(apply_calls) == 1
    assert apply_calls[0][2] == "solvix/add-a-palindrome-check"
    assert len(pr_calls) == 1
    assert pr_calls[0][1] == "solvix/add-a-palindrome-check"


def test_run_failure_never_creates_branch_or_pr(monkeypatch, tmp_path):
    apply_calls = []
    pr_calls = []

    step_result = StepResult(
        success=False,
        diff=_diff(),
        test_result=TestResult(passed=False, output="1 failed", exit_code=1),
        attempts=3,
        needs_human_help=True,
    )
    task_result = TaskResult(
        success=False,
        needs_human_help=True,
        step_results=(step_result,),
        total_attempts=3,
        culprit_step=PlanStep(file="utils/strings.py", description="add is_palindrome"),
        reason="needs human help",
    )

    _patch_common(
        monkeypatch,
        run_task_fn=lambda *a, **k: task_result,
        apply_fn=lambda *a, **k: apply_calls.append(a) or "unused",
        build_pr_fn=lambda *a, **k: pr_calls.append(a) or PullRequestResult(url="", title="", body=""),
    )

    result = CliRunner().invoke(cli.cli, ["run", "add a palindrome check", "--repo", str(tmp_path)])

    assert result.exit_code == 1
    assert apply_calls == []
    assert pr_calls == []
    assert "Opened pull request" not in result.output


def test_run_success_pr_delivery_failure_reported_as_click_exception(monkeypatch, tmp_path):
    from execution.patch_applier import PatchApplyError

    def failing_apply(*a, **k):
        raise PatchApplyError("git apply failed: does not apply")

    _patch_common(
        monkeypatch,
        run_task_fn=lambda *a, **k: _successful_task_result(),
        apply_fn=failing_apply,
    )

    result = CliRunner().invoke(cli.cli, ["run", "add a palindrome check", "--repo", str(tmp_path)])

    assert result.exit_code != 0
    assert "failed to deliver change as a pull request" in result.output
