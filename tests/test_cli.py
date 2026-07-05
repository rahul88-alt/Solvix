from click.testing import CliRunner

import cli
from config import SolvixConfig
from context.assembler import FileScore, RetrievalResult
from execution.orchestrator import DangerousOpsCheck, StepResult, TaskResult
from execution.test_runner import TestResult
from indexer.pipeline import IndexResult
from reasoning.editor import Diff
from reasoning.llm_client import OllamaUnavailableError
from reasoning.planner import Plan, PlanStep
from reasoning.task_input import TaskContext
from review.pr_builder import PullRequestResult
from review.pr_feedback import PRFeedback


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


def _patch_common(
    monkeypatch, *, plan=None, run_task_fn=None, apply_fn=None, build_pr_fn=None, check_ambiguity_fn=None
):
    monkeypatch.setattr(cli, "load_config", lambda repo_root: SolvixConfig())
    monkeypatch.setattr(cli, "ensure_docker_available", lambda: None)
    monkeypatch.setattr(cli, "ensure_ollama_available", lambda: None)
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
    monkeypatch.setattr(cli, "check_ambiguity", check_ambiguity_fn or (lambda *a, **k: None))
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
            lambda repo_root, branch_name, task_context, plan, step_results, clarification=None: PullRequestResult(
                url="https://github.com/acme/repo/pull/1", title="t", body="b"
            )
        ),
    )


def test_run_normal_no_approval_needed(monkeypatch, tmp_path):
    calls = []

    def fake_run_task(plan, context, repo_root=None, config=None, confirm_dangerous_ops=None, on_progress=None):
        calls.append((plan, confirm_dangerous_ops))
        return _successful_task_result()

    _patch_common(monkeypatch, run_task_fn=fake_run_task)

    result = CliRunner().invoke(cli.cli, ["run", "add a palindrome check", "--repo", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "Task completed successfully." in result.output
    assert len(calls) == 1


def test_run_prints_live_phase_progress_in_order_clean_run(monkeypatch, tmp_path):
    def fake_run_task(plan, context, repo_root=None, config=None, confirm_dangerous_ops=None, on_progress=None):
        on_progress("Step 1: proposing changes to utils/strings.py (attempt 1/3)...")
        on_progress("Step 1: lint clean (attempt 1)")
        on_progress("Step 1: running tests (attempt 1)...")
        on_progress("Step 1: tests passed (attempt 1)")
        return _successful_task_result()

    _patch_common(monkeypatch, run_task_fn=fake_run_task)

    result = CliRunner().invoke(cli.cli, ["run", "add a palindrome check", "--repo", str(tmp_path)])

    assert result.exit_code == 0, result.output
    output = result.output
    markers = [
        "Indexing",
        "Indexed",
        "Finding relevant files for: add a palindrome check...",
        "Relevant files:",
        "Generating plan...",
        "Plan:",
        "Executing plan...",
        "Step 1: proposing changes to utils/strings.py (attempt 1/3)...",
        "Step 1: lint clean (attempt 1)",
        "Step 1: running tests (attempt 1)...",
        "Step 1: tests passed (attempt 1)",
        "Task completed successfully.",
    ]
    for marker in markers:
        assert marker in output, f"missing marker: {marker!r}"

    positions = [output.index(marker) for marker in markers]
    assert positions == sorted(positions), "phase markers appeared out of order"


def test_run_prints_live_phase_progress_in_order_with_retry(monkeypatch, tmp_path):
    def fake_run_task(plan, context, repo_root=None, config=None, confirm_dangerous_ops=None, on_progress=None):
        on_progress("Step 1: proposing changes to utils/strings.py (attempt 1/3)...")
        on_progress("Step 1: lint clean (attempt 1)")
        on_progress("Step 1: running tests (attempt 1)...")
        on_progress("Step 1: tests failed (attempt 1)")
        on_progress("Step 1: proposing changes to utils/strings.py (attempt 2/3)...")
        on_progress("Step 1: lint clean (attempt 2)")
        on_progress("Step 1: running tests (attempt 2)...")
        on_progress("Step 1: tests passed (attempt 2)")
        return _successful_task_result()

    _patch_common(monkeypatch, run_task_fn=fake_run_task)

    result = CliRunner().invoke(cli.cli, ["run", "add a palindrome check", "--repo", str(tmp_path)])

    assert result.exit_code == 0, result.output
    output = result.output
    markers = [
        "Executing plan...",
        "Step 1: proposing changes to utils/strings.py (attempt 1/3)...",
        "Step 1: running tests (attempt 1)...",
        "Step 1: tests failed (attempt 1)",
        "Step 1: proposing changes to utils/strings.py (attempt 2/3)...",
        "Step 1: running tests (attempt 2)...",
        "Step 1: tests passed (attempt 2)",
        "Task completed successfully.",
    ]
    for marker in markers:
        assert marker in output, f"missing marker: {marker!r}"

    positions = [output.index(marker) for marker in markers]
    assert positions == sorted(positions), "phase markers appeared out of order"


def test_run_plan_requires_approval_confirmed_yes(monkeypatch, tmp_path):
    calls = []

    def fake_run_task(plan, context, repo_root=None, config=None, confirm_dangerous_ops=None, on_progress=None):
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

    def fake_run_task(plan, context, repo_root=None, config=None, confirm_dangerous_ops=None, on_progress=None):
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

    def fake_run_task(plan, context, repo_root=None, config=None, confirm_dangerous_ops=None, on_progress=None):
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


def _feedback(
    branch="solvix/my-task",
    comment_body="please also handle the empty-string case",
    url="https://github.com/acme/repo/pull/42",
):
    return PRFeedback(branch=branch, comment_body=comment_body, url=url)


def _patch_revise_common(
    monkeypatch,
    *,
    plan=None,
    run_task_fn=None,
    feedback=None,
    fetch_feedback_fn=None,
    original_branch="feature/original",
    checkout_existing_fn=None,
    commit_fn=None,
    push_fn=None,
    post_comment_fn=None,
    checkout_calls=None,
):
    _patch_common(monkeypatch, plan=plan, run_task_fn=run_task_fn)

    checkout_calls = checkout_calls if checkout_calls is not None else []

    monkeypatch.setattr(
        cli, "fetch_pr_feedback", fetch_feedback_fn or (lambda repo_root, pr_number: feedback or _feedback())
    )
    monkeypatch.setattr(cli, "get_current_branch", lambda repo_root: original_branch)
    monkeypatch.setattr(
        cli,
        "checkout_existing_branch",
        checkout_existing_fn or (lambda repo_root, branch_name: checkout_calls.append(("checkout_existing", branch_name))),
    )
    monkeypatch.setattr(
        cli,
        "checkout_branch",
        lambda repo_root, branch_name, check=True: checkout_calls.append(("checkout_branch", branch_name)),
    )
    monkeypatch.setattr(
        cli,
        "commit_to_current_branch",
        commit_fn or (lambda repo_root, diff, commit_message: checkout_calls.append(("commit", commit_message))),
    )
    monkeypatch.setattr(
        cli, "push_branch", push_fn or (lambda repo_root, branch_name: checkout_calls.append(("push", branch_name)))
    )
    monkeypatch.setattr(
        cli,
        "post_pr_comment",
        post_comment_fn or (lambda repo_root, pr_number, body: checkout_calls.append(("comment", pr_number, body))),
    )
    return checkout_calls


def test_revise_success_pushes_commit_and_comments_then_restores_branch(monkeypatch, tmp_path):
    def fake_run_task(plan, context, repo_root=None, config=None, confirm_dangerous_ops=None, on_progress=None):
        return _successful_task_result()

    calls = _patch_revise_common(monkeypatch, run_task_fn=fake_run_task)

    result = CliRunner().invoke(cli.cli, ["revise", "42", "--repo", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "please also handle the empty-string case" in result.output
    assert ("checkout_existing", "solvix/my-task") in calls
    assert any(c[0] == "commit" for c in calls)
    assert ("push", "solvix/my-task") in calls
    assert any(c[0] == "comment" for c in calls)
    # original branch restored last
    assert calls[-1] == ("checkout_branch", "feature/original")


def test_revise_failure_posts_comment_without_pushing_and_restores_branch(monkeypatch, tmp_path):
    def fake_run_task(plan, context, repo_root=None, config=None, confirm_dangerous_ops=None, on_progress=None):
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
            culprit_step=plan.steps[0],
            reason="exhausted its own per-step retry budget",
        )

    calls = _patch_revise_common(monkeypatch, run_task_fn=fake_run_task)

    result = CliRunner().invoke(cli.cli, ["revise", "42", "--repo", str(tmp_path)])

    assert result.exit_code == 1
    assert not any(c[0] == "push" for c in calls)
    assert not any(c[0] == "commit" for c in calls)
    assert any(c[0] == "comment" for c in calls)
    assert calls[-1] == ("checkout_branch", "feature/original")


def test_revise_restores_branch_even_when_checkout_fails(monkeypatch, tmp_path):
    from execution.patch_applier import PatchApplyError

    def raise_checkout(repo_root, branch_name):
        raise PatchApplyError("branch does not exist on origin")

    calls = _patch_revise_common(monkeypatch, checkout_existing_fn=raise_checkout)

    result = CliRunner().invoke(cli.cli, ["revise", "42", "--repo", str(tmp_path)])

    assert result.exit_code != 0
    # never entered the pipeline/commit/push/comment steps
    assert not any(c[0] in ("commit", "push", "comment") for c in calls)
    # a failed checkout never reached the try/finally that restores the
    # branch (nothing to restore -- checkout_existing_branch never switched
    # anything), so checkout_branch should not have been called either
    assert not any(c[0] == "checkout_branch" for c in calls)


def test_revise_fetch_feedback_error_reported_cleanly(monkeypatch, tmp_path):
    from review.pr_feedback import PRFeedbackError

    def raise_fetch(repo_root, pr_number):
        raise PRFeedbackError(f"PR #{pr_number} has no human feedback comment to revise from")

    _patch_revise_common(monkeypatch, fetch_feedback_fn=raise_fetch)

    result = CliRunner().invoke(cli.cli, ["revise", "42", "--repo", str(tmp_path)])

    assert result.exit_code != 0
    assert "no human feedback comment" in result.output


def test_run_ollama_unavailable_reported_cleanly_not_as_traceback(monkeypatch, tmp_path):
    """SLX-F4: Ollama not running must fail fast with a clean message and a
    non-zero exit code, mirroring how a Docker-unavailable preflight failure
    is already reported -- never a raw ConnectionError traceback.
    """
    _patch_common(monkeypatch)

    def raise_unavailable():
        raise OllamaUnavailableError(
            "Ollama is not available at http://localhost:11434. Start it with: ollama serve"
        )

    monkeypatch.setattr(cli, "ensure_ollama_available", raise_unavailable)

    result = CliRunner().invoke(cli.cli, ["run", "add a palindrome check", "--repo", str(tmp_path)])

    assert result.exit_code != 0
    assert result.exc_info[0] is not OllamaUnavailableError
    assert "Ollama is not available" in result.output
    assert "ollama serve" in result.output


def test_revise_ollama_unavailable_reported_cleanly_not_as_traceback(monkeypatch, tmp_path):
    _patch_revise_common(monkeypatch)

    def raise_unavailable():
        raise OllamaUnavailableError(
            "Ollama is not available at http://localhost:11434. Start it with: ollama serve"
        )

    monkeypatch.setattr(cli, "ensure_ollama_available", raise_unavailable)

    result = CliRunner().invoke(cli.cli, ["revise", "42", "--repo", str(tmp_path)])

    assert result.exit_code != 0
    assert result.exc_info[0] is not OllamaUnavailableError
    assert "Ollama is not available" in result.output
    assert "ollama serve" in result.output


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

    def fake_build_pr(repo_root, branch_name, task_context, plan, step_results, clarification=None):
        pr_calls.append((repo_root, branch_name, task_context, plan, step_results, clarification))
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


def test_run_no_clarify_flag_skips_ambiguity_check_entirely(monkeypatch, tmp_path):
    calls = []

    _patch_common(
        monkeypatch,
        run_task_fn=lambda *a, **k: _successful_task_result(),
        check_ambiguity_fn=lambda *a, **k: calls.append(1) or "should not be called",
    )

    result = CliRunner().invoke(
        cli.cli, ["run", "add a palindrome check", "--repo", str(tmp_path), "--no-clarify"]
    )

    assert result.exit_code == 0, result.output
    assert calls == []
    assert "ambiguous" not in result.output.lower()


def test_run_clear_task_does_not_prompt_for_clarification(monkeypatch, tmp_path):
    _patch_common(
        monkeypatch,
        run_task_fn=lambda *a, **k: _successful_task_result(),
        check_ambiguity_fn=lambda *a, **k: None,
    )

    result = CliRunner().invoke(cli.cli, ["run", "add a palindrome check", "--repo", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "ambiguous" not in result.output.lower()
    assert "Task completed successfully." in result.output


def test_run_ambiguous_task_prompts_and_incorporates_answer_into_plan(monkeypatch, tmp_path):
    plan_calls = []

    def fake_generate_plan(task_context, **kwargs):
        plan_calls.append(task_context)
        return _plan()

    def fake_run_task(plan, context, repo_root=None, config=None, confirm_dangerous_ops=None, on_progress=None):
        return _successful_task_result()

    _patch_common(
        monkeypatch,
        run_task_fn=fake_run_task,
        check_ambiguity_fn=lambda task_context, *a, **k: (
            "Which aspect of the calculator should be improved -- error handling "
            "or new operations?"
        ),
    )
    monkeypatch.setattr(cli, "generate_plan", fake_generate_plan)

    result = CliRunner().invoke(
        cli.cli,
        ["run", "improve the calculator", "--repo", str(tmp_path)],
        input="add support for exponentiation\n",
    )

    assert result.exit_code == 0, result.output
    assert "This task looks ambiguous" in result.output
    assert "Which aspect of the calculator should be improved" in result.output

    assert len(plan_calls) == 1
    clarified_task = plan_calls[0].task
    assert "add a palindrome check" in clarified_task  # original task text, from stubbed build_task_context
    assert "Which aspect of the calculator should be improved" in clarified_task
    assert "add support for exponentiation" in clarified_task


def test_run_ambiguous_task_clarification_passed_to_build_pr(monkeypatch, tmp_path):
    pr_calls = []

    def fake_build_pr(repo_root, branch_name, task_context, plan, step_results, clarification=None):
        pr_calls.append(clarification)
        return PullRequestResult(url="https://github.com/acme/repo/pull/9", title="t", body="b")

    _patch_common(
        monkeypatch,
        run_task_fn=lambda *a, **k: _successful_task_result(),
        build_pr_fn=fake_build_pr,
        check_ambiguity_fn=lambda task_context, *a, **k: "Which aspect should be improved?",
    )

    result = CliRunner().invoke(
        cli.cli,
        ["run", "improve the calculator", "--repo", str(tmp_path)],
        input="add exponentiation support\n",
    )

    assert result.exit_code == 0, result.output
    assert len(pr_calls) == 1
    clarification = pr_calls[0]
    assert clarification is not None
    assert clarification.question == "Which aspect should be improved?"
    assert clarification.answer == "add exponentiation support"


def test_run_clear_task_passes_no_clarification_to_build_pr(monkeypatch, tmp_path):
    pr_calls = []

    def fake_build_pr(repo_root, branch_name, task_context, plan, step_results, clarification=None):
        pr_calls.append(clarification)
        return PullRequestResult(url="https://github.com/acme/repo/pull/10", title="t", body="b")

    _patch_common(
        monkeypatch,
        run_task_fn=lambda *a, **k: _successful_task_result(),
        build_pr_fn=fake_build_pr,
        check_ambiguity_fn=lambda *a, **k: None,
    )

    result = CliRunner().invoke(cli.cli, ["run", "add a palindrome check", "--repo", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert pr_calls == [None]
