from config import SolvixConfig
from context.assembler import FileScore, RetrievalResult
from execution.orchestrator import (
    DangerousOpsCheck,
    StepResult,
    check_dangerous_ops,
    execute_step_with_verification,
)
from execution.test_runner import TestResult
from reasoning.editor import Diff
from reasoning.planner import PlanStep
from reasoning.task_input import TaskContext

_ORIGINAL_CONTENT = "def subtract(a, b):\n    return a - b\n"

_NAIVE_DIFF = (
    "--- a/calc.py\n"
    "+++ b/calc.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def subtract(a, b):\n"
    "-    return a - b\n"
    "+    return a - b  # naive, still allows negatives\n"
)

_CLAMPED_DIFF = (
    "--- a/calc.py\n"
    "+++ b/calc.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def subtract(a, b):\n"
    "-    return a - b\n"
    "+    return max(a - b, 0)\n"
)


def _write_repo(tmp_path):
    (tmp_path / "calc.py").write_text(_ORIGINAL_CONTENT)
    (tmp_path / "test_calc.py").write_text(
        "from calc import subtract\n\n\n"
        "def test_subtract_never_negative():\n"
        "    assert subtract(2, 5) == 0\n"
    )
    return tmp_path


def _task_context():
    retrieval = RetrievalResult(
        files=[FileScore(file_path="calc.py", score=1.0, reasons=("test",))],
        related_files=[],
    )
    return TaskContext(task="subtract should never return negative", retrieval=retrieval)


def _step():
    return PlanStep(file="calc.py", description="clamp subtract result at zero")


def test_execute_step_passes_immediately_when_first_diff_passes_tests(tmp_path):
    _write_repo(tmp_path)
    calls = []

    def fake_complete(system, messages):
        calls.append(messages)
        return _CLAMPED_DIFF

    result = execute_step_with_verification(
        _step(), _ORIGINAL_CONTENT, _task_context(), complete_fn=fake_complete, repo_root=tmp_path
    )

    assert isinstance(result, StepResult)
    assert result.success is True
    assert result.attempts == 1
    assert result.needs_human_help is False
    assert len(calls) == 1


def test_execute_step_retries_with_failure_output_and_succeeds_on_second_attempt(tmp_path):
    _write_repo(tmp_path)
    responses = iter([_NAIVE_DIFF, _CLAMPED_DIFF])
    calls = []

    def fake_complete(system, messages):
        calls.append(messages)
        return next(responses)

    result = execute_step_with_verification(
        _step(), _ORIGINAL_CONTENT, _task_context(), complete_fn=fake_complete, repo_root=tmp_path
    )

    assert result.success is True
    assert result.attempts == 2
    assert result.needs_human_help is False
    assert len(calls) == 2

    second_call_content = calls[1][-1]["content"]
    assert _NAIVE_DIFF.strip() in second_call_content
    assert "test_subtract_never_negative" in second_call_content or "assert" in second_call_content
    assert "Test failure output" in second_call_content


def test_execute_step_returns_needs_human_help_after_exhausting_retries(tmp_path):
    _write_repo(tmp_path)

    def fake_complete(system, messages):
        return _NAIVE_DIFF

    result = execute_step_with_verification(
        _step(),
        _ORIGINAL_CONTENT,
        _task_context(),
        complete_fn=fake_complete,
        repo_root=tmp_path,
        max_retries=3,
    )

    assert result.success is False
    assert result.needs_human_help is True
    assert result.attempts == 3
    assert isinstance(result.diff, Diff)
    assert isinstance(result.test_result, TestResult)
    assert result.test_result.passed is False


def test_execute_step_default_max_retries_is_three(tmp_path):
    _write_repo(tmp_path)
    call_count = {"n": 0}

    def fake_complete(system, messages):
        call_count["n"] += 1
        return _NAIVE_DIFF

    result = execute_step_with_verification(
        _step(), _ORIGINAL_CONTENT, _task_context(), complete_fn=fake_complete, repo_root=tmp_path
    )

    assert result.needs_human_help is True
    assert call_count["n"] == 3


def test_execute_step_blocks_denied_file_without_calling_llm_or_tests(tmp_path):
    _write_repo(tmp_path)
    config = SolvixConfig(deny_paths=("calc.py",))
    calls = []

    def fake_complete(system, messages):
        calls.append(messages)
        return _CLAMPED_DIFF

    result = execute_step_with_verification(
        _step(),
        _ORIGINAL_CONTENT,
        _task_context(),
        complete_fn=fake_complete,
        repo_root=tmp_path,
        config=config,
    )

    assert result.success is False
    assert result.blocked is True
    assert result.needs_human_help is True
    assert result.diff is None
    assert result.test_result is None
    assert result.attempts == 0
    assert calls == []  # propose_diff (and its LLM call) must never run


def test_execute_step_not_blocked_when_file_does_not_match_deny_pattern(tmp_path):
    _write_repo(tmp_path)
    config = SolvixConfig(deny_paths=("secrets/**",))

    def fake_complete(system, messages):
        return _CLAMPED_DIFF

    result = execute_step_with_verification(
        _step(),
        _ORIGINAL_CONTENT,
        _task_context(),
        complete_fn=fake_complete,
        repo_root=tmp_path,
        config=config,
    )

    assert result.blocked is False
    assert result.success is True


def test_execute_step_uses_config_max_retries_over_hardcoded_default(tmp_path):
    _write_repo(tmp_path)
    config = SolvixConfig(max_retries=1)
    call_count = {"n": 0}

    def fake_complete(system, messages):
        call_count["n"] += 1
        return _NAIVE_DIFF

    result = execute_step_with_verification(
        _step(), _ORIGINAL_CONTENT, _task_context(), complete_fn=fake_complete, repo_root=tmp_path,
        config=config,
    )

    assert result.needs_human_help is True
    assert result.attempts == 1
    assert call_count["n"] == 1


def test_check_dangerous_ops_detects_each_builtin_pattern_in_diff():
    dangerous_diffs = [
        "+    os.system('git push --force origin main')\n",
        "+    os.system('git push -f origin main')\n",
        "+    os.system('git reset --hard HEAD~1')\n",
        "+    os.system('git branch -D feature/old')\n",
        "+    os.system('git push origin --delete feature/old')\n",
        "+    cursor.execute('DROP TABLE users')\n",
        "+    cursor.execute('DROP DATABASE prod')\n",
        "+    cursor.execute('TRUNCATE orders')\n",
    ]

    for diff_text in dangerous_diffs:
        result = check_dangerous_ops(diff_text, commands=[])
        assert isinstance(result, DangerousOpsCheck)
        assert result.requires_confirmation is True, f"expected a match for: {diff_text!r}"
        assert result.reasons


def test_check_dangerous_ops_detects_pattern_in_command_not_just_diff():
    result = check_dangerous_ops("+    return a - b\n", commands=["git push --force origin main"])

    assert result.requires_confirmation is True
    assert any("git push --force" in reason for reason in result.reasons)


def test_check_dangerous_ops_config_added_pattern_is_additive_not_replacing():
    config = SolvixConfig(dangerous_ops=(r"kubectl\s+delete\s+namespace",))

    custom_match = check_dangerous_ops("+    run('kubectl delete namespace prod')\n", commands=[], config=config)
    assert custom_match.requires_confirmation is True

    builtin_still_active = check_dangerous_ops(
        "+    os.system('git reset --hard HEAD')\n", commands=[], config=config
    )
    assert builtin_still_active.requires_confirmation is True


def test_check_dangerous_ops_does_not_flag_a_normal_safe_diff():
    result = check_dangerous_ops(_CLAMPED_DIFF, commands=["pytest -q"])

    assert result.requires_confirmation is False
    assert result.reasons == ()


def test_execute_step_reports_needs_human_help_when_diff_never_applies(tmp_path):
    _write_repo(tmp_path)

    def fake_complete(system, messages):
        # Not a unified diff at all, so propose_diff's own internal
        # correction retry also fails, and it raises DiffGenerationError.
        return "I cannot produce a diff for this."

    result = execute_step_with_verification(
        _step(), _ORIGINAL_CONTENT, _task_context(), complete_fn=fake_complete, repo_root=tmp_path
    )

    assert result.success is False
    assert result.needs_human_help is True
    assert result.diff is None
    assert result.test_result is None
    assert result.requires_confirmation is False
    assert result.failure_reason is not None
    assert "diff generation failed after exhausting its own retries" in result.failure_reason


def test_execute_step_requires_confirmation_and_skips_tests_for_dangerous_diff(tmp_path):
    _write_repo(tmp_path)
    dangerous_diff = (
        "--- a/calc.py\n"
        "+++ b/calc.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def subtract(a, b):\n"
        "-    return a - b\n"
        "+    os.system('git push --force origin main')\n"
        "+    return a - b\n"
    )

    def fake_complete(system, messages):
        return dangerous_diff

    import execution.orchestrator as orchestrator_module

    real_run_tests = orchestrator_module.run_tests_on_diff
    calls = []

    def spying_run_tests(*args, **kwargs):
        calls.append((args, kwargs))
        return real_run_tests(*args, **kwargs)

    orchestrator_module.run_tests_on_diff = spying_run_tests
    try:
        result = execute_step_with_verification(
            _step(), _ORIGINAL_CONTENT, _task_context(), complete_fn=fake_complete, repo_root=tmp_path
        )
    finally:
        orchestrator_module.run_tests_on_diff = real_run_tests

    assert result.success is False
    assert result.requires_confirmation is True
    # needs_human_help stays scoped to its C4 meaning (genuine failure after
    # exhausting retries) -- a dangerous-ops match means the agent worked
    # correctly and is being cautious, not that it's stuck. Conflating the
    # two would make it impossible for a future CLI/dashboard to tell
    # "broken" apart from "fine, just needs a yes/no".
    assert result.needs_human_help is False
    assert result.test_result is None
    assert any("dangerous-ops" in reason for reason in result.confirmation_reasons)
    assert calls == []  # run_tests_on_diff must never run for a flagged diff


def test_execute_step_uses_config_test_command_over_hardcoded_default(tmp_path):
    _write_repo(tmp_path)
    config = SolvixConfig(test_command="pytest -q -k never_negative")
    seen_commands = []

    import execution.orchestrator as orchestrator_module

    real_run_tests = orchestrator_module.run_tests_on_diff

    def spying_run_tests(repo_root, diff, test_command):
        seen_commands.append(test_command)
        return real_run_tests(repo_root, diff, test_command=test_command)

    orchestrator_module.run_tests_on_diff = spying_run_tests
    try:
        execute_step_with_verification(
            _step(),
            _ORIGINAL_CONTENT,
            _task_context(),
            complete_fn=lambda system, messages: _CLAMPED_DIFF,
            repo_root=tmp_path,
            config=config,
        )
    finally:
        orchestrator_module.run_tests_on_diff = real_run_tests

    assert seen_commands == ["pytest -q -k never_negative"]
