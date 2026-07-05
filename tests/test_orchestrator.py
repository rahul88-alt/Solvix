from config import SolvixConfig
from context.assembler import FileScore, RetrievalResult
from execution.orchestrator import (
    DangerousOpsCheck,
    StepResult,
    check_dangerous_ops,
    check_test_coverage_sanity,
    detect_assertion_gaming,
    execute_step_with_verification,
)
from execution.test_runner import TestResult
from reasoning.editor import Diff, DiffGenerationError
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


def test_check_dangerous_ops_does_not_flag_truncate_as_a_plain_identifier():
    # SLX-E4: a diff adding a function/variable literally named `truncate`
    # (the exact sample_repo/utils/strings.py shape that caused real
    # false-positive confirmation prompts) must not trip the SQL pattern.
    diff_text = (
        "--- a/utils/strings.py\n"
        "+++ b/utils/strings.py\n"
        "@@ -10,0 +11,4 @@\n"
        "+def truncate(text, length=100):\n"
        "+    if len(text) <= length:\n"
        "+        return text\n"
        "+    return text[:length] + '...'\n"
    )

    result = check_dangerous_ops(diff_text, commands=[])

    assert result.requires_confirmation is False
    assert result.reasons == ()


def test_check_dangerous_ops_does_not_flag_truncate_string_identifier():
    diff_text = (
        "--- a/utils/strings.py\n"
        "+++ b/utils/strings.py\n"
        "@@ -1,0 +2,2 @@\n"
        "+def truncate_string(value, max_len):\n"
        "+    return truncate(value, max_len)\n"
    )

    result = check_dangerous_ops(diff_text, commands=[])

    assert result.requires_confirmation is False
    assert result.reasons == ()


def test_check_dangerous_ops_does_not_flag_truncate_in_comment_or_docstring():
    diff_text = (
        "--- a/utils/strings.py\n"
        "+++ b/utils/strings.py\n"
        "@@ -1,0 +2,2 @@\n"
        '+    """Truncate a string to fit within max_length, appending an ellipsis."""\n'
        "+    # truncate the text if it exceeds the max length\n"
    )

    result = check_dangerous_ops(diff_text, commands=[])

    assert result.requires_confirmation is False
    assert result.reasons == ()


def test_check_dangerous_ops_still_flags_real_truncate_table_statement():
    diff_text = (
        "--- a/migrations/0001_reset.py\n"
        "+++ b/migrations/0001_reset.py\n"
        "@@ -1,0 +2,1 @@\n"
        "+    cursor.execute('TRUNCATE TABLE orders;')\n"
    )

    result = check_dangerous_ops(diff_text, commands=[])

    assert result.requires_confirmation is True
    assert result.reasons


def test_check_dangerous_ops_still_flags_real_drop_table_and_drop_database():
    for diff_text in (
        "+    cursor.execute('DROP TABLE orders')\n",
        "+    cursor.execute('DROP DATABASE prod')\n",
    ):
        result = check_dangerous_ops(diff_text, commands=[])
        assert result.requires_confirmation is True, f"expected a match for: {diff_text!r}"
        assert result.reasons


def test_check_dangerous_ops_still_flags_drop_with_if_exists_clause():
    # The `IF [NOT] EXISTS` clause is a common defensive migration idiom and
    # must not let a genuinely destructive DROP/TRUNCATE slip past undetected.
    for diff_text in (
        "+    cursor.execute('DROP TABLE IF EXISTS orders')\n",
        "+    cursor.execute('DROP TABLE IF NOT EXISTS orders')\n",
        "+    cursor.execute('DROP DATABASE IF EXISTS prod')\n",
        "+    cursor.execute('TRUNCATE TABLE IF EXISTS orders')\n",
    ):
        result = check_dangerous_ops(diff_text, commands=[])
        assert result.requires_confirmation is True, f"expected a match for: {diff_text!r}"
        assert result.reasons


def test_execute_step_retries_diff_generation_error_instead_of_terminating_immediately(tmp_path):
    _write_repo(tmp_path)
    call_count = {"n": 0}

    def fake_complete(system, messages):
        call_count["n"] += 1
        # Not a unified diff at all, so propose_diff's own internal
        # correction retry also fails, and it raises DiffGenerationError on
        # every outer attempt -- SLX-C6: this must consume the outer
        # per-step retry budget (default 3 attempts), not terminate on the
        # very first attempt. Each outer attempt itself costs 2 calls here
        # (propose_diff's initial call plus its own internal correction
        # retry), so 3 outer attempts means 6 total calls.
        return "I cannot produce a diff for this."

    result = execute_step_with_verification(
        _step(), _ORIGINAL_CONTENT, _task_context(), complete_fn=fake_complete, repo_root=tmp_path
    )

    assert result.success is False
    assert result.needs_human_help is True
    assert result.diff is None
    assert result.test_result is None
    assert result.requires_confirmation is False
    assert result.attempts == 3  # exhausted the full outer per-step budget, not just 1
    assert call_count["n"] == 6
    assert result.failure_reason is not None
    assert "exhausting the outer per-step retry budget" in result.failure_reason
    assert len(result.attempt_failures) == 3


def test_execute_step_recovers_from_diff_generation_error_on_a_later_attempt(tmp_path):
    _write_repo(tmp_path)
    # Attempt 1: both propose_diff's initial call and its own internal
    # correction retry return unparseable text, so propose_diff itself
    # raises DiffGenerationError up to the outer loop. Attempt 2: the
    # first call already returns a good diff, so propose_diff never even
    # needs its own internal retry.
    responses = iter(["not a diff at all", "still not a diff", _CLAMPED_DIFF])
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
    assert result.failure_reason is None
    assert len(calls) == 3
    # the second outer attempt's prompt must carry the first attempt's
    # parse failure forward, the same way a test failure would be relayed.
    third_call_content = calls[2][-1]["content"]
    assert "could not be parsed" in third_call_content or "did not apply cleanly" in third_call_content


def test_execute_step_returns_clean_result_when_diff_apply_crashes_during_verification(tmp_path):
    """SLX-C9: execution.test_runner.apply_diff now converts a `patch`
    failure during the real (non-dry-run) apply -- e.g. the exact
    count_vowels smoke-test failure, a diff for utils/strings.py whose
    parent directory doesn't exist and whose is_new_file was False -- into
    DiffGenerationError instead of letting a raw subprocess.
    CalledProcessError escape (see tests/test_runner.py for that
    conversion, exercised directly against the real `patch` subprocess).
    This confirms execute_step_with_verification's outer loop absorbs a
    DiffGenerationError raised from run_tests_on_diff via the exact same
    plumbing a DiffGenerationError from propose_diff itself already uses,
    ending in a clean needs_human_help StepResult rather than a raised
    exception reaching the caller.
    """
    _write_repo(tmp_path)

    def fake_complete(system, messages):
        return _CLAMPED_DIFF

    import execution.orchestrator as orchestrator_module

    real_run_tests = orchestrator_module.run_tests_on_diff

    def always_crashes(repo_root, diff, test_command):
        raise DiffGenerationError(
            f"diff failed to apply to {diff.target_file}: "
            "Command ['patch', ...] returned non-zero exit status 2."
        )

    orchestrator_module.run_tests_on_diff = always_crashes
    try:
        result = execute_step_with_verification(
            _step(), _ORIGINAL_CONTENT, _task_context(), complete_fn=fake_complete, repo_root=tmp_path
        )
    finally:
        orchestrator_module.run_tests_on_diff = real_run_tests

    assert result.success is False
    assert result.needs_human_help is True
    assert result.test_result is None
    assert result.failure_reason is not None
    assert "exhausting the outer per-step retry budget" in result.failure_reason
    assert len(result.attempt_failures) == 3


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


def test_execute_step_emits_progress_events_in_order_for_clean_pass(tmp_path):
    _write_repo(tmp_path)
    events = []

    result = execute_step_with_verification(
        _step(),
        _ORIGINAL_CONTENT,
        _task_context(),
        complete_fn=lambda system, messages: _CLAMPED_DIFF,
        repo_root=tmp_path,
        step_index=1,
        on_progress=events.append,
    )

    assert result.success is True
    joined = "\n".join(events)
    # Real phase markers, in the order they actually happen: propose ->
    # lint -> run tests -> pass. Not asserting exact wording elsewhere.
    assert events[0].startswith("Step 1: proposing changes to calc.py (attempt 1/")
    assert "Step 1: lint clean (attempt 1)" in events
    assert "Step 1: running tests (attempt 1)..." in events
    assert "Step 1: tests passed (attempt 1)" in events
    assert joined.index("proposing changes") < joined.index("running tests")
    assert joined.index("running tests") < joined.index("tests passed")


def test_execute_step_emits_progress_events_in_order_across_a_retry(tmp_path):
    _write_repo(tmp_path)
    responses = iter([_NAIVE_DIFF, _CLAMPED_DIFF])
    events = []

    result = execute_step_with_verification(
        _step(),
        _ORIGINAL_CONTENT,
        _task_context(),
        complete_fn=lambda system, messages: next(responses),
        repo_root=tmp_path,
        step_index=2,
        on_progress=events.append,
    )

    assert result.success is True
    assert result.attempts == 2
    assert events[0].startswith("Step 2: proposing changes to calc.py (attempt 1/")
    assert "Step 2: tests failed (attempt 1)" in events
    attempt_2_propose = next(i for i, e in enumerate(events) if "attempt 2/" in e)
    attempt_1_fail = events.index("Step 2: tests failed (attempt 1)")
    assert attempt_1_fail < attempt_2_propose
    assert "Step 2: tests passed (attempt 2)" in events
    assert events.index("Step 2: tests passed (attempt 2)") > attempt_2_propose


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


# --- SLX-C5: assertion-gaming detection -------------------------------

_ADD_FAILURE_OUTPUT = (
    "=================== FAILURES ===================\n"
    "____________________ test_add ____________________\n"
    "\n"
    "    def test_add():\n"
    ">       assert add(0.1, 0.2) == 0.3\n"
    "E       assert 0.30000000000000004 == 0.3\n"
    "E        +  where 0.30000000000000004 = add(0.1, 0.2)\n"
    "\n"
    "test_calc.py:5: AssertionError\n"
)

_ADD_GAMED_DIFF = (
    "--- a/test_calc.py\n"
    "+++ b/test_calc.py\n"
    "@@ -1,5 +1,5 @@\n"
    " from calc import add\n"
    " \n"
    " \n"
    " def test_add():\n"
    "-    assert add(0.1, 0.2) == 0.3\n"
    "+    assert add(0.1, 0.2) == 0.30000000000000004\n"
)


def test_detect_assertion_gaming_flags_the_real_slx_d2_scenario():
    message = detect_assertion_gaming(_ADD_FAILURE_OUTPUT, _ADD_GAMED_DIFF)

    assert message is not None
    assert "0.3" in message
    assert "0.30000000000000004" in message


def test_detect_assertion_gaming_not_flagged_when_only_implementation_changes():
    legitimate_fix_diff = (
        "--- a/calc.py\n"
        "+++ b/calc.py\n"
        "@@ -1,2 +1,3 @@\n"
        " def add(a, b):\n"
        "-    return a + b\n"
        "+    from decimal import Decimal\n"
        "+    return float(Decimal(str(a)) + Decimal(str(b)))\n"
    )

    message = detect_assertion_gaming(_ADD_FAILURE_OUTPUT, legitimate_fix_diff)

    assert message is None


def test_detect_assertion_gaming_not_flagged_for_unrelated_test_file_edit():
    unrelated_test_edit_diff = (
        "--- a/test_calc.py\n"
        "+++ b/test_calc.py\n"
        "@@ -1,3 +1,7 @@\n"
        " from calc import add\n"
        " \n"
        "+def test_add_negative_numbers():\n"
        "+    \"\"\"Regression test for negative inputs.\"\"\"\n"
        "+    assert add(-1, -1) == -2\n"
        "+\n"
        " def test_add():\n"
        "     assert add(0.1, 0.2) == 0.3\n"
    )

    message = detect_assertion_gaming(_ADD_FAILURE_OUTPUT, unrelated_test_edit_diff)

    assert message is None


def test_detect_assertion_gaming_not_flagged_when_new_literal_does_not_match_actual():
    # A genuinely different guess at the expected value (not a match to the
    # previous attempt's actual failing output) is not flagged -- see
    # detect_assertion_gaming's docstring / SLX-C5 for why: with the
    # implementation untouched, this diff would just fail again on a
    # deterministic rerun, so there's no "passes by rewriting" case here.
    different_guess_diff = (
        "--- a/test_calc.py\n"
        "+++ b/test_calc.py\n"
        "@@ -3,4 +3,4 @@\n"
        " from calc import add\n"
        " \n"
        " def test_add():\n"
        "-    assert add(0.1, 0.2) == 0.3\n"
        "+    assert add(0.1, 0.2) == 0.5\n"
    )

    message = detect_assertion_gaming(_ADD_FAILURE_OUTPUT, different_guess_diff)

    assert message is None


def test_detect_assertion_gaming_returns_none_when_no_previous_assertion_failure():
    message = detect_assertion_gaming("1 failed - some unrelated error\n", _ADD_GAMED_DIFF)

    assert message is None


def test_execute_step_flags_assertion_gaming_on_a_gaming_retry(tmp_path):
    (tmp_path / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(0.1, 0.2) == 0.3\n"
    )
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b\n")

    first_attempt_diff = (
        "--- a/test_calc.py\n"
        "+++ b/test_calc.py\n"
        "@@ -1,5 +1,5 @@\n"
        " from calc import add\n"
        " \n"
        " \n"
        " def test_add():\n"
        "-    assert add(0.1, 0.2) == 0.3\n"
        "+    assert add(0.1, 0.2) == 0.3  # comment only, still fails\n"
    )
    responses = iter([first_attempt_diff, _ADD_GAMED_DIFF])

    step = PlanStep(file="test_calc.py", description="fix add() to satisfy the test")

    result = execute_step_with_verification(
        step,
        (tmp_path / "test_calc.py").read_text(),
        _task_context(),
        complete_fn=lambda system, messages: next(responses),
        repo_root=tmp_path,
    )

    assert result.success is True
    assert result.assertion_gaming_suspected is True
    assert "0.30000000000000004" in result.assertion_gaming_details


# --- Epic C2: check_test_coverage_sanity ----------------------------------

_IMPL_DIFF_ADDING_DISCOUNT = (
    "--- a/pricing.py\n"
    "+++ b/pricing.py\n"
    "@@ -1,2 +1,5 @@\n"
    " def base_price(items):\n"
    "     return sum(items)\n"
    "+\n"
    "+def apply_discount(price, percent):\n"
    "+    return price * (1 - percent / 100)\n"
)


def test_check_test_coverage_sanity_passes_for_genuine_covering_test():
    test_diff = (
        "--- a/tests/test_pricing.py\n"
        "+++ b/tests/test_pricing.py\n"
        "@@ -1,2 +1,5 @@\n"
        " from pricing import base_price\n"
        "+from pricing import apply_discount\n"
        "+\n"
        "+def test_apply_discount():\n"
        "+    assert apply_discount(100, 10) == 90\n"
    )

    assert check_test_coverage_sanity(_IMPL_DIFF_ADDING_DISCOUNT, test_diff) is None


def test_check_test_coverage_sanity_passes_for_unittest_style_assertion():
    # Confirmed during this story's real-Ollama smoke test: the model wrote
    # a genuinely covering test using unittest.TestCase's self.assertEqual
    # instead of a bare `assert` -- that must not be flagged as lacking a
    # real assertion just because it isn't pytest-style.
    test_diff = (
        "--- a/tests/test_pricing.py\n"
        "+++ b/tests/test_pricing.py\n"
        "@@ -1,2 +1,7 @@\n"
        " from pricing import base_price\n"
        "+import unittest\n"
        "+from pricing import apply_discount\n"
        "+\n"
        "+class TestPricing(unittest.TestCase):\n"
        "+    def test_apply_discount(self):\n"
        "+        self.assertEqual(apply_discount(100, 10), 90)\n"
    )

    assert check_test_coverage_sanity(_IMPL_DIFF_ADDING_DISCOUNT, test_diff) is None


def test_check_test_coverage_sanity_flags_test_that_does_not_reference_changed_symbol():
    test_diff = (
        "--- a/tests/test_pricing.py\n"
        "+++ b/tests/test_pricing.py\n"
        "@@ -1,2 +1,5 @@\n"
        " from pricing import base_price\n"
        "+\n"
        "+def test_base_price_unrelated():\n"
        "+    assert base_price([1, 2]) == 3\n"
    )

    message = check_test_coverage_sanity(_IMPL_DIFF_ADDING_DISCOUNT, test_diff)

    assert message is not None
    assert "apply_discount" in message


def test_check_test_coverage_sanity_flags_test_with_no_real_assertion():
    test_diff = (
        "--- a/tests/test_pricing.py\n"
        "+++ b/tests/test_pricing.py\n"
        "@@ -1,2 +1,5 @@\n"
        " from pricing import apply_discount\n"
        "+\n"
        "+def test_apply_discount():\n"
        "+    apply_discount(100, 10)\n"
        "+    assert True\n"
    )

    message = check_test_coverage_sanity(_IMPL_DIFF_ADDING_DISCOUNT, test_diff)

    assert message is not None
    assert "real assertion" in message


def test_check_test_coverage_sanity_flags_bare_pass_test_body():
    test_diff = (
        "--- a/tests/test_pricing.py\n"
        "+++ b/tests/test_pricing.py\n"
        "@@ -1,2 +1,4 @@\n"
        " from pricing import apply_discount\n"
        "+\n"
        "+def test_apply_discount():\n"
        "+    pass\n"
    )

    message = check_test_coverage_sanity(_IMPL_DIFF_ADDING_DISCOUNT, test_diff)

    assert message is not None
    assert "apply_discount" in message
    assert "real assertion" in message


def test_check_test_coverage_sanity_skips_symbol_check_when_impl_diff_has_no_def_or_class():
    # A diff that only edits an existing function's body (no `def`/`class`
    # line among its added lines) gives no symbol name to check against --
    # the assertion check should still run on its own.
    impl_diff = (
        "--- a/pricing.py\n"
        "+++ b/pricing.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def base_price(items):\n"
        "-    return sum(items)\n"
        "+    return sum(items) if items else 0\n"
    )
    test_diff = (
        "--- a/tests/test_pricing.py\n"
        "+++ b/tests/test_pricing.py\n"
        "@@ -1,2 +1,4 @@\n"
        " from pricing import base_price\n"
        "+def test_base_price_empty():\n"
        "+    assert base_price([]) == 0\n"
    )

    assert check_test_coverage_sanity(impl_diff, test_diff) is None
