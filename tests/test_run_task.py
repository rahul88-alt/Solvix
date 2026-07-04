import difflib
import tempfile
import types
from pathlib import Path

import pytest

from config import SolvixConfig
from context.assembler import FileScore, RetrievalResult
from execution import orchestrator
from execution.orchestrator import StepResult, TaskResult, run_task
from reasoning.planner import Plan, PlanStep
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


def _plan():
    return Plan(
        steps=[PlanStep(file="calc.py", description="clamp subtract result at zero")],
        requires_approval=False,
        approval_reasons=(),
    )


def test_run_task_completes_normal_1_step_plan_within_budget(tmp_path):
    _write_repo(tmp_path)
    calls = []

    def fake_complete(system, messages):
        calls.append(messages)
        return _CLAMPED_DIFF

    config = SolvixConfig(max_retries=3, max_task_attempts=10)

    result = run_task(
        _plan(), _task_context(), complete_fn=fake_complete, repo_root=tmp_path, config=config
    )

    assert isinstance(result, TaskResult)
    assert result.success is True
    assert result.needs_human_help is False
    assert result.total_attempts == 1
    assert len(result.step_results) == 1
    assert result.step_results[0].success is True
    assert result.culprit_step is None
    assert result.reason is None
    assert len(calls) == 1


def test_run_task_needs_human_help_when_one_step_exhausts_many_retries(tmp_path):
    _write_repo(tmp_path)

    def fake_complete(system, messages):
        return _NAIVE_DIFF

    # per-step cap (5) and task cap (5) are equal here so the single step's
    # own exhaustion is exactly what trips the task-level accounting.
    config = SolvixConfig(max_retries=5, max_task_attempts=5)

    result = run_task(
        _plan(), _task_context(), complete_fn=fake_complete, repo_root=tmp_path, config=config
    )

    assert result.success is False
    assert result.needs_human_help is True
    assert result.total_attempts == 5
    assert len(result.step_results) == 1
    assert result.step_results[0].needs_human_help is True
    assert result.culprit_step == PlanStep(file="calc.py", description="clamp subtract result at zero")
    assert "task-level retry cap" in result.reason


def test_run_task_reports_diff_generation_failure_distinctly(tmp_path):
    _write_repo(tmp_path)
    call_count = {"n": 0}

    def fake_complete(system, messages):
        call_count["n"] += 1
        return "not a diff at all"

    config = SolvixConfig(max_retries=3, max_task_attempts=10)

    result = run_task(
        _plan(), _task_context(), complete_fn=fake_complete, repo_root=tmp_path, config=config
    )

    assert result.success is False
    assert result.needs_human_help is True
    # SLX-C6: DiffGenerationError consumes the outer per-step retry budget
    # rather than terminating on the first occurrence, so all 3 configured
    # attempts should have actually run before this bubbles up. Each outer
    # attempt costs 2 complete_fn calls (propose_diff's own internal
    # correction retry also fails on unparseable text), so 3 attempts is 6
    # total calls.
    assert call_count["n"] == 6
    assert result.total_attempts == 3
    assert "exhausting the outer per-step retry budget" in result.reason
    assert result.step_results[0].failure_reason is not None


def test_run_task_low_per_step_cap_high_task_cap_reports_step_exhaustion_not_task_cap(tmp_path):
    _write_repo(tmp_path)
    call_count = {"n": 0}

    def fake_complete(system, messages):
        call_count["n"] += 1
        return _NAIVE_DIFF

    config = SolvixConfig(max_retries=1, max_task_attempts=100)

    result = run_task(
        _plan(), _task_context(), complete_fn=fake_complete, repo_root=tmp_path, config=config
    )

    assert result.success is False
    assert result.needs_human_help is True
    assert result.total_attempts == 1
    assert call_count["n"] == 1
    assert "task-level cap" in result.reason
    assert "not reached" in result.reason
    assert "exhausted its own per-step retry budget" in result.reason


def test_run_task_high_per_step_cap_low_task_cap_cuts_step_short_mid_loop(tmp_path):
    _write_repo(tmp_path)
    call_count = {"n": 0}

    def fake_complete(system, messages):
        call_count["n"] += 1
        return _NAIVE_DIFF

    # per-step cap is generous (20) but the task-level cap (3) is what
    # actually governs here: run_task must clamp the step's own retry loop
    # to the remaining task budget so it genuinely stops after 3 attempts
    # (3 real LLM calls) instead of burning through all 20 before run_task
    # gets a chance to notice the cap was exceeded.
    config = SolvixConfig(max_retries=20, max_task_attempts=3)

    result = run_task(
        _plan(), _task_context(), complete_fn=fake_complete, repo_root=tmp_path, config=config
    )

    assert result.success is False
    assert result.needs_human_help is True
    assert result.total_attempts == 3
    assert call_count["n"] == 3  # cut short, not all 20 of the step's own allowance
    assert result.step_results[0].attempts == 3
    assert "cut step" in result.reason
    assert "its own per-step cap is 20" in result.reason


def test_run_task_stops_before_starting_a_step_once_cap_already_reached(tmp_path):
    _write_repo(tmp_path)
    (tmp_path / "other.py").write_text("def noop():\n    pass\n")

    step_one = PlanStep(file="calc.py", description="clamp subtract result at zero")
    step_two = PlanStep(file="other.py", description="should never run")
    plan = Plan(steps=[step_one, step_two], requires_approval=False, approval_reasons=())

    calls = []

    def fake_complete(system, messages):
        calls.append(messages)
        return _NAIVE_DIFF

    config = SolvixConfig(max_retries=2, max_task_attempts=2)

    result = run_task(
        plan, _task_context(), complete_fn=fake_complete, repo_root=tmp_path, config=config
    )

    assert result.success is False
    assert result.needs_human_help is True
    assert len(result.step_results) == 1  # step_two never ran
    assert result.total_attempts == 2
    assert result.culprit_step == step_one
    assert len(calls) == 2  # only step_one's attempts, none for step_two


# --- Epic C2: paired implementation + test step ---------------------------

_IMPL_DIFF_CLAMPING_SUBTRACT = _CLAMPED_DIFF

_ORIGINAL_TEST_CALC_CONTENT = (
    "from calc import subtract\n\n\n"
    "def test_subtract_never_negative():\n"
    "    assert subtract(2, 5) == 0\n"
)


def _make_diff(original: str, updated: str, path: str) -> str:
    return "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )


_GENUINE_TEST_DIFF = _make_diff(
    _ORIGINAL_TEST_CALC_CONTENT,
    _ORIGINAL_TEST_CALC_CONTENT
    + "\n\ndef test_subtract_clamps_to_zero():\n    assert subtract(2, 5) == 0\n",
    "test_calc.py",
)

_TRIVIAL_TEST_DIFF = _make_diff(
    _ORIGINAL_TEST_CALC_CONTENT,
    _ORIGINAL_TEST_CALC_CONTENT + "\n\ndef test_subtract_placeholder():\n    assert True\n",
    "test_calc.py",
)


def _two_step_plan():
    return Plan(
        steps=[
            PlanStep(file="calc.py", description="clamp subtract result at zero"),
            PlanStep(file="test_calc.py", description="add a test for the clamp"),
        ],
        requires_approval=False,
        approval_reasons=(),
    )


def test_run_task_does_not_flag_a_genuinely_covering_paired_test(tmp_path):
    _write_repo(tmp_path)
    responses = iter([_IMPL_DIFF_CLAMPING_SUBTRACT, _GENUINE_TEST_DIFF])

    config = SolvixConfig(max_retries=3, max_task_attempts=10)

    result = run_task(
        _two_step_plan(),
        _task_context(),
        complete_fn=lambda system, messages: next(responses),
        repo_root=tmp_path,
        config=config,
    )

    assert result.success is True
    assert len(result.step_results) == 2
    assert result.step_results[1].weak_test_coverage_suspected is False


def test_run_task_flags_a_paired_test_with_no_real_assertion(tmp_path):
    _write_repo(tmp_path)
    responses = iter([_IMPL_DIFF_CLAMPING_SUBTRACT, _TRIVIAL_TEST_DIFF])

    config = SolvixConfig(max_retries=3, max_task_attempts=10)

    result = run_task(
        _two_step_plan(),
        _task_context(),
        complete_fn=lambda system, messages: next(responses),
        repo_root=tmp_path,
        config=config,
    )

    assert result.success is True
    assert len(result.step_results) == 2
    assert result.step_results[1].weak_test_coverage_suspected is True
    assert "real assertion" in result.step_results[1].weak_test_coverage_details


# --- Epic C2: cross-step diff accumulation + symbol-reference flagging ----
#
# calc.py starts with only add(); the implementation step introduces a
# brand-new is_positive() function. Verifying the test step's diff imports
# and calls is_positive() successfully in the sandbox is only possible
# because run_task now carries the implementation step's diff forward into
# the working copy the test step is verified against -- before that fix,
# this scenario would fail with an ImportError regardless of how good the
# test was, since the sandboxed test step ran only against the pristine,
# pre-implementation calc.py (see this story's design discussion).

_ORIGINAL_CALC_WITH_ADD = "def add(a, b):\n    return a + b\n"

_IMPL_DIFF_ADDING_IS_POSITIVE = _make_diff(
    _ORIGINAL_CALC_WITH_ADD,
    _ORIGINAL_CALC_WITH_ADD + "\n\ndef is_positive(n):\n    return n > 0\n",
    "calc.py",
)

_ORIGINAL_TEST_CALC_WITH_ADD = "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"

_TEST_DIFF_REFERENCING_NEW_FUNCTION = _make_diff(
    _ORIGINAL_TEST_CALC_WITH_ADD,
    "from calc import add, is_positive\n\n\n"
    "def test_add():\n    assert add(2, 3) == 5\n\n\n"
    "def test_is_positive():\n    assert is_positive(5) is True\n",
    "test_calc.py",
)

_TEST_DIFF_UNRELATED_TO_NEW_FUNCTION = _make_diff(
    _ORIGINAL_TEST_CALC_WITH_ADD,
    _ORIGINAL_TEST_CALC_WITH_ADD + "\n\ndef test_add_again():\n    assert add(1, 1) == 2\n",
    "test_calc.py",
)


def _write_add_repo(tmp_path):
    (tmp_path / "calc.py").write_text(_ORIGINAL_CALC_WITH_ADD)
    (tmp_path / "test_calc.py").write_text(_ORIGINAL_TEST_CALC_WITH_ADD)
    return tmp_path


def _add_task_context():
    retrieval = RetrievalResult(
        files=[FileScore(file_path="calc.py", score=1.0, reasons=("test",))],
        related_files=[],
    )
    return TaskContext(task="add an is_positive helper to calc.py", retrieval=retrieval)


def _is_positive_plan():
    return Plan(
        steps=[
            PlanStep(file="calc.py", description="add an is_positive(n) helper function"),
            PlanStep(file="test_calc.py", description="add a test for is_positive"),
        ],
        requires_approval=False,
        approval_reasons=(),
    )


def test_run_task_accumulates_diffs_so_paired_test_can_import_new_function(tmp_path):
    _write_add_repo(tmp_path)
    responses = iter([_IMPL_DIFF_ADDING_IS_POSITIVE, _TEST_DIFF_REFERENCING_NEW_FUNCTION])

    config = SolvixConfig(max_retries=3, max_task_attempts=10)

    result = run_task(
        _is_positive_plan(),
        _add_task_context(),
        complete_fn=lambda system, messages: next(responses),
        repo_root=tmp_path,
        config=config,
    )

    assert result.success is True
    assert result.step_results[1].test_result.passed is True
    assert result.step_results[1].weak_test_coverage_suspected is False


def test_run_task_flags_a_paired_test_that_does_not_reference_the_new_function(tmp_path):
    _write_add_repo(tmp_path)
    responses = iter([_IMPL_DIFF_ADDING_IS_POSITIVE, _TEST_DIFF_UNRELATED_TO_NEW_FUNCTION])

    config = SolvixConfig(max_retries=3, max_task_attempts=10)

    result = run_task(
        _is_positive_plan(),
        _add_task_context(),
        complete_fn=lambda system, messages: next(responses),
        repo_root=tmp_path,
        config=config,
    )

    assert result.success is True
    assert result.step_results[1].weak_test_coverage_suspected is True
    assert "is_positive" in result.step_results[1].weak_test_coverage_details


# --- Epic C2: task-scoped working copy is always cleaned up ---------------
#
# run_task's diff-accumulation working copy (a tempfile.TemporaryDirectory)
# must not leak disk space on any exit path -- success, needs_human_help,
# or a genuinely unanticipated exception mid-task -- the same category of
# concern as SLX-E1's orphaned-container cleanup. Verified here by wrapping
# tempfile.TemporaryDirectory to record every directory run_task creates,
# then asserting each one is gone from disk once run_task returns or raises.


class _TrackingTemporaryDirectory(tempfile.TemporaryDirectory):
    created: list[str] = []

    def __enter__(self):
        name = super().__enter__()
        _TrackingTemporaryDirectory.created.append(name)
        return name


@pytest.fixture
def _tracked_scratch_dirs(monkeypatch):
    # Replaces the `tempfile` name inside execution.orchestrator's own
    # module namespace only (not the real tempfile module everyone else,
    # including execution.test_runner's own per-attempt scratch copies,
    # imports) -- so only run_task's own task-scoped working copy is
    # tracked here, not every other TemporaryDirectory created during the
    # same run (each step's sandboxed test run makes its own, separately).
    _TrackingTemporaryDirectory.created = []
    fake_tempfile_module = types.SimpleNamespace(TemporaryDirectory=_TrackingTemporaryDirectory)
    monkeypatch.setattr(orchestrator, "tempfile", fake_tempfile_module)
    return _TrackingTemporaryDirectory.created


def test_run_task_cleans_up_working_copy_on_success(tmp_path, _tracked_scratch_dirs):
    _write_repo(tmp_path)
    config = SolvixConfig(max_retries=3, max_task_attempts=10)

    result = run_task(
        _plan(), _task_context(), complete_fn=lambda s, m: _CLAMPED_DIFF, repo_root=tmp_path, config=config
    )

    assert result.success is True
    assert len(_tracked_scratch_dirs) == 1
    assert not Path(_tracked_scratch_dirs[0]).exists()


def test_run_task_cleans_up_working_copy_on_needs_human_help(tmp_path, _tracked_scratch_dirs):
    _write_repo(tmp_path)
    config = SolvixConfig(max_retries=2, max_task_attempts=2)

    result = run_task(
        _plan(), _task_context(), complete_fn=lambda s, m: _NAIVE_DIFF, repo_root=tmp_path, config=config
    )

    assert result.needs_human_help is True
    assert len(_tracked_scratch_dirs) == 1
    assert not Path(_tracked_scratch_dirs[0]).exists()


def test_run_task_cleans_up_working_copy_on_unanticipated_exception(tmp_path, _tracked_scratch_dirs):
    _write_repo(tmp_path)
    config = SolvixConfig(max_retries=3, max_task_attempts=10)

    def boom(system, messages):
        raise RuntimeError("simulated network failure talking to the LLM backend")

    with pytest.raises(RuntimeError):
        run_task(_plan(), _task_context(), complete_fn=boom, repo_root=tmp_path, config=config)

    assert len(_tracked_scratch_dirs) == 1
    assert not Path(_tracked_scratch_dirs[0]).exists()
