from config import SolvixConfig
from context.assembler import FileScore, RetrievalResult
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

    def fake_complete(system, messages):
        return "not a diff at all"

    config = SolvixConfig(max_retries=3, max_task_attempts=10)

    result = run_task(
        _plan(), _task_context(), complete_fn=fake_complete, repo_root=tmp_path, config=config
    )

    assert result.success is False
    assert result.needs_human_help is True
    assert "diff generation failed after exhausting its own retries" in result.reason
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
