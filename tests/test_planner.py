import pytest

from config import SolvixConfig
from context.assembler import FileScore, RetrievalResult
from reasoning.planner import PlanGenerationError, PlanStep, generate_plan
from reasoning.task_input import TaskContext


def _task_context(files=("calculator.py",), related=()):
    retrieval = RetrievalResult(
        files=[FileScore(file_path=f, score=1.0, reasons=("test",)) for f in files],
        related_files=[FileScore(file_path=f, score=0.1, reasons=("one_hop_import",)) for f in related],
    )
    return TaskContext(task="fix the subtract function", retrieval=retrieval)


def test_generate_plan_parses_clean_json():
    canned = '[{"file": "calculator.py", "description": "fix subtract to handle negatives"}]'
    plan = generate_plan(_task_context(), complete_fn=lambda system, messages: canned)

    assert plan.steps == [
        PlanStep(file="calculator.py", description="fix subtract to handle negatives")
    ]
    assert plan.requires_approval is False
    assert plan.approval_reasons == ()


def test_generate_plan_strips_surrounding_prose_and_fences():
    canned = (
        "Here is the plan:\n```json\n"
        '[{"file": "calculator.py", "description": "fix subtract"}]\n'
        "```\nLet me know if you have questions."
    )
    plan = generate_plan(_task_context(), complete_fn=lambda system, messages: canned)

    assert len(plan.steps) == 1
    assert plan.steps[0].file == "calculator.py"


def test_generate_plan_raises_clear_error_on_malformed_json():
    canned = "Sure! I'll fix calculator.py by updating subtract."  # no JSON at all

    with pytest.raises(PlanGenerationError, match="did not contain a JSON array"):
        generate_plan(_task_context(), complete_fn=lambda system, messages: canned)


def test_generate_plan_raises_clear_error_on_invalid_json_syntax():
    canned = "[{\"file\": \"calculator.py\", \"description\": }]"  # syntax error

    with pytest.raises(PlanGenerationError, match="not valid JSON"):
        generate_plan(_task_context(), complete_fn=lambda system, messages: canned)


def test_generate_plan_raises_clear_error_on_missing_fields():
    canned = '[{"file": "calculator.py"}]'  # missing "description"

    with pytest.raises(PlanGenerationError, match="missing required"):
        generate_plan(_task_context(), complete_fn=lambda system, messages: canned)


def test_generate_plan_raises_clear_error_on_empty_plan():
    canned = "[]"

    with pytest.raises(PlanGenerationError, match="empty plan"):
        generate_plan(_task_context(), complete_fn=lambda system, messages: canned)


def test_plan_requires_approval_when_touching_more_than_max_files():
    canned = (
        '[{"file": "a.py", "description": "x"}, '
        '{"file": "b.py", "description": "x"}, '
        '{"file": "c.py", "description": "x"}, '
        '{"file": "d.py", "description": "x"}]'
    )
    plan = generate_plan(
        _task_context(), complete_fn=lambda system, messages: canned, max_files=3
    )

    assert plan.requires_approval is True
    assert any("touches 4 files" in r for r in plan.approval_reasons)


def test_plan_requires_approval_when_touching_sensitive_path():
    canned = '[{"file": "auth/login.py", "description": "add rate limiting"}]'
    plan = generate_plan(_task_context(), complete_fn=lambda system, messages: canned)

    assert plan.requires_approval is True
    assert any("sensitive path" in r for r in plan.approval_reasons)
    assert any("auth/login.py" in r for r in plan.approval_reasons)


def test_generate_plan_retries_once_with_correction_prompt_and_recovers():
    responses = iter(
        [
            "Sure, here you go: file=calculator.py, fix subtract",  # malformed
            '[{"file": "calculator.py", "description": "fix subtract"}]',  # corrected
        ]
    )
    calls = []

    def fake_complete(system, messages):
        calls.append(messages)
        return next(responses)

    plan = generate_plan(_task_context(), complete_fn=fake_complete)

    assert len(calls) == 2
    # the retry call includes the correction prompt referencing the parse error
    assert "corrected JSON array" in calls[1][-1]["content"]
    assert plan.steps == [PlanStep(file="calculator.py", description="fix subtract")]


def test_generate_plan_raises_after_retry_still_malformed():
    with pytest.raises(PlanGenerationError):
        generate_plan(
            _task_context(),
            complete_fn=lambda system, messages: "still not json, sorry",
        )


def test_plan_does_not_require_approval_for_small_non_sensitive_plan():
    canned = '[{"file": "calculator.py", "description": "fix subtract"}]'
    plan = generate_plan(_task_context(), complete_fn=lambda system, messages: canned)

    assert plan.requires_approval is False
    assert plan.approval_reasons == ()


def test_plan_uses_config_sensitive_paths_over_hardcoded_default():
    canned = '[{"file": "reports/export.py", "description": "add CSV export"}]'
    config = SolvixConfig(sensitive_paths=("reports/",))

    plan = generate_plan(
        _task_context(), complete_fn=lambda system, messages: canned, config=config
    )

    assert plan.requires_approval is True
    assert any("reports/export.py" in r for r in plan.approval_reasons)


def test_plan_config_sensitive_paths_are_additive_to_hardcoded_default():
    # config supplies a sensitive path that doesn't include auth/, but the
    # built-in auth/ protection must still apply -- config can only add
    # protection, never silently remove it (see SolvixConfig.__post_init__).
    canned = '[{"file": "auth/login.py", "description": "add rate limiting"}]'
    config = SolvixConfig(sensitive_paths=("reports/",))

    plan = generate_plan(
        _task_context(), complete_fn=lambda system, messages: canned, config=config
    )

    assert plan.requires_approval is True
    assert any("auth/login.py" in r for r in plan.approval_reasons)
