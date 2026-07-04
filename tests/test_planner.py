import pytest

from config import SolvixConfig
from context.assembler import FileScore, RetrievalResult
from reasoning.planner import PlanGenerationError, PlanStep, check_ambiguity, generate_plan
from reasoning.task_input import TaskContext


def _task_context(files=("calculator.py",), related=(), task="fix the subtract function"):
    retrieval = RetrievalResult(
        files=[FileScore(file_path=f, score=1.0, reasons=("test",)) for f in files],
        related_files=[FileScore(file_path=f, score=0.1, reasons=("one_hop_import",)) for f in related],
    )
    return TaskContext(task=task, retrieval=retrieval)


def test_generate_plan_parses_clean_json():
    # "fix the subtract function" (the default task text) is a genuine
    # behavior change, so a paired test step is expected alongside the
    # implementation step -- see the dedicated pairing tests below for
    # focused coverage of that behavior on its own.
    canned = '[{"file": "calculator.py", "description": "fix subtract to handle negatives"}]'
    plan = generate_plan(_task_context(), complete_fn=lambda system, messages: canned)

    assert plan.steps == [
        PlanStep(file="calculator.py", description="fix subtract to handle negatives"),
        PlanStep(
            file="tests/test_calculator.py",
            description=(
                "Add or update a test covering the change in calculator.py: "
                "fix subtract to handle negatives"
            ),
        ),
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

    assert len(plan.steps) == 2
    assert plan.steps[0].file == "calculator.py"
    assert plan.steps[1].file == "tests/test_calculator.py"


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
    # Uses a refactor-flavored task so the Epic C2 test-pairing heuristic
    # doesn't add extra steps here -- this test is isolated to file-count
    # approval logic; see test_generate_plan_pairs_test_step_for_behavior_change
    # for pairing-specific coverage.
    canned = (
        '[{"file": "a.py", "description": "x"}, '
        '{"file": "b.py", "description": "x"}, '
        '{"file": "c.py", "description": "x"}, '
        '{"file": "d.py", "description": "x"}]'
    )
    plan = generate_plan(
        _task_context(task="refactor internal helpers for readability"),
        complete_fn=lambda system, messages: canned,
        max_files=3,
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
    # Refactor-flavored task keeps this isolated to the retry/correction
    # mechanism -- unrelated to Epic C2 test pairing.
    responses = iter(
        [
            "Sure, here you go: file=calculator.py, rename subtract",  # malformed
            '[{"file": "calculator.py", "description": "rename subtract for clarity"}]',  # corrected
        ]
    )
    calls = []

    def fake_complete(system, messages):
        calls.append(messages)
        return next(responses)

    plan = generate_plan(
        _task_context(task="rename subtract for clarity"), complete_fn=fake_complete
    )

    assert len(calls) == 2
    # the retry call includes the correction prompt referencing the parse error
    assert "corrected JSON array" in calls[1][-1]["content"]
    assert plan.steps == [
        PlanStep(file="calculator.py", description="rename subtract for clarity")
    ]


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


# --- Epic C2: paired test steps -------------------------------------------


def test_generate_plan_pairs_test_step_for_behavior_change():
    canned = '[{"file": "app.py", "description": "add input validation to the login form"}]'
    plan = generate_plan(
        _task_context(files=("app.py",), task="add input validation to the login form"),
        complete_fn=lambda system, messages: canned,
    )

    assert len(plan.steps) == 2
    assert plan.steps[0].file == "app.py"
    assert plan.steps[1].file == "tests/test_app.py"
    assert "app.py" in plan.steps[1].description


def test_generate_plan_does_not_pair_test_step_for_refactor_task():
    canned = '[{"file": "calculator.py", "description": "extract a helper function"}]'
    plan = generate_plan(
        _task_context(task="refactor calculator internals for readability"),
        complete_fn=lambda system, messages: canned,
    )

    assert plan.steps == [
        PlanStep(file="calculator.py", description="extract a helper function")
    ]


def test_generate_plan_does_not_duplicate_existing_paired_test_step():
    canned = (
        '[{"file": "calculator.py", "description": "fix subtract"}, '
        '{"file": "tests/test_calculator.py", "description": "add a test for subtract"}]'
    )
    plan = generate_plan(_task_context(), complete_fn=lambda system, messages: canned)

    assert len(plan.steps) == 2
    assert plan.steps[1] == PlanStep(
        file="tests/test_calculator.py", description="add a test for subtract"
    )


def test_generate_plan_does_not_pair_a_step_that_already_targets_a_test_file():
    canned = '[{"file": "tests/test_calculator.py", "description": "add a test for subtract"}]'
    plan = generate_plan(_task_context(), complete_fn=lambda system, messages: canned)

    assert plan.steps == [
        PlanStep(file="tests/test_calculator.py", description="add a test for subtract")
    ]


# --- check_ambiguity (Master Document Epic B2) ---


def test_check_ambiguity_returns_none_for_clear_task():
    canned = '{"ambiguous": false, "question": null}'
    question = check_ambiguity(_task_context(), complete_fn=lambda system, messages: canned)

    assert question is None


def test_check_ambiguity_returns_specific_question_for_ambiguous_task():
    canned = (
        '{"ambiguous": true, "question": "Which aspect of the calculator should be '
        'improved -- error handling, new operations, or something else?"}'
    )
    question = check_ambiguity(
        _task_context(task="improve the calculator"),
        complete_fn=lambda system, messages: canned,
    )

    assert question == (
        "Which aspect of the calculator should be improved -- error handling, "
        "new operations, or something else?"
    )


def test_check_ambiguity_strips_surrounding_prose_and_fences():
    canned = (
        "Here you go:\n```json\n"
        '{"ambiguous": true, "question": "Which behavior should change?"}\n'
        "```\n"
    )
    question = check_ambiguity(_task_context(), complete_fn=lambda system, messages: canned)

    assert question == "Which behavior should change?"


def test_check_ambiguity_fails_open_on_malformed_json():
    canned = "Sure, I'll take a look."  # no JSON object at all
    question = check_ambiguity(_task_context(), complete_fn=lambda system, messages: canned)

    assert question is None


def test_check_ambiguity_fails_open_when_ambiguous_true_but_question_missing():
    canned = '{"ambiguous": true}'
    question = check_ambiguity(_task_context(), complete_fn=lambda system, messages: canned)

    assert question is None


def test_check_ambiguity_fails_open_on_invalid_json_syntax():
    canned = '{"ambiguous": true, "question": }'
    question = check_ambiguity(_task_context(), complete_fn=lambda system, messages: canned)

    assert question is None
