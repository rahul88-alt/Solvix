import pytest

from context.assembler import FileScore, RetrievalResult
from reasoning.editor import Diff, DiffGenerationError, propose_diff
from reasoning.llm_client import OllamaUnavailableError
from reasoning.planner import PlanStep
from reasoning.task_input import TaskContext

_ORIGINAL_CONTENT = "def subtract(a, b):\n    return a - b\n"

_CLEAN_DIFF = (
    "--- a/calculator.py\n"
    "+++ b/calculator.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def subtract(a, b):\n"
    "-    return a - b\n"
    "+    return a - b if isinstance(a, (int, float)) else None\n"
)

_NEW_FILE_DIFF = (
    "--- /dev/null\n"
    "+++ b/newfile.py\n"
    "@@ -0,0 +1,2 @@\n"
    "+def foo():\n"
    "+    return 1\n"
)


def _task_context():
    retrieval = RetrievalResult(
        files=[FileScore(file_path="calculator.py", score=1.0, reasons=("test",))],
        related_files=[],
    )
    return TaskContext(task="fix the subtract function", retrieval=retrieval)


def _step():
    return PlanStep(file="calculator.py", description="fix subtract to handle negatives")


def test_propose_diff_parses_clean_diff():
    diff = propose_diff(
        _step(), _ORIGINAL_CONTENT, _task_context(), complete_fn=lambda system, messages: _CLEAN_DIFF
    )

    assert isinstance(diff, Diff)
    assert diff.target_file == "calculator.py"
    assert diff.is_new_file is False
    assert diff.diff_text.strip() == _CLEAN_DIFF.strip()


def test_propose_diff_strips_markdown_fences():
    fenced = f"```diff\n{_CLEAN_DIFF}```"
    diff = propose_diff(
        _step(), _ORIGINAL_CONTENT, _task_context(), complete_fn=lambda system, messages: fenced
    )

    assert diff.diff_text.strip() == _CLEAN_DIFF.strip()


def test_propose_diff_detects_new_file_creation():
    new_file_step = PlanStep(file="newfile.py", description="add a new helper module")
    diff = propose_diff(
        new_file_step, "", _task_context(), complete_fn=lambda system, messages: _NEW_FILE_DIFF
    )

    assert diff.is_new_file is True
    assert diff.target_file == "newfile.py"


def test_propose_diff_retries_once_on_malformed_output_and_recovers():
    responses = iter(
        [
            "Sure, I'll fix subtract by using isinstance checks.",  # no diff at all
            _CLEAN_DIFF,  # corrected
        ]
    )
    calls = []

    def fake_complete(system, messages):
        calls.append(messages)
        return next(responses)

    diff = propose_diff(_step(), _ORIGINAL_CONTENT, _task_context(), complete_fn=fake_complete)

    assert len(calls) == 2
    assert "corrected unified diff" in calls[1][-1]["content"]
    assert diff.diff_text.strip() == _CLEAN_DIFF.strip()


def test_propose_diff_retries_once_when_diff_does_not_apply_cleanly():
    bad_diff = (
        "--- a/calculator.py\n"
        "+++ b/calculator.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def subtract(a, b):\n"
        "-    return a * b\n"  # doesn't match actual content
        "+    return a - b if isinstance(a, (int, float)) else None\n"
    )
    responses = iter([bad_diff, _CLEAN_DIFF])
    calls = []

    def fake_complete(system, messages):
        calls.append(messages)
        return next(responses)

    diff = propose_diff(_step(), _ORIGINAL_CONTENT, _task_context(), complete_fn=fake_complete)

    assert len(calls) == 2
    assert "did not apply cleanly" in calls[1][-1]["content"]
    assert diff.diff_text.strip() == _CLEAN_DIFF.strip()


def test_propose_diff_converts_ollama_unavailable_into_diff_generation_error():
    """SLX-F4: a mid-run Ollama outage (e.g. it was stopped between attempts)
    must surface as the existing DiffGenerationError failure mode -- which
    execute_step_with_verification's outer retry loop already knows how to
    absorb into a needs_human_help StepResult -- not a raw exception.
    """

    def fake_complete(system, messages):
        raise OllamaUnavailableError("lost connection to Ollama at http://localhost:11434/v1")

    with pytest.raises(DiffGenerationError):
        propose_diff(_step(), _ORIGINAL_CONTENT, _task_context(), complete_fn=fake_complete)


def test_propose_diff_raises_after_retry_still_malformed():
    with pytest.raises(DiffGenerationError):
        propose_diff(
            _step(),
            _ORIGINAL_CONTENT,
            _task_context(),
            complete_fn=lambda system, messages: "still no diff here, sorry",
        )


def test_propose_diff_raises_after_retry_still_does_not_apply():
    bad_diff = (
        "--- a/calculator.py\n"
        "+++ b/calculator.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def subtract(a, b):\n"
        "-    return a * b\n"
        "+    return a - b if isinstance(a, (int, float)) else None\n"
    )

    with pytest.raises(DiffGenerationError, match="did not apply cleanly"):
        propose_diff(
            _step(), _ORIGINAL_CONTENT, _task_context(), complete_fn=lambda system, messages: bad_diff
        )


def _diff_with_unused_import():
    return (
        "--- a/calculator.py\n"
        "+++ b/calculator.py\n"
        "@@ -1,2 +1,2 @@\n"
        "+import os\n"
        " def subtract(a, b):\n"
        "-    return a - b\n"
        "+    return a - b if isinstance(a, (int, float)) else None\n"
    )


def test_propose_diff_clean_lint_has_no_warnings():
    diff = propose_diff(
        _step(), _ORIGINAL_CONTENT, _task_context(), complete_fn=lambda system, messages: _CLEAN_DIFF
    )

    assert diff.lint_warnings == ()


def test_propose_diff_retries_once_when_diff_introduces_new_lint_violation():
    responses = iter([_diff_with_unused_import(), _CLEAN_DIFF])
    calls = []

    def fake_complete(system, messages):
        calls.append(messages)
        return next(responses)

    diff = propose_diff(_step(), _ORIGINAL_CONTENT, _task_context(), complete_fn=fake_complete)

    assert len(calls) == 2
    assert "new lint violations" in calls[1][-1]["content"]
    assert "F401" in calls[1][-1]["content"]
    assert diff.lint_warnings == ()
    assert diff.diff_text.strip() == _CLEAN_DIFF.strip()


def test_propose_diff_accepts_with_warning_when_still_violating_after_retry():
    bad_diff = _diff_with_unused_import()
    calls = []

    def fake_complete(system, messages):
        calls.append(messages)
        return bad_diff

    diff = propose_diff(_step(), _ORIGINAL_CONTENT, _task_context(), complete_fn=fake_complete)

    assert len(calls) == 2
    assert "import os" in diff.diff_text
    assert len(diff.lint_warnings) == 1
    assert "F401" in diff.lint_warnings[0]


def test_propose_diff_respects_project_ruff_config_via_repo_root(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[tool.ruff]\nline-length = 20\n\n[tool.ruff.lint]\nselect = [\"E\", \"F\"]\n"
    )
    long_line_diff = (
        "--- a/calculator.py\n"
        "+++ b/calculator.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def subtract(a, b):\n"
        "-    return a - b\n"
        "+    return a - b  # a much longer trailing comment than twenty characters\n"
    )

    diff_without_root = propose_diff(
        _step(),
        _ORIGINAL_CONTENT,
        _task_context(),
        complete_fn=lambda system, messages: long_line_diff,
    )
    assert diff_without_root.lint_warnings == ()

    calls = []

    def fake_complete(system, messages):
        calls.append(messages)
        return long_line_diff

    diff_with_root = propose_diff(
        _step(),
        _ORIGINAL_CONTENT,
        _task_context(),
        complete_fn=fake_complete,
        repo_root=tmp_path,
    )

    assert len(calls) == 2
    assert len(diff_with_root.lint_warnings) == 1
    assert "E501" in diff_with_root.lint_warnings[0]


def test_propose_diff_skips_lint_gate_for_non_python_file():
    non_py_step = PlanStep(file="README.md", description="update readme")
    diff_text = (
        "--- a/README.md\n"
        "+++ b/README.md\n"
        "@@ -1,1 +1,1 @@\n"
        "-old\n"
        "+new\n"
    )
    diff = propose_diff(
        non_py_step, "old\n", _task_context(), complete_fn=lambda system, messages: diff_text
    )

    assert diff.lint_warnings == ()
