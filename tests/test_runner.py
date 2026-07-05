import pytest

from execution.test_runner import TestResult, apply_diff, run_tests, run_tests_on_diff
from reasoning.editor import Diff, DiffGenerationError


def _write_passing_repo(tmp_path):
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    return tmp_path


def _write_failing_repo(tmp_path):
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 999\n"
    )
    return tmp_path


def test_run_tests_reports_pass_for_passing_suite(tmp_path):
    _write_passing_repo(tmp_path)

    result = run_tests(tmp_path, test_command="pytest -q")

    assert isinstance(result, TestResult)
    assert result.passed is True
    assert result.exit_code == 0
    assert "1 passed" in result.output


def test_run_tests_reports_failure_with_output(tmp_path):
    _write_failing_repo(tmp_path)

    result = run_tests(tmp_path, test_command="pytest -q")

    assert result.passed is False
    assert result.exit_code == 1
    assert "1 failed" in result.output
    assert "test_add" in result.output


def test_run_tests_reports_error_for_missing_command(tmp_path):
    _write_passing_repo(tmp_path)

    result = run_tests(tmp_path, test_command="not-a-real-test-runner")

    assert result.passed is False
    assert result.exit_code != 0


def test_run_tests_on_diff_applies_diff_to_scratch_copy_and_leaves_original_untouched(tmp_path):
    _write_failing_repo(tmp_path)
    original_content = (tmp_path / "calc.py").read_text()

    fix_diff = Diff(
        target_file="calc.py",
        diff_text=(
            "--- a/calc.py\n"
            "+++ b/calc.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def add(a, b):\n"
            "-    return a + b\n"
            "+    return 999\n"
        ),
        is_new_file=False,
    )

    result = run_tests_on_diff(tmp_path, fix_diff, test_command="pytest -q")

    assert result.passed is True
    assert result.exit_code == 0
    assert (tmp_path / "calc.py").read_text() == original_content


def test_run_tests_on_diff_reports_failure_from_scratch_copy(tmp_path):
    _write_passing_repo(tmp_path)

    breaking_diff = Diff(
        target_file="calc.py",
        diff_text=(
            "--- a/calc.py\n"
            "+++ b/calc.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def add(a, b):\n"
            "-    return a + b\n"
            "+    return a - b\n"
        ),
        is_new_file=False,
    )

    result = run_tests_on_diff(tmp_path, breaking_diff, test_command="pytest -q")

    assert result.passed is False
    assert "1 failed" in result.output


def test_apply_diff_creates_missing_parent_directory_even_when_is_new_file_is_false(tmp_path):
    """SLX-C9: is_new_file is only as reliable as reasoning.editor's
    "/dev/null" heuristic for detecting a new-file diff. A pure-addition
    diff for a file whose containing directory doesn't exist yet, but that
    the model didn't mark as a new file, must still apply cleanly --
    apply_diff can't rely on is_new_file to decide whether to create the
    directory.
    """
    diff = Diff(
        target_file="utils/strings.py",
        diff_text=(
            "--- a/utils/strings.py\n"
            "+++ b/utils/strings.py\n"
            "@@ -0,0 +1,2 @@\n"
            "+def count_vowels(text):\n"
            "+    return sum(1 for c in text.lower() if c in 'aeiou')\n"
        ),
        is_new_file=False,
    )

    apply_diff(tmp_path, diff)

    assert (tmp_path / "utils" / "strings.py").exists()
    assert "count_vowels" in (tmp_path / "utils" / "strings.py").read_text()


def test_apply_diff_raises_diff_generation_error_not_raw_subprocess_error(tmp_path):
    """SLX-C9: a `patch` failure during the real apply must surface as the
    project's own DiffGenerationError -- the same exception
    execution.orchestrator's outer retry loop already knows how to absorb
    into a clean StepResult -- not a raw subprocess.CalledProcessError that
    would crash the whole task.
    """
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    diff = Diff(
        target_file="calc.py",
        diff_text=(
            "--- a/calc.py\n"
            "+++ b/calc.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def add(a, b):\n"
            "-    return a * b\n"  # doesn't match calc.py's actual content
            "+    return a - b\n"
        ),
        is_new_file=False,
    )

    with pytest.raises(DiffGenerationError):
        apply_diff(tmp_path, diff)
