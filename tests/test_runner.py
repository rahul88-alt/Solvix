from execution.test_runner import TestResult, run_tests, run_tests_on_diff
from reasoning.editor import Diff


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
