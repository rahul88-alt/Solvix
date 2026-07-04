from pathlib import Path

from reasoning.linter import LintResult, LintViolation, new_violations, run_linter


def test_run_linter_clean_file_has_no_violations(tmp_path):
    file_path = tmp_path / "clean.py"
    file_path.write_text("def add(a, b):\n    return a + b\n")

    result = run_linter(file_path)

    assert result.ran is True
    assert result.violations == ()


def test_run_linter_detects_unused_import(tmp_path):
    file_path = tmp_path / "bad.py"
    file_path.write_text("import os\n\n\ndef add(a, b):\n    return a + b\n")

    result = run_linter(file_path)

    assert result.ran is True
    assert any(v.rule == "F401" for v in result.violations)


def test_run_linter_missing_binary_returns_not_ran(monkeypatch):
    import subprocess

    def fake_run(*args, **kwargs):
        raise FileNotFoundError("ruff not found")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_linter(Path("anything.py"))

    assert result.ran is False
    assert result.violations == ()


def test_new_violations_returns_only_violations_absent_before():
    before = LintResult(
        violations=(LintViolation(rule="E501", line=3, message="line too long (100 > 88)"),),
        ran=True,
    )
    after = LintResult(
        violations=(
            LintViolation(rule="E501", line=3, message="line too long (100 > 88)"),
            LintViolation(rule="F401", line=1, message="`os` imported but unused"),
        ),
        ran=True,
    )

    result = new_violations(before, after)

    assert result == (LintViolation(rule="F401", line=1, message="`os` imported but unused"),)


def test_new_violations_shifted_line_not_counted_as_new():
    before = LintResult(
        violations=(LintViolation(rule="F401", line=1, message="`os` imported but unused"),),
        ran=True,
    )
    after = LintResult(
        violations=(LintViolation(rule="F401", line=5, message="`os` imported but unused"),),
        ran=True,
    )

    assert new_violations(before, after) == ()


def test_new_violations_returns_empty_when_lint_did_not_run():
    not_ran = LintResult(violations=(), ran=False)
    ran = LintResult(violations=(LintViolation(rule="F401", line=1, message="x"),), ran=True)

    assert new_violations(not_ran, ran) == ()
    assert new_violations(ran, not_ran) == ()
