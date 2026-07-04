"""Runs the target repo's test suite and captures the result (Master
Document 7.2/7.3, Epic C3/E1): the Execution & Verification stage's test
runner.

Tests are never run against the real on-disk repo directly, and never run
directly on the host: `run_tests` takes a diff (as produced by
reasoning.editor.propose_diff) and a repo_path, copies the repo into a
scratch directory, applies the diff there with the same `patch` mechanism
the Editor already uses to validate diffs, and runs the project's test
command inside a network-isolated Docker sandbox (execution.sandbox)
against that disposable copy -- so a bad diff or a flaky/destructive test
can never touch the real repo or the host machine.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from reasoning.editor import Diff

from execution.sandbox import Sandbox

_IGNORED_DIRS = shutil.ignore_patterns(
    ".git", "__pycache__", ".ruff_cache", ".pytest_cache", "*.egg-info", ".venv"
)


@dataclass(frozen=True)
class TestResult:
    __test__ = False  # not a pytest test case despite the name

    passed: bool
    output: str
    exit_code: int


def _apply_diff_in_scratch(scratch_root: Path, diff: Diff) -> None:
    target_path = scratch_root / diff.target_file
    if diff.is_new_file:
        target_path.parent.mkdir(parents=True, exist_ok=True)

    diff_path = scratch_root / ".solvix_scratch.diff"
    diff_path.write_text(diff.diff_text)
    try:
        subprocess.run(
            ["patch", "--forward", str(target_path), str(diff_path)],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            check=True,
        )
    finally:
        diff_path.unlink(missing_ok=True)


def run_tests(repo_path: str | Path, test_command: str = "pytest -q") -> TestResult:
    """Run test_command against repo_path inside an isolated Docker sandbox
    and capture the result.

    repo_path is run as-is (no scratch copy is made here) -- callers that
    need to verify a proposed diff without touching the real repo should go
    through `run_tests_on_diff` instead, which applies the diff to a scratch
    copy first. Docker unavailability raises (execution.sandbox.
    DockerUnavailableError) rather than silently falling back to running
    the command directly on the host.
    """
    with Sandbox(repo_path) as sandbox:
        result = sandbox.run(test_command)

    output = result.stdout + result.stderr
    return TestResult(passed=result.exit_code == 0, output=output, exit_code=result.exit_code)


def run_tests_on_diff(
    repo_path: str | Path, diff: Diff, test_command: str = "pytest -q"
) -> TestResult:
    """Apply diff to a scratch copy of repo_path and run test_command there.

    This is the entry point the edit flow should use after a diff has
    passed lint: it never mutates repo_path itself.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        scratch_root = Path(tmp_dir) / "repo"
        shutil.copytree(repo_path, scratch_root, ignore=_IGNORED_DIRS)
        _apply_diff_in_scratch(scratch_root, diff)
        return run_tests(scratch_root, test_command=test_command)
