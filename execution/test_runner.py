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

copy_repo and apply_diff are exposed publicly (not just used internally by
run_tests_on_diff) so execution.orchestrator.run_task can reuse the exact
same copy/patch mechanics for its own task-scoped working copy (Epic C2):
a later step's diff needs to be verified against everything earlier steps
in the same task already changed, not just the original on-disk repo.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from reasoning.editor import Diff, DiffGenerationError

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


def copy_repo(repo_path: str | Path, dest: str | Path) -> None:
    """Copy repo_path to dest, skipping the same VCS/cache/venv clutter a
    real repo accumulates (Master Document Epic C2/E3) -- shared by
    run_tests_on_diff's own per-call scratch copy and by
    execution.orchestrator.run_task's task-scoped working copy, which needs
    the exact same ignore list to avoid a subtly different scratch tree
    between the two.
    """
    shutil.copytree(repo_path, dest, ignore=_IGNORED_DIRS)


def apply_diff(root: Path, diff: Diff) -> None:
    """Apply diff to root in place via the same `patch` mechanism the
    Editor uses to validate a diff before this ever runs. Used both for a
    single call's disposable scratch copy (run_tests_on_diff) and for
    execution.orchestrator.run_task's task-scoped working copy, which
    persists a step's diff across the rest of that task's steps (Epic C2)
    -- the same function either way, since applying a diff has no notion of
    "scratch" or "persistent", only a target directory.

    The target's parent directory is always created up front (SLX-C9),
    regardless of diff.is_new_file: that flag is only as reliable as
    reasoning.editor's "/dev/null" heuristic for detecting a new-file diff,
    and a model that emits a plain-addition diff without that marker for a
    file whose containing directory doesn't exist yet would otherwise make
    `patch` fail outright, a failure reasoning.editor's own dry-run
    validation can never catch (its scratch target always lives directly in
    an already-existing tempdir, so there's never a missing directory to
    trip over there). Creating the directory unconditionally costs nothing
    when it already exists (exist_ok=True) and removes this blind spot
    without having to make the is_new_file heuristic itself any more
    reliable.

    A `patch` failure here is raised as reasoning.editor.DiffGenerationError
    (SLX-C9) -- the same exception execution.orchestrator's outer retry loop
    already knows how to absorb into a clean StepResult -- rather than
    letting the raw subprocess.CalledProcessError (from `check=True`) or an
    OSError (e.g. the `patch` binary itself missing) escape uncaught.
    """
    target_path = root / diff.target_file
    target_path.parent.mkdir(parents=True, exist_ok=True)

    diff_path = root / ".solvix_scratch.diff"
    diff_path.write_text(diff.diff_text)
    try:
        subprocess.run(
            ["patch", "--forward", str(target_path), str(diff_path)],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError) as error:
        raise DiffGenerationError(
            f"diff failed to apply to {diff.target_file}: {error}"
        ) from error
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
        copy_repo(repo_path, scratch_root)
        apply_diff(scratch_root, diff)
        return run_tests(scratch_root, test_command=test_command)
