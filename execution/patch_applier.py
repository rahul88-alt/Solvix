"""Applies a task's diff on a fresh branch and commits it (Master Document
7.2/7.3, Epic D1): the only place in the pipeline allowed to run `git
checkout -b`, `git apply --check`-free application to the working tree, and
`git commit` against the user's real repo (everything upstream of this only
ever touches scratch copies -- see execution.test_runner).

`apply_to_new_branch` never checks out or modifies the branch the caller was
on when Solvix started: it records that branch, creates and switches to a
new one derived from the task, applies the diff and commits there, then
always switches back -- including on any failure partway through, so a
failed `git apply` or `git commit` never leaves the working tree stranded on
a half-finished branch.
"""

from __future__ import annotations

import re
import subprocess
import uuid
from pathlib import Path

_PROTECTED_BRANCHES = ("main", "master")


class PatchApplyError(RuntimeError):
    """Raised when a git operation needed to create/commit the branch fails."""


def _git(repo_root: str | Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args], capture_output=True, text=True
    )
    if check and result.returncode != 0:
        raise PatchApplyError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result


def get_current_branch(repo_root: str | Path) -> str:
    """Return the branch checked out in repo_root right now."""
    return _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()


def slugify_task(task: str, max_len: int = 40) -> str:
    """Turn free-text task description into a branch-name-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", task.lower()).strip("-")
    slug = slug[:max_len].rstrip("-")
    return slug or "task"


def _branch_exists(repo_root: str | Path, branch_name: str) -> bool:
    return _git(repo_root, "rev-parse", "--verify", "--quiet", branch_name, check=False).returncode == 0


_DIFF_TARGET_RE = re.compile(r"^\+\+\+ (?:b/(?P<path>.+)|(?P<devnull>/dev/null))$", re.MULTILINE)


def _touched_files(diff: str) -> list[str]:
    """Files the diff actually creates/modifies, parsed from its `+++ b/...`
    headers. Used so committing only ever stages what the diff touched,
    never sweeping in unrelated changes already sitting in the working tree
    or index (e.g. a repo with other in-progress, uncommitted work).
    """
    return [m.group("path") for m in _DIFF_TARGET_RE.finditer(diff) if m.group("path")]


def unique_branch_name(repo_root: str | Path, base_name: str) -> str:
    """base_name if it's free, otherwise base_name with a short random
    suffix appended -- retried until a free name is found. Never reuses or
    overwrites an existing branch.
    """
    if not _branch_exists(repo_root, base_name):
        return base_name
    for _ in range(10):
        candidate = f"{base_name}-{uuid.uuid4().hex[:6]}"
        if not _branch_exists(repo_root, candidate):
            return candidate
    raise PatchApplyError(f"could not find a unique branch name derived from {base_name!r}")


def apply_to_new_branch(
    repo_root: str | Path, diff: str, branch_name: str, commit_message: str
) -> str:
    """Create a new branch off the current HEAD, apply diff to the working
    tree, and commit it with commit_message.

    branch_name is a request, not a guarantee: if it collides with an
    existing branch, a short suffix is appended (see unique_branch_name) and
    the actual name used is returned. Refuses to run at all if branch_name
    (or its collision-resolved variant) is a protected branch name.

    The repo is always left back on whatever branch was checked out when
    this was called -- on success as well as on any failure partway through
    (a `git apply` that doesn't apply cleanly, or a `git commit` that fails)
    -- so a caller never has to remember to clean up after an error here.
    """
    original_branch = get_current_branch(repo_root)
    if branch_name in _PROTECTED_BRANCHES:
        raise PatchApplyError(f"refusing to create a protected branch name: {branch_name!r}")

    final_branch = unique_branch_name(repo_root, branch_name)

    _git(repo_root, "checkout", "-b", final_branch)
    try:
        diff_path = Path(repo_root) / ".solvix_patch.diff"
        diff_path.write_text(diff)
        try:
            apply_result = subprocess.run(
                ["git", "apply", str(diff_path)],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
            )
        finally:
            diff_path.unlink(missing_ok=True)

        if apply_result.returncode != 0:
            raise PatchApplyError(f"git apply failed: {apply_result.stderr.strip()}")

        touched = _touched_files(diff)
        if not touched:
            raise PatchApplyError("could not determine which file(s) the diff touches")
        # Scoping both the add and the commit itself to `touched` (rather
        # than `git add -A` / a pathspec-less `git commit`, which commits
        # the whole index) means any other content already staged or
        # modified in the working tree before this ran -- a repo with other
        # in-progress work -- is left exactly as it was, staged but
        # uncommitted, never swept into this commit.
        _git(repo_root, "add", "--", *touched)
        _git(repo_root, "commit", "-m", commit_message, "--", *touched)
    except Exception:
        _git(repo_root, "checkout", original_branch, check=False)
        _git(repo_root, "branch", "-D", final_branch, check=False)
        raise

    _git(repo_root, "checkout", original_branch)
    return final_branch
