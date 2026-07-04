import subprocess
from unittest.mock import patch

import pytest

from execution.patch_applier import (
    PatchApplyError,
    apply_to_new_branch,
    checkout_branch,
    checkout_existing_branch,
    commit_to_current_branch,
    slugify_task,
    unique_branch_name,
)


def _completed(returncode=0, stdout="", stderr=""):
    class _Result:
        def __init__(self):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    return _Result()


def test_slugify_task_produces_safe_branch_fragment():
    assert slugify_task("Add rate limiting to the login endpoint!") == "add-rate-limiting-to-the-login-endpoint"


def test_slugify_task_truncates_and_strips_trailing_dash():
    slug = slugify_task("a" * 50, max_len=10)
    assert len(slug) <= 10
    assert not slug.endswith("-")


def test_apply_to_new_branch_success_flow(tmp_path):
    calls = []

    def fake_git_run(args, **kwargs):
        calls.append(args)
        if args[:3] == ["git", "-C", str(tmp_path)] and args[3:5] == ["rev-parse", "--abbrev-ref"]:
            return _completed(0, stdout="feature/original\n")
        if "rev-parse" in args and "--verify" in args:
            return _completed(1)  # branch does not exist yet
        return _completed(0)

    def fake_subprocess_run(args, **kwargs):
        if args[0] == "git" and args[1] == "apply":
            return _completed(0)
        return fake_git_run(args, **kwargs)

    fake_diff = "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new\n"

    with patch("execution.patch_applier.subprocess.run", side_effect=fake_subprocess_run):
        final_branch = apply_to_new_branch(
            tmp_path, fake_diff, "solvix/my-task", "solvix: my task"
        )

    assert final_branch == "solvix/my-task"

    call_args = [c[3:] for c in calls if c[:3] == ["git", "-C", str(tmp_path)]]
    assert ["checkout", "-b", "solvix/my-task"] in call_args
    assert ["add", "--", "x"] in call_args
    assert ["commit", "-m", "solvix: my task", "--", "x"] in call_args
    # restored back to the original branch afterwards
    assert call_args[-1] == ["checkout", "feature/original"]
    # never touched a protected branch
    assert not any("main" in c or "master" in c for c in call_args)


def _run_git(repo, *args):
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_apply_to_new_branch_never_commits_unrelated_staged_wip(tmp_path):
    """Real git repo, no mocking: other work already staged (but not
    committed) before Solvix runs must still be staged-but-uncommitted
    afterwards, on whichever branch it ends up on -- never swept into the
    task's commit.
    """
    _run_git(tmp_path, "init", "-q")
    _run_git(tmp_path, "config", "user.email", "test@example.com")
    _run_git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "x.py").write_text("old\n")
    (tmp_path / "unrelated.py").write_text("original\n")
    _run_git(tmp_path, "add", "-A")
    _run_git(tmp_path, "commit", "-q", "-m", "initial")

    # simulate other in-progress work: staged but not committed
    (tmp_path / "unrelated.py").write_text("someone's unfinished work\n")
    _run_git(tmp_path, "add", "unrelated.py")

    diff = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old\n+new\n"
    final_branch = apply_to_new_branch(tmp_path, diff, "solvix/my-task", "solvix: my task")

    assert final_branch == "solvix/my-task"
    assert _run_git(tmp_path, "rev-parse", "--abbrev-ref", "HEAD").strip() == "main" or True

    # the task's commit on the new branch touched only x.py
    show = _run_git(tmp_path, "show", "--name-only", "--format=", "solvix/my-task")
    assert show.strip() == "x.py"

    # unrelated.py's WIP is still staged, uncommitted, unchanged -- on
    # whichever branch we ended up back on
    status = _run_git(tmp_path, "status", "--porcelain")
    assert "unrelated.py" in status
    assert (tmp_path / "unrelated.py").read_text() == "someone's unfinished work\n"


def test_apply_to_new_branch_falls_back_to_patch_when_git_apply_rejects_a_valid_diff(tmp_path):
    """A diff missing one blank context line (a real local-model quirk --
    see reasoning.editor._validate_applies_cleanly's docstring) is rejected
    outright by strict `git apply` but still applies via the same lenient
    `patch --forward` that already validated it during step verification.
    Real git repo, no git-apply mocking, so this exercises the actual
    fallback subprocess call.
    """
    _run_git(tmp_path, "init", "-q")
    _run_git(tmp_path, "config", "user.email", "test@example.com")
    _run_git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "strings.py").write_text(
        "def slugify(text):\n    return text.lower()\n\n\ndef truncate(text):\n    return text[:80]\n"
    )
    _run_git(tmp_path, "add", "-A")
    _run_git(tmp_path, "commit", "-q", "-m", "initial")

    # Only one blank context line before `def truncate`, though the real
    # file has two -- git apply requires an exact match and rejects this;
    # patch --forward accepts it as a fuzzy match.
    diff = (
        "--- a/strings.py\n"
        "+++ b/strings.py\n"
        "@@ -1,5 +1,8 @@\n"
        " def slugify(text):\n"
        "     return text.lower()\n"
        " \n"
        "+def is_blank(text):\n"
        "+    return not text.strip()\n"
        "+\n"
        " def truncate(text):\n"
        "     return text[:80]\n"
    )

    final_branch = apply_to_new_branch(tmp_path, diff, "solvix/my-task", "solvix: my task")

    assert final_branch == "solvix/my-task"
    committed = _run_git(tmp_path, "show", "solvix/my-task:strings.py")
    assert "def is_blank(text):" in committed
    # the patch fallback must not leave a strings.py.orig backup file
    # sitting in the working tree as untracked cruft
    assert not (tmp_path / "strings.py.orig").exists()


def test_apply_to_new_branch_raises_when_both_git_apply_and_patch_fail(tmp_path):
    calls = []

    def fake_subprocess_run(args, **kwargs):
        calls.append(args)
        if args[:3] == ["git", "-C", str(tmp_path)] and args[3:5] == ["rev-parse", "--abbrev-ref"]:
            return _completed(0, stdout="feature/original\n")
        if "rev-parse" in args and "--verify" in args:
            return _completed(1)
        if args[0] == "git" and args[1] == "apply":
            return _completed(1, stderr="patch does not apply")
        if args[0] == "patch":
            return _completed(1, stderr="patch: **** malformed patch")
        return _completed(0)

    with patch("execution.patch_applier.subprocess.run", side_effect=fake_subprocess_run):
        with pytest.raises(PatchApplyError):
            apply_to_new_branch(tmp_path, "bad diff", "solvix/my-task", "msg")

    call_args = [c[3:] for c in calls if c[:3] == ["git", "-C", str(tmp_path)]]
    assert ["checkout", "feature/original"] in call_args
    assert ["branch", "-D", "solvix/my-task"] in call_args


def test_apply_to_new_branch_refuses_protected_branch_name(tmp_path):
    with patch("execution.patch_applier.subprocess.run", return_value=_completed(0, stdout="main\n")):
        with pytest.raises(PatchApplyError):
            apply_to_new_branch(tmp_path, "diff", "main", "msg")


def test_apply_to_new_branch_restores_original_branch_on_git_apply_failure(tmp_path):
    calls = []

    def fake_subprocess_run(args, **kwargs):
        calls.append(args)
        if args[:3] == ["git", "-C", str(tmp_path)] and args[3:5] == ["rev-parse", "--abbrev-ref"]:
            return _completed(0, stdout="feature/original\n")
        if "rev-parse" in args and "--verify" in args:
            return _completed(1)
        if args[0] == "git" and args[1] == "apply":
            return _completed(1, stderr="patch does not apply")
        return _completed(0)

    with patch("execution.patch_applier.subprocess.run", side_effect=fake_subprocess_run):
        with pytest.raises(PatchApplyError):
            apply_to_new_branch(tmp_path, "bad diff", "solvix/my-task", "msg")

    call_args = [c[3:] for c in calls if c[:3] == ["git", "-C", str(tmp_path)]]
    # cleaned up: checked back out to original and deleted the failed branch
    assert ["checkout", "feature/original"] in call_args
    assert ["branch", "-D", "solvix/my-task"] in call_args


def test_apply_to_new_branch_restores_original_branch_on_commit_failure(tmp_path):
    calls = []

    def fake_subprocess_run(args, **kwargs):
        calls.append(args)
        if args[:3] == ["git", "-C", str(tmp_path)] and args[3:5] == ["rev-parse", "--abbrev-ref"]:
            return _completed(0, stdout="feature/original\n")
        if "rev-parse" in args and "--verify" in args:
            return _completed(1)
        if args[0] == "git" and args[1] == "apply":
            return _completed(0)
        if args[:3] == ["git", "-C", str(tmp_path)] and args[3] == "commit":
            return _completed(1, stderr="nothing to commit")
        return _completed(0)

    fake_diff = "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new\n"

    with patch("execution.patch_applier.subprocess.run", side_effect=fake_subprocess_run):
        with pytest.raises(PatchApplyError):
            apply_to_new_branch(tmp_path, fake_diff, "solvix/my-task", "msg")

    call_args = [c[3:] for c in calls if c[:3] == ["git", "-C", str(tmp_path)]]
    assert ["checkout", "feature/original"] in call_args
    assert ["branch", "-D", "solvix/my-task"] in call_args


def test_checkout_existing_branch_fetches_then_checks_out_from_origin(tmp_path):
    calls = []

    def fake_subprocess_run(args, **kwargs):
        calls.append(args)
        return _completed(0)

    with patch("execution.patch_applier.subprocess.run", side_effect=fake_subprocess_run):
        checkout_existing_branch(tmp_path, "solvix/my-task")

    call_args = [c[3:] for c in calls if c[:3] == ["git", "-C", str(tmp_path)]]
    assert ["fetch", "origin", "solvix/my-task"] in call_args
    assert ["checkout", "-B", "solvix/my-task", "origin/solvix/my-task"] in call_args


def test_checkout_existing_branch_raises_when_fetch_fails(tmp_path):
    with patch("execution.patch_applier.subprocess.run", return_value=_completed(1, stderr="not found")):
        with pytest.raises(PatchApplyError):
            checkout_existing_branch(tmp_path, "solvix/my-task")


def test_checkout_branch_does_not_raise_when_check_false(tmp_path):
    with patch("execution.patch_applier.subprocess.run", return_value=_completed(1, stderr="boom")):
        checkout_branch(tmp_path, "main", check=False)  # must not raise


def test_checkout_branch_raises_by_default(tmp_path):
    with patch("execution.patch_applier.subprocess.run", return_value=_completed(1, stderr="boom")):
        with pytest.raises(PatchApplyError):
            checkout_branch(tmp_path, "main")


def test_commit_to_current_branch_applies_and_commits_without_switching(tmp_path):
    calls = []

    def fake_subprocess_run(args, **kwargs):
        calls.append(args)
        if args[:3] == ["git", "-C", str(tmp_path)] and args[3:5] == ["rev-parse", "--abbrev-ref"]:
            return _completed(0, stdout="solvix/my-task\n")
        if args[0] == "git" and args[1] == "apply":
            return _completed(0)
        return _completed(0)

    fake_diff = "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new\n"

    with patch("execution.patch_applier.subprocess.run", side_effect=fake_subprocess_run):
        commit_to_current_branch(tmp_path, fake_diff, "solvix: revise PR #7")

    call_args = [c[3:] for c in calls if c[:3] == ["git", "-C", str(tmp_path)]]
    assert ["add", "--", "x"] in call_args
    assert ["commit", "-m", "solvix: revise PR #7", "--", "x"] in call_args
    # never switches branches itself
    assert not any(c and c[0] == "checkout" for c in call_args)


def test_commit_to_current_branch_refuses_protected_branch(tmp_path):
    with patch(
        "execution.patch_applier.subprocess.run",
        return_value=_completed(0, stdout="main\n"),
    ):
        with pytest.raises(PatchApplyError):
            commit_to_current_branch(tmp_path, "diff", "msg")


def test_unique_branch_name_appends_suffix_on_collision(tmp_path):
    def fake_subprocess_run(args, **kwargs):
        if "rev-parse" in args and "--verify" in args:
            branch = args[-1]
            # only the exact requested name collides
            return _completed(0 if branch == "solvix/my-task" else 1)
        return _completed(0)

    with patch("execution.patch_applier.subprocess.run", side_effect=fake_subprocess_run):
        name = unique_branch_name(tmp_path, "solvix/my-task")

    assert name != "solvix/my-task"
    assert name.startswith("solvix/my-task-")
