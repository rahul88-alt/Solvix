"""Builds and opens the pull request for a completed task (Master Document
7.2/7.3, Epic D1): pushes the branch execution.patch_applier committed to
and shells out to the already-authenticated `gh` CLI to open the PR, rather
than reimplementing GitHub's REST API directly.

The PR title/body are assembled from data the pipeline already produced --
the task text, reasoning.planner.Plan, and the execution.orchestrator.
StepResult list -- not reformatted from a separate reasoning trace, so the
PR description can never drift from what actually ran.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from execution.orchestrator import StepResult
from reasoning.planner import Plan
from reasoning.task_input import TaskContext

_TITLE_MAX_LEN = 72


class PRBuildError(RuntimeError):
    """Raised when pushing the branch or invoking `gh pr create` fails."""


@dataclass(frozen=True)
class PullRequestResult:
    url: str
    title: str
    body: str


def _run(args: list[str], cwd: str | Path) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=str(cwd), capture_output=True, text=True)


def _push_branch(repo_root: str | Path, branch_name: str) -> None:
    result = _run(["git", "push", "-u", "origin", branch_name], cwd=repo_root)
    if result.returncode != 0:
        raise PRBuildError(f"git push failed: {result.stderr.strip()}")


def build_title(task_context: TaskContext) -> str:
    task = task_context.task.strip().replace("\n", " ")
    title = f"solvix: {task}"
    if len(title) > _TITLE_MAX_LEN:
        title = title[: _TITLE_MAX_LEN - 1].rstrip() + "…"
    return title


def _format_test_results(step_results: tuple[StepResult, ...]) -> str:
    lines = []
    for i, step_result in enumerate(step_results, start=1):
        file_name = step_result.diff.target_file if step_result.diff else "(no diff)"
        if step_result.test_result is not None:
            status = "passed" if step_result.test_result.passed else "failed"
            lines.append(f"- Step {i} ({file_name}): tests {status}, {step_result.attempts} attempt(s)")
        else:
            lines.append(f"- Step {i} ({file_name}): no test run ({step_result.attempts} attempt(s))")
    return "\n".join(lines)


def build_body(
    task_context: TaskContext,
    plan: Plan,
    step_results: tuple[StepResult, ...],
) -> str:
    plan_lines = "\n".join(
        f"{i}. {step.file}: {step.description}" for i, step in enumerate(plan.steps, start=1)
    )
    files_changed = "\n".join(
        f"- {step_result.diff.target_file}" for step_result in step_results if step_result.diff
    )
    test_results = _format_test_results(step_results)

    return (
        f"## Task\n{task_context.task}\n\n"
        f"## Plan\n{plan_lines}\n\n"
        f"## Files changed\n{files_changed or '(none)'}\n\n"
        f"## Test results\n{test_results or '(none)'}\n\n"
        "---\nOpened automatically by Solvix.\n"
    )


def build_pr(
    repo_root: str | Path,
    branch_name: str,
    task_context: TaskContext,
    plan: Plan,
    step_results: tuple[StepResult, ...],
) -> PullRequestResult:
    """Push branch_name and open a PR for it via `gh pr create`, returning
    the created PR's URL.

    Assumes execution.patch_applier.apply_to_new_branch has already created
    and committed branch_name (this function never applies a diff or
    commits anything itself) and that the repo is authenticated with `gh`
    already (Master Document 7.2: "simplest path given it's already
    authenticated").
    """
    _push_branch(repo_root, branch_name)

    title = build_title(task_context)
    body = build_body(task_context, plan, step_results)

    result = _run(
        [
            "gh", "pr", "create",
            "--title", title,
            "--body", body,
            "--head", branch_name,
        ],
        cwd=repo_root,
    )
    if result.returncode != 0:
        raise PRBuildError(f"gh pr create failed: {result.stderr.strip()}")

    url = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    return PullRequestResult(url=url, title=title, body=body)
