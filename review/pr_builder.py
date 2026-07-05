"""Builds and opens the pull request for a completed task (Master Document
7.2/7.3, Epic D1/D2): pushes the branch execution.patch_applier committed to
and shells out to the already-authenticated `gh` CLI to open the PR, rather
than reimplementing GitHub's REST API directly.

The PR title/body are assembled from data the pipeline already produced --
the task text, reasoning.planner.Plan, and the execution.orchestrator.
TaskResult (including its per-step StepResults) -- not reformatted from a
separate reasoning trace, so the PR description can never drift from what
actually ran. This also means the reasoning-trace sections (plan-approval
flag, dangerous-ops confirmations, retry counts/reasons, needs-human-help
callouts) are only ever as good as the fields StepResult/TaskResult already
carry -- see execution.orchestrator for what's tracked and why.

The optional clarification argument threaded through build_body/build_pr
(Master Document Epic B2) follows the same rule: if cli.py's check_ambiguity
round shaped what task_context.task ended up meaning, that question/answer
belongs in "Key decisions" too, not just in whatever the user saw live in
their terminal.

format_test_results/format_key_decisions/format_needs_attention and
push_branch are public (not module-private) specifically so
review.pr_feedback (Epic D3, `solvix revise`) can reuse the same
formatting/push logic for a revision comment instead of reimplementing it --
see that module for how a revision's summary comment is scoped to just its
own round rather than restating the whole PR.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from execution.orchestrator import StepResult, TaskResult
from reasoning.planner import Clarification, Plan
from reasoning.task_input import TaskContext

_TITLE_MAX_LEN = 72

SOLVIX_COMMENT_MARKER = "Posted automatically by Solvix."


class PRBuildError(RuntimeError):
    """Raised when pushing the branch or invoking `gh pr create` fails."""


@dataclass(frozen=True)
class PullRequestResult:
    url: str
    title: str
    body: str


def _run(args: list[str], cwd: str | Path) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=str(cwd), capture_output=True, text=True, stdin=subprocess.DEVNULL)


def push_branch(repo_root: str | Path, branch_name: str) -> None:
    result = _run(["git", "push", "-u", "origin", branch_name], cwd=repo_root)
    if result.returncode != 0:
        raise PRBuildError(f"git push failed: {result.stderr.strip()}")


def build_title(task_context: TaskContext) -> str:
    task = task_context.task.strip().replace("\n", " ")
    title = f"solvix: {task}"
    if len(title) > _TITLE_MAX_LEN:
        title = title[: _TITLE_MAX_LEN - 1].rstrip() + "…"
    return title


def format_test_results(step_results: tuple[StepResult, ...]) -> str:
    lines = []
    for i, step_result in enumerate(step_results, start=1):
        file_name = step_result.diff.target_file if step_result.diff else "(no diff)"
        if step_result.test_result is not None:
            status = "passed" if step_result.test_result.passed else "failed"
            lines.append(f"- Step {i} ({file_name}): tests {status}, {step_result.attempts} attempt(s)")
            if not step_result.test_result.passed:
                last_output = step_result.test_result.output.strip()
                if last_output:
                    lines.append(f"  - final failure output: {last_output.splitlines()[-1]}")
        else:
            lines.append(f"- Step {i} ({file_name}): no test run ({step_result.attempts} attempt(s))")
    return "\n".join(lines)


def format_key_decisions(
    plan: Plan,
    step_results: tuple[StepResult, ...],
    clarification: Clarification | None = None,
) -> str:
    lines = []

    if clarification is not None:
        lines.append(
            f"- Clarification requested: '{clarification.question}' — "
            f"answered: '{clarification.answer}'"
        )

    if plan.requires_approval:
        lines.append(
            "- Plan flagged for approval before execution — "
            f"{'; '.join(plan.approval_reasons)}"
        )
    else:
        lines.append("- Plan was not flagged for approval (no sensitive paths, file count within limit).")

    for i, step_result in enumerate(step_results, start=1):
        file_name = step_result.diff.target_file if step_result.diff else "(no diff)"

        if step_result.requires_confirmation:
            lines.append(
                f"- Step {i} ({file_name}): dangerous-ops confirmation was required and declined — "
                f"{'; '.join(step_result.confirmation_reasons)}"
            )
        elif step_result.dangerous_ops_confirmed:
            lines.append(
                f"- Step {i} ({file_name}): dangerous-ops confirmation was required and approved — "
                f"{'; '.join(step_result.dangerous_ops_confirmed_reasons)}"
            )

        if step_result.attempts > 1:
            reasons = "; ".join(step_result.attempt_failures) or "no failure details recorded"
            lines.append(f"- Step {i} ({file_name}): needed {step_result.attempts} attempts — {reasons}")

    return "\n".join(lines) if lines else "(none)"


def format_needs_attention(task_result: TaskResult) -> str | None:
    flagged_steps: list[tuple[int, StepResult, list[str]]] = []
    for i, step_result in enumerate(task_result.step_results, start=1):
        reasons = []
        if step_result.needs_human_help:
            reasons.append(step_result.failure_reason or "exhausted its retry budget")
        if step_result.assertion_gaming_suspected:
            reasons.append(
                "suspected assertion-gaming (a retry appears to have rewritten a test's "
                f"expected value to match the code's actual output instead of fixing the "
                f"implementation) -- {step_result.assertion_gaming_details}"
            )
        if step_result.weak_test_coverage_suspected:
            reasons.append(
                "added test may not meaningfully cover the changed behavior -- "
                f"{step_result.weak_test_coverage_details}"
            )
        if reasons:
            flagged_steps.append((i, step_result, reasons))

    if not flagged_steps:
        return None

    lines = [
        "⚠️ The following step(s) were individually flagged and should be double-checked, "
        "even though the overall task completed:"
    ]
    for i, step_result, reasons in flagged_steps:
        file_name = step_result.diff.target_file if step_result.diff else "(no diff)"
        for reason in reasons:
            lines.append(f"- Step {i} ({file_name}): {reason}")
    return "\n".join(lines)


def build_body(
    task_context: TaskContext,
    plan: Plan,
    task_result: TaskResult,
    clarification: Clarification | None = None,
) -> str:
    step_results = task_result.step_results

    plan_lines = "\n".join(
        f"{i}. {step.file}: {step.description}" for i, step in enumerate(plan.steps, start=1)
    )
    files_changed = "\n".join(
        f"- {step_result.diff.target_file}" for step_result in step_results if step_result.diff
    )
    test_results = format_test_results(step_results)
    key_decisions = format_key_decisions(plan, step_results, clarification)

    sections = [
        f"## Task\n{task_context.task}",
        f"## Plan\n{plan_lines}",
        f"## Files changed\n{files_changed or '(none)'}",
        f"## Key decisions\n{key_decisions}",
        f"## Test results\n{test_results or '(none)'}",
    ]

    needs_attention = format_needs_attention(task_result)
    if needs_attention is not None:
        sections.append(f"## Needs attention\n{needs_attention}")

    sections.append("---\nOpened automatically by Solvix.\n")
    return "\n\n".join(sections)


def build_pr(
    repo_root: str | Path,
    branch_name: str,
    task_context: TaskContext,
    plan: Plan,
    task_result: TaskResult,
    clarification: Clarification | None = None,
) -> PullRequestResult:
    """Push branch_name and open a PR for it via `gh pr create`, returning
    the created PR's URL.

    Assumes execution.patch_applier.apply_to_new_branch has already created
    and committed branch_name (this function never applies a diff or
    commits anything itself) and that the repo is authenticated with `gh`
    already (Master Document 7.2: "simplest path given it's already
    authenticated").

    clarification, when given, is the check_ambiguity question/answer round
    (Master Document Epic B2) that shaped task_context.task, if one
    happened -- see format_key_decisions.
    """
    push_branch(repo_root, branch_name)

    title = build_title(task_context)
    body = build_body(task_context, plan, task_result, clarification)

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
