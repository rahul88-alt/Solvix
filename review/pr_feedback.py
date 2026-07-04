"""Fetches conversational PR feedback and posts revision summaries back to
the PR (Master Document 7.2, Epic D3): reuses the already-authenticated
`gh` CLI exactly like review.pr_builder's PR creation does, and reuses its
comment-formatting helpers (format_test_results/format_key_decisions/
format_needs_attention) so a revision summary reads consistently with the
PR's original body instead of inventing separate formatting.

fetch_pr_feedback treats a PR's *branch* and its most recent *human*
comment as the two pieces of state a revision needs: the branch tells
cli.revise what to check out, and the comment becomes the new task
description fed through the pipeline unchanged (context assembler ->
planner -> editor -> sandbox verify -> retry loop). "Most recent human
comment" excludes comments Solvix itself posted (identified by
review.pr_builder.SOLVIX_COMMENT_MARKER, not by GitHub login -- `gh` posts
as whatever account is authenticated, the same account a human reviewer
might also be commenting from, so login can't distinguish the two). Without
this filter, revising the same PR a second time would feed Solvix's own
round-1 summary back into round 2 as if it were the human's request.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from execution.orchestrator import TaskResult
from reasoning.planner import Plan
from review.pr_builder import (
    SOLVIX_COMMENT_MARKER,
    format_key_decisions,
    format_needs_attention,
    format_test_results,
)


class PRFeedbackError(RuntimeError):
    """Raised when fetching PR feedback or posting a revision comment fails."""


@dataclass(frozen=True)
class PRFeedback:
    branch: str
    comment_body: str
    url: str


def _run(args: list[str], cwd: str | Path) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=str(cwd), capture_output=True, text=True)


def fetch_pr_feedback(repo_root: str | Path, pr_number: int) -> PRFeedback:
    """Fetch pr_number's branch name and most recent human comment via
    `gh pr view`. Raises PRFeedbackError if `gh` fails, the output can't be
    parsed, or there is no human comment to revise from (e.g. a fresh PR
    nobody has commented on yet).
    """
    result = _run(
        ["gh", "pr", "view", str(pr_number), "--json", "headRefName,comments,url"],
        cwd=repo_root,
    )
    if result.returncode != 0:
        raise PRFeedbackError(f"gh pr view failed: {result.stderr.strip()}")

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise PRFeedbackError(f"could not parse `gh pr view` output: {error}")

    branch = data.get("headRefName")
    if not branch:
        raise PRFeedbackError(f"PR #{pr_number} has no headRefName in `gh pr view` output")

    url = data.get("url")
    if not url:
        raise PRFeedbackError(f"PR #{pr_number} has no url in `gh pr view` output")

    comments = data.get("comments") or []
    human_comments = [c for c in comments if SOLVIX_COMMENT_MARKER not in (c.get("body") or "")]
    if not human_comments:
        raise PRFeedbackError(
            f"PR #{pr_number} has no human feedback comment to revise from -- "
            "post a comment describing the requested change first "
            f"(e.g. `gh pr comment {pr_number} --body \"...\"` or via the GitHub UI)"
        )

    comment_body = (human_comments[-1].get("body") or "").strip()
    if not comment_body:
        raise PRFeedbackError(f"PR #{pr_number}'s latest human comment is empty")

    return PRFeedback(branch=branch, comment_body=comment_body, url=url)


def build_revision_comment(plan: Plan, task_result: TaskResult, feedback_comment: str) -> str:
    """Summarize a single revision round for posting back to the PR --
    scoped to what changed in this round (the feedback that was addressed,
    this round's plan/files/decisions/tests), not a full re-statement of the
    whole PR. Covers both outcomes: a successful revision (pushed as an
    additional commit) and one that didn't complete (nothing pushed).
    """
    step_results = task_result.step_results

    plan_lines = "\n".join(
        f"{i}. {step.file}: {step.description}" for i, step in enumerate(plan.steps, start=1)
    )
    files_changed = "\n".join(
        f"- {step_result.diff.target_file}" for step_result in step_results if step_result.diff
    )

    if task_result.success:
        outcome = "This revision completed successfully and has been pushed as a new commit on this PR's branch."
    elif task_result.needs_human_help:
        outcome = f"This revision needs human help: {task_result.reason or 'see step details below'}."
        if task_result.culprit_step is not None:
            outcome += f" Culprit step: {task_result.culprit_step.file}"
    else:
        outcome = f"This revision did not complete successfully: {task_result.reason or 'see step details below'}."

    sections = [
        f"## Feedback addressed in this revision\n{feedback_comment}",
        f"## Outcome\n{outcome}",
        f"## Plan for this revision\n{plan_lines or '(no steps)'}",
        f"## Files changed\n{files_changed or '(none)'}",
        f"## Key decisions\n{format_key_decisions(plan, step_results)}",
        f"## Test results\n{format_test_results(step_results) or '(none)'}",
    ]

    needs_attention = format_needs_attention(task_result)
    if needs_attention is not None:
        sections.append(f"## Needs attention\n{needs_attention}")

    sections.append(f"---\n{SOLVIX_COMMENT_MARKER}\n")
    return "\n\n".join(sections)


def post_pr_comment(repo_root: str | Path, pr_number: int, body: str) -> None:
    result = _run(["gh", "pr", "comment", str(pr_number), "--body", body], cwd=repo_root)
    if result.returncode != 0:
        raise PRFeedbackError(f"gh pr comment failed: {result.stderr.strip()}")
