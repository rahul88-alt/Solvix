"""CLI entry point: `solvix run "<task description>"` (Master Document 7.2,
Epic F1) and `solvix revise <pr_number>` (Master Document Epic D3). Wires the
full existing pipeline together in order -- config, sandbox preflight,
indexing, retrieval, clarification, planning, execution -- and is the first
place any of the pipeline's "don't auto-proceed" gates (reasoning.planner's
check_ambiguity and plan-approval flag, and execution.orchestrator's
dangerous-ops check) gets a real interactive prompt in front of it.

`revise` shares that entire pipeline via _execute_pipeline (the only
difference from `run` is what feeds in as the task -- a PR's fetched human
comment instead of a fresh CLI argument -- and what happens after: an
additional commit + push + PR comment on the PR's existing branch, instead
of a brand-new branch + PR). It never weakens or bypasses any of `run`'s
guardrails just because it's "only a revision."

check_ambiguity (Master Document Epic B2) runs once, right after the task's
relevant files are found and before generate_plan: if it returns a
question, this module prints it, reads one free-text answer via
click.prompt(), and appends both to the task text used for planning (not to
branch/commit/PR naming, which still use the original raw task string --
see reasoning.planner.check_ambiguity's docstring for why this is a single
round, not a loop). Passing --no-clarify skips the check entirely, so
scripted/automated invocations always proceed as given.

Uses click rather than typer: this command takes one plain string argument
and needs an interactive confirm prompt plus a CliRunner-testable surface --
exactly what click.command/click.confirm/click.testing.CliRunner give
directly, without typer's extra dependency weight (rich, shellingham) or
its type-hint-driven ergonomics, which have nothing to add for a single
one-argument command like this one.
"""

from __future__ import annotations

from pathlib import Path

import click

from config import load_config
from context.assembler import RetrievalResult
from execution.orchestrator import DangerousOpsCheck, StepResult, TaskResult, run_task
from execution.patch_applier import (
    PatchApplyError,
    apply_to_new_branch,
    checkout_branch,
    checkout_existing_branch,
    commit_to_current_branch,
    get_current_branch,
    slugify_task,
)
from execution.sandbox import DockerUnavailableError, ensure_docker_available, reap_orphans
from indexer.embedder import get_default_embedder
from indexer.pipeline import index_repo
from memory.task_state import (
    OUTCOME_FAILED,
    OUTCOME_NEEDS_HUMAN_HELP,
    OUTCOME_SUCCESS,
    TaskStateStore,
)
from reasoning.planner import Clarification, Plan, check_ambiguity, generate_plan
from reasoning.task_input import InvalidTaskInputError, TaskContext, build_task_context
from review.pr_builder import PRBuildError, PullRequestResult, build_pr, push_branch
from review.pr_feedback import PRFeedback, PRFeedbackError, build_revision_comment, fetch_pr_feedback, post_pr_comment


def _echo_progress(message: str) -> None:
    lowered = message.lower()
    if "failed" in lowered or "declined" in lowered or "blocked" in lowered:
        click.secho(message, fg="red")
    elif "passed" in lowered or "clean" in lowered:
        click.secho(message, fg="green")
    elif "warning" in lowered or "flagged" in lowered:
        click.secho(message, fg="yellow")
    else:
        click.secho(message, fg="cyan")


def _print_plan(plan: Plan) -> None:
    click.echo("Plan:")
    for i, step in enumerate(plan.steps, start=1):
        click.echo(f"  {i}. {step.file}: {step.description}")


def _confirm_plan_approval(plan: Plan) -> bool:
    click.echo()
    click.echo("This plan requires approval before proceeding:")
    _print_plan(plan)
    click.echo()
    click.echo("Reasons:")
    for reason in plan.approval_reasons:
        click.echo(f"  - {reason}")
    click.echo()
    return click.confirm("Proceed with this plan?", default=False)


def _confirm_dangerous_ops(safety_check: DangerousOpsCheck) -> bool:
    click.echo()
    click.echo("A proposed change was flagged as a dangerous operation:")
    for reason in safety_check.reasons:
        click.echo(f"  - {reason}")
    click.echo()
    return click.confirm("Apply and test this change anyway?", default=False)


def _print_step_result(index: int, step_result: StepResult) -> None:
    click.echo(f"Step {index}:")
    if step_result.blocked:
        click.echo(f"  blocked: {step_result.block_reason}")
        return
    if step_result.diff is not None:
        click.echo(f"  file: {step_result.diff.target_file}")
        click.echo("  diff:")
        for line in step_result.diff.diff_text.splitlines():
            click.echo(f"    {line}")
    if step_result.test_result is not None:
        click.echo(f"  tests passed: {step_result.test_result.passed}")
        click.echo("  test output:")
        for line in step_result.test_result.output.splitlines():
            click.echo(f"    {line}")
    if step_result.requires_confirmation:
        click.echo("  stopped: dangerous-ops confirmation was declined")
        for reason in step_result.confirmation_reasons:
            click.echo(f"    - {reason}")
    if step_result.failure_reason is not None:
        click.echo(f"  failed: {step_result.failure_reason}")
    if step_result.assertion_gaming_suspected:
        click.secho(
            f"  ⚠ suspected assertion-gaming: {step_result.assertion_gaming_details}",
            fg="yellow",
        )
    if step_result.weak_test_coverage_suspected:
        click.secho(
            f"  ⚠ added test may not meaningfully cover the changed behavior: "
            f"{step_result.weak_test_coverage_details}",
            fg="yellow",
        )
    click.echo(f"  attempts: {step_result.attempts}")


def _print_task_result(result: TaskResult) -> None:
    click.echo()
    if result.success:
        click.echo("Task completed successfully.")
    else:
        click.echo("Task did not complete successfully.")

    click.echo()
    for i, step_result in enumerate(result.step_results, start=1):
        _print_step_result(i, step_result)
        click.echo()

    if result.needs_human_help:
        click.echo("This task needs human help.")
        click.echo(f"Reason: {result.reason}")
        if result.culprit_step is not None:
            click.echo(f"Culprit step: {result.culprit_step.file}")
    elif not result.success and result.reason is not None:
        click.echo(f"Reason: {result.reason}")


@click.group()
def cli() -> None:
    """Solvix: an autonomous coding agent for your local repo."""


@cli.command()
@click.argument("task")
@click.option(
    "--repo",
    "repo_path",
    default=".",
    show_default=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to the repo to run the task against.",
)
@click.option(
    "--no-clarify",
    is_flag=True,
    default=False,
    help=(
        "Skip the ambiguity check and proceed with the task as given, "
        "without prompting for clarification. Useful for scripting/automation."
    ),
)
def run(task: str, repo_path: str, no_clarify: bool) -> None:
    """Run TASK against the repo at --repo (defaults to the current directory)."""
    repo_root = Path(repo_path).resolve()
    config = load_config(repo_root)

    try:
        ensure_docker_available()
    except DockerUnavailableError as error:
        raise click.ClickException(str(error))
    reap_orphans()

    store = TaskStateStore(repo_root)
    task_id = store.start_task(task, is_revision=False)

    outcome = OUTCOME_FAILED
    total_attempts = 0
    pr_url: str | None = None
    had_clarification = False
    try:
        task_context, plan, result, clarification = _execute_pipeline(task, repo_root, config, no_clarify)
        had_clarification = clarification is not None
        total_attempts = result.total_attempts
        outcome = _outcome_for(result)

        if not result.success:
            raise SystemExit(1)

        pr_result = _deliver_as_pull_request(repo_root, task, task_context, plan, result, clarification)
        pr_url = pr_result.url
    finally:
        # Recorded in every path -- success, a declined guardrail
        # (SystemExit out of _execute_pipeline before a result even exists,
        # leaving outcome/total_attempts at their failed/0 defaults), or an
        # unhandled exception -- so a crashed or interrupted run still shows
        # up as a real (if incomplete) history row instead of vanishing.
        store.finish_task(
            task_id,
            outcome=outcome,
            total_attempts=total_attempts,
            pr_url=pr_url,
            had_clarification=had_clarification,
        )
        store.close()


def _outcome_for(result: TaskResult) -> str:
    """Map a TaskResult onto memory.task_state's outcome vocabulary (Epic
    D4). needs_human_help is only True when the agent genuinely exhausted
    its retry budget (see StepResult's docstring) -- a declined plan
    approval or dangerous-ops confirmation is success=False,
    needs_human_help=False, a deliberate stop rather than the agent being
    stuck, so it's OUTCOME_FAILED rather than inflating the
    retry-exhaustion rate.
    """
    if result.success:
        return OUTCOME_SUCCESS
    if result.needs_human_help:
        return OUTCOME_NEEDS_HUMAN_HELP
    return OUTCOME_FAILED


def _execute_pipeline(
    task: str, repo_root: Path, config, no_clarify: bool
) -> tuple[TaskContext, Plan, TaskResult, Clarification | None]:
    """Index repo_root, retrieve context for task, optionally clarify, plan,
    and execute -- the full pipeline shared by `run` (a brand-new task) and
    `revise` (Epic D3: a PR's fetched feedback comment treated as the task).
    Always operates against whatever is currently checked out in repo_root,
    so callers control which branch's state context retrieval/execution see
    by checking out the right branch before calling this.

    Raises SystemExit(1) (via click) if a plan-approval or dangerous-ops
    gate is declined, or if execution raises an unanticipated exception --
    matching cli.run's original behavior before this was extracted.
    """
    click.secho(f"Indexing {repo_root}...", fg="cyan")
    index_result = index_repo(repo_root)
    click.secho(
        f"Indexed {index_result.num_files_indexed} file(s), {index_result.num_chunks} chunk(s).",
        fg="green",
    )

    embedder = get_default_embedder()
    click.secho(f"Finding relevant files for: {task}...", fg="cyan")
    try:
        task_context = build_task_context(task, index_result, embedder)
    except InvalidTaskInputError as error:
        raise click.ClickException(str(error))

    retrieval: RetrievalResult = task_context.retrieval
    click.echo("Relevant files:")
    for f in retrieval.files:
        click.echo(f"  - {f.file_path} (score={f.score:.2f})")

    clarification: Clarification | None = None
    if not no_clarify:
        question = check_ambiguity(task_context)
        if question is not None:
            click.echo()
            click.secho("This task looks ambiguous:", fg="yellow")
            click.echo(f"  {question}")
            answer = click.prompt("Your answer")
            clarification = Clarification(question=question, answer=answer)
            clarified_task = f"{task_context.task}\n\nClarifying question: {question}\nAnswer: {answer}"
            task_context = TaskContext(task=clarified_task, retrieval=task_context.retrieval)

    click.secho("Generating plan...", fg="cyan")
    plan = generate_plan(task_context, config=config)
    _print_plan(plan)

    if plan.requires_approval:
        if not _confirm_plan_approval(plan):
            click.echo("Aborted: plan was not approved.")
            raise SystemExit(1)

    click.echo()
    click.secho("Executing plan...", fg="cyan")
    try:
        result = run_task(
            plan,
            task_context,
            repo_root=repo_root,
            config=config,
            confirm_dangerous_ops=_confirm_dangerous_ops,
            on_progress=_echo_progress,
        )
    except (click.exceptions.Abort, click.ClickException):
        # EOF on a confirm() prompt (e.g. Ctrl-D) and click's own usage
        # errors already have clean, click-native handling ("Aborted!" /
        # a formatted usage error) -- let those propagate as-is rather
        # than relabeling them as a pipeline failure below.
        raise
    except Exception as error:  # noqa: BLE001 -- last-resort CLI boundary
        # execute_step_with_verification/run_task already translate every
        # failure mode they anticipate (test failure, retry exhaustion,
        # DiffGenerationError, dangerous ops) into a StepResult/TaskResult
        # rather than raising. A genuinely unanticipated exception this
        # deep should still never surface as a raw traceback to a CLI
        # user, so this is a final safety net, not the primary handling
        # path for any expected failure.
        click.echo()
        click.echo("Task did not complete successfully.")
        click.echo("This task needs human help.")
        click.echo(f"Reason: unhandled error while executing the plan: {error}")
        raise SystemExit(1)

    _print_task_result(result)
    return task_context, plan, result, clarification


@cli.command()
@click.argument("pr_number", type=int)
@click.option(
    "--repo",
    "repo_path",
    default=".",
    show_default=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to the repo the PR belongs to.",
)
@click.option(
    "--no-clarify",
    is_flag=True,
    default=False,
    help=(
        "Skip the ambiguity check and proceed with the feedback as given, "
        "without prompting for clarification. Useful for scripting/automation."
    ),
)
def revise(pr_number: int, repo_path: str, no_clarify: bool) -> None:
    """Revise an existing PR (PR_NUMBER) using its most recent human review
    comment as the new task description (Master Document Epic D3): checks
    out the PR's own branch, re-runs the full pipeline against it, and
    either pushes an additional commit + comments the outcome, or comments
    what was tried and why it didn't resolve. Never leaves the repo on the
    PR's branch afterward -- the branch checked out when this command
    started is always restored, success or failure.
    """
    repo_root = Path(repo_path).resolve()
    config = load_config(repo_root)

    try:
        ensure_docker_available()
    except DockerUnavailableError as error:
        raise click.ClickException(str(error))
    reap_orphans()

    click.secho(f"Fetching feedback for PR #{pr_number}...", fg="cyan")
    try:
        feedback = fetch_pr_feedback(repo_root, pr_number)
    except PRFeedbackError as error:
        raise click.ClickException(str(error))

    click.echo(f"Branch: {feedback.branch}")
    click.echo("Latest feedback comment:")
    click.echo(f"  {feedback.comment_body}")

    # feedback.comment_body is only known after the fetch above succeeds, so
    # that's the earliest point a history row is meaningful -- a fetch
    # failure (bad PR number, no human comment yet) has no task text to log
    # and never reaches the pipeline, so nothing is recorded for it.
    store = TaskStateStore(repo_root)
    task_id = store.start_task(feedback.comment_body, is_revision=True, pr_url=feedback.url)

    original_branch = get_current_branch(repo_root)
    try:
        checkout_existing_branch(repo_root, feedback.branch)
    except PatchApplyError as error:
        store.finish_task(task_id, outcome=OUTCOME_FAILED, total_attempts=0)
        store.close()
        raise click.ClickException(f"failed to check out PR branch {feedback.branch!r}: {error}")

    try:
        _revise_pr(repo_root, pr_number, feedback, config, no_clarify, store, task_id)
    finally:
        # Restore unconditionally: success, a declined guardrail (SystemExit),
        # or any other failure must never leave the repo checked out on the
        # PR's branch instead of what the user had checked out originally.
        checkout_branch(repo_root, original_branch, check=False)
        store.close()


def _revise_pr(
    repo_root: Path,
    pr_number: int,
    feedback: PRFeedback,
    config,
    no_clarify: bool,
    store: TaskStateStore,
    task_id: int,
) -> None:
    outcome = OUTCOME_FAILED
    total_attempts = 0
    had_clarification = False
    try:
        task_context, plan, result, clarification = _execute_pipeline(
            feedback.comment_body, repo_root, config, no_clarify
        )
        had_clarification = clarification is not None
        total_attempts = result.total_attempts
        outcome = _outcome_for(result)

        if result.success:
            commit_message = f"solvix: revise PR #{pr_number}: {feedback.comment_body.splitlines()[0]}"
            try:
                commit_to_current_branch(repo_root, _combined_diff_text(result), commit_message)
                push_branch(repo_root, feedback.branch)
            except (PatchApplyError, PRBuildError) as error:
                raise click.ClickException(f"failed to push revision to PR #{pr_number}: {error}")

        comment_body = build_revision_comment(plan, result, feedback.comment_body)
        try:
            post_pr_comment(repo_root, pr_number, comment_body)
        except PRFeedbackError as error:
            raise click.ClickException(f"revision {'succeeded' if result.success else 'failed'} but posting the PR comment failed: {error}")

        if result.success:
            click.secho(f"Pushed revision to PR #{pr_number} ({feedback.branch}).", fg="green")
        else:
            click.secho(f"Revision did not complete; posted details to PR #{pr_number}.", fg="yellow")
            raise SystemExit(1)
    finally:
        # pr_url is omitted here (not re-passed) -- feedback.url was already
        # recorded at start_task() and finish_task()'s COALESCE keeps it,
        # since a revision never changes which PR it belongs to.
        store.finish_task(
            task_id,
            outcome=outcome,
            total_attempts=total_attempts,
            had_clarification=had_clarification,
        )


def _combined_diff_text(result: TaskResult) -> str:
    return "\n".join(
        step_result.diff.diff_text for step_result in result.step_results if step_result.diff
    )


def _deliver_as_pull_request(
    repo_root: Path,
    task: str,
    task_context: TaskContext,
    plan: Plan,
    result: TaskResult,
    clarification: Clarification | None = None,
) -> PullRequestResult:
    """Turn a successful TaskResult into a real branch + commit + PR
    (Master Document 7.2, Epic D1). Only ever called after result.success is
    True -- a failed or needs-human-help task never reaches here, so nothing
    is committed or pushed for it.

    clarification, when given, is the check_ambiguity question/answer round
    that shaped task_context.task (Epic B2); it's passed straight through to
    build_pr so it shows up in the PR body's "Key decisions" section instead
    of being visible only to whoever ran the command live.
    """
    click.echo()
    click.echo("Delivering change as a pull request...")

    branch_name = f"solvix/{slugify_task(task)}"
    commit_message = f"solvix: {task}"

    try:
        final_branch = apply_to_new_branch(repo_root, _combined_diff_text(result), branch_name, commit_message)
        pr_result = build_pr(repo_root, final_branch, task_context, plan, result, clarification)
    except (PatchApplyError, PRBuildError) as error:
        raise click.ClickException(f"failed to deliver change as a pull request: {error}")

    click.echo(f"Opened pull request: {pr_result.url}")
    return pr_result


_HISTORY_TASK_WIDTH = 50


def _truncate(text: str, width: int) -> str:
    flat = " ".join(text.split())
    if len(flat) <= width:
        return flat
    return flat[: width - 1].rstrip() + "…"


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{int(secs):02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes):02d}m"


def _format_timestamp(dt) -> str:
    return dt.strftime("%Y-%m-%d %H:%M UTC")


@cli.command()
@click.option(
    "--repo",
    "repo_path",
    default=".",
    show_default=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to the repo whose task history to show.",
)
@click.option(
    "--limit",
    default=20,
    show_default=True,
    type=int,
    help="Maximum number of most-recent tasks to show.",
)
def history(repo_path: str, limit: int) -> None:
    """Show recent solvix run/revise task outcomes (Master Document Epic D4)."""
    repo_root = Path(repo_path).resolve()
    store = TaskStateStore(repo_root)
    try:
        records = store.list_tasks(limit=limit)
    finally:
        store.close()

    if not records:
        click.echo("No task history yet -- run `solvix run` or `solvix revise` first.")
        return

    headers = ("Started", "Type", "Task", "Outcome", "Attempts", "Time", "PR")
    rows = [
        (
            _format_timestamp(r.started_at),
            "revise" if r.is_revision else "run",
            _truncate(r.task_text, _HISTORY_TASK_WIDTH),
            r.outcome,
            str(r.total_attempts),
            _format_duration(r.duration_seconds),
            r.pr_url or "-",
        )
        for r in records
    ]

    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    _print_row(headers, widths)
    _print_row(["-" * w for w in widths], widths)
    for row in rows:
        _print_row(row, widths)


def _print_row(cells, widths: list[int]) -> None:
    click.echo("  ".join(cell.ljust(width) for cell, width in zip(cells, widths)))


@cli.command()
@click.option(
    "--repo",
    "repo_path",
    default=".",
    show_default=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to the repo whose task stats to show.",
)
def stats(repo_path: str) -> None:
    """Show aggregate task-outcome stats (Master Document Section 5 /
    Epic D4): resolution rate, retry-exhaustion rate, and median
    time-to-resolution across every recorded run/revise task.
    """
    repo_root = Path(repo_path).resolve()
    store = TaskStateStore(repo_root)
    try:
        s = store.stats()
    finally:
        store.close()

    if s.total == 0:
        click.echo("No task history yet -- run `solvix run` or `solvix revise` first.")
        return

    def _pct(fraction: float | None) -> str:
        return f"{fraction * 100:.1f}%" if fraction is not None else "n/a"

    click.echo(f"Tasks recorded: {s.total} ({s.finished} finished, {s.in_progress} in progress)")
    click.echo()
    click.echo("Outcomes:")
    click.echo(f"  success:           {s.success}")
    click.echo(f"  needs_human_help:  {s.needs_human_help}")
    click.echo(f"  failed:            {s.failed}")
    click.echo()
    click.echo(f"Resolution rate (success / finished):        {_pct(s.resolution_rate)}")
    click.echo(f"Retry-exhaustion rate (needs_help / finished): {_pct(s.retry_exhaustion_rate)}")
    click.echo(
        "Median time-to-resolution (successful tasks): "
        f"{_format_duration(s.median_time_to_resolution_seconds)}"
    )


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
