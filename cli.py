"""CLI entry point: `solvix run "<task description>"` (Master Document 7.2,
Epic F1). Wires the full existing pipeline together in order -- config,
sandbox preflight, indexing, retrieval, planning, execution -- and is the
first place either of the pipeline's two "don't auto-proceed" gates
(reasoning.planner's plan-approval flag and execution.orchestrator's
dangerous-ops check) gets a real interactive yes/no prompt in front of it.
Before this story both gates only produced data; nothing consumed it.

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
from execution.patch_applier import PatchApplyError, apply_to_new_branch, slugify_task
from execution.sandbox import DockerUnavailableError, ensure_docker_available, reap_orphans
from indexer.embedder import get_default_embedder
from indexer.pipeline import index_repo
from reasoning.planner import Plan, generate_plan
from reasoning.task_input import InvalidTaskInputError, TaskContext, build_task_context
from review.pr_builder import PRBuildError, build_pr


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
def run(task: str, repo_path: str) -> None:
    """Run TASK against the repo at --repo (defaults to the current directory)."""
    repo_root = Path(repo_path).resolve()
    config = load_config(repo_root)

    try:
        ensure_docker_available()
    except DockerUnavailableError as error:
        raise click.ClickException(str(error))
    reap_orphans()

    click.echo(f"Indexing {repo_root}...")
    index_result = index_repo(repo_root)
    click.echo(
        f"Indexed {index_result.num_files_indexed} file(s), {index_result.num_chunks} chunk(s)."
    )

    embedder = get_default_embedder()
    try:
        task_context = build_task_context(task, index_result, embedder)
    except InvalidTaskInputError as error:
        raise click.ClickException(str(error))

    retrieval: RetrievalResult = task_context.retrieval
    click.echo("Relevant files:")
    for f in retrieval.files:
        click.echo(f"  - {f.file_path} (score={f.score:.2f})")

    click.echo("Generating plan...")
    plan = generate_plan(task_context, config=config)
    _print_plan(plan)

    if plan.requires_approval:
        if not _confirm_plan_approval(plan):
            click.echo("Aborted: plan was not approved.")
            raise SystemExit(1)

    click.echo()
    click.echo("Executing plan...")
    try:
        result = run_task(
            plan,
            task_context,
            repo_root=repo_root,
            config=config,
            confirm_dangerous_ops=_confirm_dangerous_ops,
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

    if not result.success:
        raise SystemExit(1)

    _deliver_as_pull_request(repo_root, task, task_context, plan, result)


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
) -> None:
    """Turn a successful TaskResult into a real branch + commit + PR
    (Master Document 7.2, Epic D1). Only ever called after result.success is
    True -- a failed or needs-human-help task never reaches here, so nothing
    is committed or pushed for it.
    """
    click.echo()
    click.echo("Delivering change as a pull request...")

    branch_name = f"solvix/{slugify_task(task)}"
    commit_message = f"solvix: {task}"

    try:
        final_branch = apply_to_new_branch(repo_root, _combined_diff_text(result), branch_name, commit_message)
        pr_result = build_pr(repo_root, final_branch, task_context, plan, result.step_results)
    except (PatchApplyError, PRBuildError) as error:
        raise click.ClickException(f"failed to deliver change as a pull request: {error}")

    click.echo(f"Opened pull request: {pr_result.url}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
