"""Drives the outer test-failure retry loop around a single plan step
(Master Document 7.3, Epic C4): propose a diff, run the real test suite
against it, and -- if tests fail -- feed the actual failure output back to
the Editor for a genuinely revised diff, up to a configurable retry limit.

This is distinct from the lint-correction retry inside
reasoning.editor.propose_diff, which only fixes lint issues within a single
attempt and never sees test results. This loop sits one level up: each
attempt here is a full propose_diff() + run_tests_on_diff() cycle.

Also home to check_dangerous_ops() (Master Document 7.3/7.6, Epic E2): a
scan of the proposed diff text and the command(s) about to run in the
sandbox against `.solvix.yml`'s dangerous_ops patterns (force-push, hard
reset, branch deletion, destructive SQL, plus anything a repo adds). A
match halts the step before run_tests_on_diff ever executes. By default
this only returns a "needs confirmation" result without prompting; SLX-F1's
CLI supplies the optional confirm_dangerous_ops callback (threaded through
run_task -> execute_step_with_verification) to turn that into a real
interactive yes/no gate -- when the callback approves, the same diff
proceeds to run_tests_on_diff instead of the step stopping right there.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from config import SolvixConfig
from reasoning.editor import Diff, DiffGenerationError, PreviousAttempt, propose_diff
from reasoning.llm_client import complete
from reasoning.planner import Plan, PlanStep
from reasoning.task_input import TaskContext

from execution.test_runner import TestResult, run_tests_on_diff

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_MAX_TASK_ATTEMPTS = 10


@dataclass(frozen=True)
class DangerousOpsCheck:
    """Mirrors the shape of reasoning.planner.Plan's requires_approval /
    approval_reasons pair, for consistency between the two "don't
    auto-proceed, a human has to look at this" gates in the pipeline.
    """

    requires_confirmation: bool
    reasons: tuple[str, ...]


def check_dangerous_ops(
    diff_text: str,
    commands: Iterable[str],
    config: SolvixConfig | None = None,
) -> DangerousOpsCheck:
    """Scan diff_text and commands for any of config.dangerous_ops's regex
    patterns (case-insensitive), flagging a match as requiring explicit
    human confirmation rather than letting it proceed automatically.

    config defaults to SolvixConfig() (built-in patterns only) when not
    given, so this is always safe to call even before a repo's
    `.solvix.yml` has been loaded.
    """
    patterns = config.dangerous_ops if config is not None else SolvixConfig().dangerous_ops
    commands = tuple(commands)

    reasons: list[str] = []
    for pattern in patterns:
        regex = re.compile(pattern, re.IGNORECASE)
        if regex.search(diff_text):
            reasons.append(f"diff matches dangerous-ops pattern: {pattern}")
        for command in commands:
            if regex.search(command):
                reasons.append(f"command {command!r} matches dangerous-ops pattern: {pattern}")

    return DangerousOpsCheck(requires_confirmation=bool(reasons), reasons=tuple(reasons))


@dataclass(frozen=True)
class StepResult:
    success: bool
    diff: Diff | None
    test_result: TestResult | None
    attempts: int
    needs_human_help: bool
    blocked: bool = False
    block_reason: str | None = None
    requires_confirmation: bool = False
    confirmation_reasons: tuple[str, ...] = ()
    failure_reason: str | None = None


def execute_step_with_verification(
    step: PlanStep,
    file_content: str,
    context: TaskContext,
    complete_fn: Callable[[str, list[dict]], str] = complete,
    repo_root: str | Path | None = None,
    test_command: str = "pytest -q",
    max_retries: int | None = None,
    config: SolvixConfig | None = None,
    confirm_dangerous_ops: Callable[["DangerousOpsCheck"], bool] | None = None,
) -> StepResult:
    """Propose a diff for step and verify it against the real test suite,
    self-correcting on test failure.

    If config is given and step.file matches a `.solvix.yml` paths.deny
    pattern, this is a hard block (Master Document Epic F3): the function
    returns immediately with blocked=True and never calls propose_diff (so
    no LLM call is made at all for a denied file) or run_tests_on_diff. This
    is stronger than reasoning.planner's sensitive-path approval flag, which
    still lets the plan and diff be generated for human review.

    max_retries and test_command come from config when config is given
    (config.max_retries / config.test_command from `.solvix.yml`
    retries.max_attempts / test_command); otherwise the argument defaults
    apply, matching the same hardcoded-but-parameterized pattern used
    elsewhere before config loading existed. An explicit max_retries
    argument always wins over config.max_retries -- this is how
    execution.orchestrator.run_task (Epic E3) clamps a single step's retry
    loop to whatever's left of the task-level budget, without that override
    getting silently discarded in favor of config's own per-step number.

    If tests still fail after max_retries attempts, returns a failure result
    with needs_human_help=True rather than looping forever.

    propose_diff itself already gives a single attempt its own internal
    correction retry for a diff that fails to parse or apply (reasoning.
    editor's own docstring); if that's exhausted too, propose_diff raises
    DiffGenerationError (Epic C1) instead of returning. That's an expected,
    anticipated failure mode -- not a bug -- so it's caught here and turned
    into the same shape of result as test-failure exhaustion above
    (needs_human_help=True, this time with failure_reason set), rather than
    propagating as a raw exception. Doing the translation here rather than
    leaving it to each caller means every consumer of this function (CLI,
    a future dashboard, a future GitHub App integration) gets this safety
    net automatically instead of having to reimplement it.

    Once a diff is proposed, it (and the test command about to run) is
    scanned via check_dangerous_ops before run_tests_on_diff executes
    anything (Master Document Epic E2). With no confirm_dangerous_ops
    callback (the default), a match returns immediately with
    requires_confirmation=True and never runs the sandboxed test command --
    unlike the paths.deny hard block above, this isn't a permanent refusal,
    it's a "a human must explicitly approve this" gate. When a caller (the
    CLI) passes confirm_dangerous_ops, a match instead calls it with the
    DangerousOpsCheck; True lets this same diff proceed to run_tests_on_diff
    as if nothing were flagged, False stops the step exactly as the no-
    callback case does (still returning requires_confirmation=True, so a
    caller can tell "declined" apart from "never asked").

    requires_confirmation is deliberately kept separate from
    needs_human_help: the latter (from Epic C4) means the agent tried
    repeatedly and genuinely failed -- something's wrong and a human needs
    to debug. requires_confirmation means the opposite: the agent worked
    correctly and is being appropriately cautious about a risky-but-routine
    operation, and a human just needs to approve it. Conflating the two
    would make it impossible for a future consumer (CLI, dashboard, PR
    builder) to tell "this is broken" apart from "this is fine, just needs
    a yes/no."
    """
    if config is not None and config.is_denied(step.file):
        return StepResult(
            success=False,
            diff=None,
            test_result=None,
            attempts=0,
            needs_human_help=True,
            blocked=True,
            block_reason=f"{step.file} matches a paths.deny pattern in .solvix.yml",
        )

    if max_retries is not None:
        effective_max_retries = max_retries
    elif config is not None:
        effective_max_retries = config.max_retries
    else:
        effective_max_retries = _DEFAULT_MAX_RETRIES
    effective_test_command = config.test_command if config is not None else test_command

    previous_attempt: PreviousAttempt | None = None

    for attempt in range(1, effective_max_retries + 1):
        try:
            diff = propose_diff(
                step,
                file_content,
                context,
                complete_fn=complete_fn,
                repo_root=repo_root,
                previous_attempt=previous_attempt,
            )
        except DiffGenerationError as error:
            return StepResult(
                success=False,
                diff=None,
                test_result=None,
                attempts=attempt,
                needs_human_help=True,
                failure_reason=f"diff generation failed after exhausting its own retries: {error}",
            )

        safety_check = check_dangerous_ops(diff.diff_text, [effective_test_command], config)
        if safety_check.requires_confirmation:
            approved = confirm_dangerous_ops is not None and confirm_dangerous_ops(safety_check)
            if not approved:
                return StepResult(
                    success=False,
                    diff=diff,
                    test_result=None,
                    attempts=attempt,
                    needs_human_help=False,
                    requires_confirmation=True,
                    confirmation_reasons=safety_check.reasons,
                )

        test_result = run_tests_on_diff(repo_root, diff, test_command=effective_test_command)

        if test_result.passed:
            return StepResult(
                success=True,
                diff=diff,
                test_result=test_result,
                attempts=attempt,
                needs_human_help=False,
            )

        previous_attempt = PreviousAttempt(
            diff_text=diff.diff_text, failure_output=test_result.output
        )

    return StepResult(
        success=False,
        diff=diff,
        test_result=test_result,
        attempts=effective_max_retries,
        needs_human_help=True,
    )


@dataclass(frozen=True)
class TaskResult:
    """Outcome of run_task (Master Document 7.3, Epic E3): a task-level cap
    on total retries spent across an entire plan, separate from and on top
    of execute_step_with_verification's own per-step cap (Epic C4).

    culprit_step names whichever step's attempts caused the task-level cap
    to be reached -- either the step that was about to run when the
    accumulated total from earlier steps had already hit the cap, or the
    step whose own attempts pushed the running total over it. step_results
    holds every StepResult produced before run_task stopped, so a caller can
    see exactly how far the task got rather than a bare failure.

    needs_human_help is False (not True) when a step stopped because its
    dangerous-ops confirmation was declined -- that's a deliberate user
    choice, not the "agent is stuck and a human needs to debug it" meaning
    the flag has everywhere else (see StepResult's docstring).
    """

    success: bool
    needs_human_help: bool
    step_results: tuple[StepResult, ...]
    total_attempts: int
    culprit_step: PlanStep | None = None
    reason: str | None = None


def run_task(
    plan: Plan,
    context: TaskContext,
    complete_fn: Callable[[str, list[dict]], str] = complete,
    repo_root: str | Path | None = None,
    config: SolvixConfig | None = None,
    confirm_dangerous_ops: Callable[[DangerousOpsCheck], bool] | None = None,
) -> TaskResult:
    """Run every step of plan via execute_step_with_verification in order,
    tracking total attempts spent across ALL steps combined.

    confirm_dangerous_ops, when given, is passed straight through to every
    execute_step_with_verification call so a step flagged by
    check_dangerous_ops can be interactively approved (Epic E2) instead of
    unconditionally stopping the task -- see execute_step_with_verification's
    docstring for exactly how a True/False response changes its behavior.

    This is a task-level cap (config.max_task_attempts / `.solvix.yml`
    retries.max_task_attempts), distinct from and independent of
    config.max_retries, which stays scoped to bounding each individual
    step's own propose/test/retry loop inside
    execute_step_with_verification. A task can hit its cap either because
    one step alone burned through many retries, or because several steps
    each used a modest number that added up -- either way, run_task stops
    immediately once the running total reaches the cap and does not start
    any further steps, returning needs_human_help=True.

    file content for each step is read fresh from repo_root/step.file
    immediately before that step runs, since a prior step in the same plan
    may have modified a file a later step also touches.

    Critically, the task-level budget doesn't just get checked *between*
    steps -- it also clamps the per-step retry loop *while it runs*. Each
    step is given max_retries = min(its own configured per-step cap,
    whatever's left of the task budget), passed as an explicit override to
    execute_step_with_verification (which always prefers an explicit
    max_retries over config.max_retries -- see its docstring). Without
    this, a single long-retrying step would burn through its own full
    per-step allowance -- real LLM calls and real sandboxed test runs --
    before run_task ever got a chance to notice the task cap had been
    exceeded, making the cap purely cosmetic for the common 1-step-plan
    case. Clamping the limit handed to the step itself is what makes the
    cap actually prevent the unbounded time/cost it exists to prevent,
    not just report on it after the fact.

    Works the same for a 1-step plan (the common case today, via B3) and a
    future multi-step plan without any change: the loop and the cap
    accounting are step-count agnostic.
    """
    effective_max_task_attempts = (
        config.max_task_attempts if config is not None else _DEFAULT_MAX_TASK_ATTEMPTS
    )
    per_step_cap = config.max_retries if config is not None else _DEFAULT_MAX_RETRIES

    step_results: list[StepResult] = []
    total_attempts = 0

    for step in plan.steps:
        remaining_budget = effective_max_task_attempts - total_attempts
        if remaining_budget <= 0:
            return TaskResult(
                success=False,
                needs_human_help=True,
                step_results=tuple(step_results),
                total_attempts=total_attempts,
                culprit_step=step,
                reason=(
                    f"task-level retry cap ({effective_max_task_attempts}) was already "
                    f"reached (total attempts so far: {total_attempts}) before step "
                    f"{step.file!r} could start; remaining steps were not attempted"
                ),
            )

        file_path = Path(repo_root) / step.file if repo_root is not None else Path(step.file)
        file_content = file_path.read_text() if file_path.exists() else ""

        clamped_by_task_budget = remaining_budget < per_step_cap
        effective_step_cap = min(per_step_cap, remaining_budget)

        step_result = execute_step_with_verification(
            step,
            file_content,
            context,
            complete_fn=complete_fn,
            repo_root=repo_root,
            config=config,
            max_retries=effective_step_cap,
            confirm_dangerous_ops=confirm_dangerous_ops,
        )
        step_results.append(step_result)
        total_attempts += step_result.attempts

        if not step_result.success:
            if step_result.requires_confirmation:
                reason = (
                    f"step {step.file!r} was not applied because dangerous-ops "
                    f"confirmation was declined: {'; '.join(step_result.confirmation_reasons)}"
                )
                return TaskResult(
                    success=False,
                    needs_human_help=False,
                    step_results=tuple(step_results),
                    total_attempts=total_attempts,
                    culprit_step=step,
                    reason=reason,
                )
            if step_result.failure_reason is not None:
                return TaskResult(
                    success=False,
                    needs_human_help=True,
                    step_results=tuple(step_results),
                    total_attempts=total_attempts,
                    culprit_step=step,
                    reason=step_result.failure_reason,
                )
            if clamped_by_task_budget:
                reason = (
                    f"task-level retry cap ({effective_max_task_attempts}) cut step "
                    f"{step.file!r}'s retries short at {effective_step_cap} attempt(s) "
                    f"(its own per-step cap is {per_step_cap}) instead of letting it run "
                    f"to its own full allowance; task total is now {total_attempts}"
                )
            elif total_attempts >= effective_max_task_attempts:
                reason = (
                    f"task-level retry cap ({effective_max_task_attempts}) reached: step "
                    f"{step.file!r} alone used {step_result.attempts} attempt(s), bringing "
                    f"the task's total attempts to {total_attempts}"
                )
            else:
                reason = (
                    f"step {step.file!r} exhausted its own per-step retry budget "
                    f"({step_result.attempts} attempt(s)) and needs human help; the "
                    f"task-level cap ({effective_max_task_attempts}) was not reached "
                    f"(total attempts so far: {total_attempts})"
                )
            return TaskResult(
                success=False,
                needs_human_help=True,
                step_results=tuple(step_results),
                total_attempts=total_attempts,
                culprit_step=step,
                reason=reason,
            )

    return TaskResult(
        success=True,
        needs_human_help=False,
        step_results=tuple(step_results),
        total_attempts=total_attempts,
        culprit_step=None,
        reason=None,
    )
