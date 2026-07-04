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

Also home to detect_assertion_gaming() (Master Document Epic C5): pure-
pattern-matching detection of a retry that makes tests pass by rewriting a
test assertion's expected literal to match the previous attempt's actual
(failing) output, instead of genuinely fixing the implementation -- the
"add(0.1, 0.2) == 0.3 -> == 0.30000000000000004" failure mode found during
SLX-D2 smoke testing against a real local model. This is a detection-only,
surface-it-in-the-PR guardrail (see review.pr_builder's "Needs attention"
section), not a blocking one -- see its own docstring for exactly what
does and doesn't count as a match.

Also home to check_test_coverage_sanity() (Master Document Epic C2): a
second, adjacent detection-only guardrail for the paired test steps
reasoning.planner now generates -- writing its *own* test hands the model
more latitude than satisfying an existing one, and a model could satisfy
"a test was added" with busy-work (a test that never references the
changed function, or has no real assertion) rather than a test that
actually exercises the change. Same philosophy as detect_assertion_gaming:
cheap, deterministic, pattern-only, flags via StepResult for the PR body
rather than blocking or forcing a retry.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

from config import SolvixConfig
from reasoning.editor import Diff, DiffGenerationError, PreviousAttempt, propose_diff
from reasoning.llm_client import complete
from reasoning.planner import Plan, PlanStep, is_test_file
from reasoning.task_input import TaskContext

from execution.test_runner import TestResult, apply_diff, copy_repo, run_tests_on_diff

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_MAX_TASK_ATTEMPTS = 10


def _emit(on_progress: Callable[[str], None] | None, message: str) -> None:
    """Call on_progress(message) if a callback was given (Master Document
    Epic F2: live progress updates), else do nothing. Centralizing the
    None-check here means every call site below can fire progress events
    unconditionally instead of repeating `if on_progress is not None`.
    """
    if on_progress is not None:
        on_progress(message)


_ASSERT_E_LINE_RE = re.compile(r"^E\s+assert\s+(.+?)\s*==\s*(.+?)\s*$")
_ASSERT_WHERE_LINE_RE = re.compile(r"^E\s+\+\s*where\s+(.+?)\s*=\s*.+$")
_ASSERT_STMT_RE = re.compile(r"^assert\s+(.+?)\s*(==|!=)\s*(.+?)\s*$")
_LITERAL_TOKEN_RE = re.compile(
    r"^-?\d+(\.\d+)?$|^(['\"]).*\2$|^(True|False|None)$"
)


def _is_literal_token(token: str) -> bool:
    """True if token looks like a plain literal (number, quoted string,
    True/False/None) rather than a variable name or function call --
    used to make sure assertion-gaming detection only ever matches a
    genuine "expected value" being swapped, not some other expression.
    """
    return bool(_LITERAL_TOKEN_RE.match(token.strip()))


def _literal_values_equal(a: str, b: str) -> bool:
    """Compare two literal tokens for equality, numerically when both
    parse as numbers (so "0.3" and "0.30" -- an unlikely but harmless
    formatting difference -- still count as the same value) and by exact
    text otherwise.
    """
    try:
        return float(a) == float(b)
    except ValueError:
        return a.strip() == b.strip()


def _parse_actual_expected_from_failure(output: str) -> tuple[str, str] | None:
    """Parse pytest's assertion-failure output for the actual (computed)
    value and the expected (literal) value of a failed `assert x == y`.

    Looks for pytest's own "E       assert <lhs> == <rhs>" detail line,
    plus the optional following "E        +  where <value> = <expr>" line
    it emits when one side of the comparison was a function call -- that
    "where" line is what actually tells us which of the two operands is
    the *computed* value, since assert statements are just as often
    written `assert expected == actual()` as `assert actual() ==
    expected`. Falls back to treating the left-hand side as "actual" (the
    common `assert result == expected` convention) only when no "where"
    line disambiguates it, e.g. for a bare `assert 5 == 6`. Returns None
    if the output doesn't contain a recognizable two-literal equality
    failure at all.
    """
    lines = output.splitlines()
    for i, line in enumerate(lines):
        match = _ASSERT_E_LINE_RE.match(line.strip())
        if not match:
            continue
        lhs, rhs = match.group(1).strip(), match.group(2).strip()

        where_value = None
        for follow in lines[i + 1 : i + 4]:
            where_match = _ASSERT_WHERE_LINE_RE.match(follow.strip())
            if where_match:
                where_value = where_match.group(1).strip()
                break

        if where_value == rhs:
            return rhs, lhs  # actual, expected
        return lhs, rhs  # actual, expected (default convention, or where_value == lhs)
    return None


def _split_assert_operands(line: str) -> tuple[str, str, str] | None:
    """Split a single source line of the form `assert <lhs> <op> <rhs>`
    into its operands, or None if the line isn't a plain equality/
    inequality assertion (e.g. it has no comparison, or is some other
    statement entirely).
    """
    match = _ASSERT_STMT_RE.match(line.strip())
    if not match:
        return None
    return match.group(1).strip(), match.group(2), match.group(3).strip()


def _diff_assertion_literal_changes(diff_text: str) -> list[tuple[str, str]]:
    """Scan a unified diff for removed/added line pairs that are both
    `assert <lhs> <op> <rhs>` statements differing in exactly one
    literal operand, returning each (old_literal, new_literal) pair
    found.

    Only lines that are actually replaced (a `-` line immediately
    followed, in equal number, by a `+` line in the same hunk position)
    are considered -- a pure addition (e.g. a new test function with no
    corresponding removed line) can never match, which is what keeps
    "a test file was legitimately extended" from ever being flagged.
    The non-differing operand must also match exactly between old and
    new, so this only fires on a *literal* swap, not a rewritten
    expression that happens to also change the other side.
    """
    lines = diff_text.splitlines()
    changes: list[tuple[str, str]] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("-") and not lines[i].startswith("---"):
            removed = []
            j = i
            while j < len(lines) and lines[j].startswith("-") and not lines[j].startswith("---"):
                removed.append(lines[j][1:])
                j += 1
            added = []
            k = j
            while k < len(lines) and lines[k].startswith("+") and not lines[k].startswith("+++"):
                added.append(lines[k][1:])
                k += 1
            if len(removed) == len(added):
                for old_line, new_line in zip(removed, added):
                    old_ops = _split_assert_operands(old_line)
                    new_ops = _split_assert_operands(new_line)
                    if old_ops is None or new_ops is None:
                        continue
                    old_lhs, old_op, old_rhs = old_ops
                    new_lhs, new_op, new_rhs = new_ops
                    if old_op != new_op:
                        continue
                    if (
                        old_lhs == new_lhs
                        and old_rhs != new_rhs
                        and _is_literal_token(old_rhs)
                        and _is_literal_token(new_rhs)
                    ):
                        changes.append((old_rhs, new_rhs))
                    elif (
                        old_rhs == new_rhs
                        and old_lhs != new_lhs
                        and _is_literal_token(old_lhs)
                        and _is_literal_token(new_lhs)
                    ):
                        changes.append((old_lhs, new_lhs))
            i = k
        else:
            i += 1
    return changes


def detect_assertion_gaming(previous_failure_output: str, diff_text: str) -> str | None:
    """Detect the SLX-C5 assertion-gaming pattern: a retry diff that
    changes an assertion's expected literal to the previous attempt's
    actual (failing) output, rather than changing the implementation to
    genuinely satisfy the original expectation.

    Deliberately only flags the exact-match case (old literal ==
    previous attempt's expected value, new literal == previous attempt's
    actual value) -- see execute_step_with_verification's docstring for
    why a literal change to some other, non-matching value is not
    flagged here.
    """
    parsed = _parse_actual_expected_from_failure(previous_failure_output)
    if parsed is None:
        return None
    actual, expected = parsed

    for old_literal, new_literal in _diff_assertion_literal_changes(diff_text):
        if _literal_values_equal(old_literal, expected) and _literal_values_equal(new_literal, actual):
            return (
                f"assertion's expected value was changed from {old_literal!r} to {new_literal!r} "
                f"-- {new_literal!r} matches the previous attempt's actual (failing) output, "
                "suggesting the test was rewritten to match the code rather than the code being "
                "fixed to satisfy the original expectation"
            )
    return None


_DEF_OR_CLASS_LINE_RE = re.compile(r"^(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")
_BARE_ASSERT_LINE_RE = re.compile(r"^assert\b(.*)$")
_TRIVIAL_ASSERT_BODIES = frozenset({"", "true", "1", "true is true", "true, ", "true,"})
# unittest.TestCase-style assertions (self.assertEqual(...), assertTrue(...),
# etc.) are just as real a check as a bare `assert` statement -- sample_repo
# and its planner-generated tests both stick to pytest's bare assert, but a
# local model is free to write unittest style instead (confirmed during this
# story's own real-Ollama smoke test), and this heuristic shouldn't punish
# that with a false "no real assertion" flag.
_UNITTEST_ASSERT_CALL_RE = re.compile(r"(?:^|\W)(?:self\.)?assert[A-Za-z_]+\s*\(")


def _added_lines(diff_text: str) -> list[str]:
    """Every added-line body (the '+' stripped off, diff-header '+++'
    lines excluded) from a unified diff -- the only lines that represent
    genuinely new content, as opposed to unchanged context or removed
    lines.
    """
    return [
        line[1:]
        for line in diff_text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]


def _changed_symbol_names(diff_text: str) -> set[str]:
    """Every function/method/class name introduced or modified by
    diff_text, read off its added `def`/`async def`/`class` lines. Used as
    the "changed behavior" a paired test step's diff is expected to
    reference by name.
    """
    names = set()
    for line in _added_lines(diff_text):
        match = _DEF_OR_CLASS_LINE_RE.match(line.strip())
        if match:
            names.add(match.group(1))
    return names


def _test_diff_references_symbol(diff_text: str, symbol_names: set[str]) -> bool:
    added_text = "\n".join(_added_lines(diff_text))
    return any(re.search(rf"\b{re.escape(name)}\b", added_text) for name in symbol_names)


def _test_diff_has_real_assertion(diff_text: str) -> bool:
    """True if diff_text adds at least one assert statement that isn't
    trivially always-true (`assert True`, `assert 1`, or a bare `assert`
    with nothing after it), whether written as a bare pytest-style `assert`
    or a unittest.TestCase-style `self.assertEqual(...)` call -- the
    cheapest possible signal that a "test" isn't just busy-work with no
    actual check in it.
    """
    for line in _added_lines(diff_text):
        stripped = line.strip()
        bare_match = _BARE_ASSERT_LINE_RE.match(stripped)
        if bare_match is not None:
            body = bare_match.group(1).strip().rstrip(",").strip().lower()
            if body not in _TRIVIAL_ASSERT_BODIES:
                return True
            continue
        if _UNITTEST_ASSERT_CALL_RE.search(stripped):
            return True
    return False


def check_test_coverage_sanity(impl_diff_text: str, test_diff_text: str) -> str | None:
    """Cheap, deterministic (non-LLM) sanity check on a paired test step's
    diff (Master Document Epic C2): does it actually reference the
    function/class name(s) changed by impl_diff_text, and does it contain
    at least one real (non-trivial) assertion?

    Returns a human-readable reason string if either check fails, else
    None. This is intentionally forgiving rather than a strict linter --
    if impl_diff_text introduces no recognizable def/class at all (e.g. it
    only edits an existing function body without a `def` line appearing
    in the diff's added lines, as a diff that only touches interior
    statements would), the symbol-reference check is skipped entirely
    rather than flagging every such step; the assertion check still runs
    regardless, since "does this test have a real assertion at all" needs
    no knowledge of what changed.
    """
    reasons: list[str] = []

    symbol_names = _changed_symbol_names(impl_diff_text)
    if symbol_names and not _test_diff_references_symbol(test_diff_text, symbol_names):
        reasons.append(
            "test diff does not reference any of the changed function/class name(s) "
            f"({', '.join(sorted(symbol_names))}) from the implementation diff"
        )

    if not _test_diff_has_real_assertion(test_diff_text):
        reasons.append(
            "test diff does not contain a real assertion (only a trivial/empty one, or none at all)"
        )

    if not reasons:
        return None
    return "; ".join(reasons)


def _summarize_failure_output(output: str) -> str:
    """Pick the single most useful line out of a pytest failure's output for
    a human-readable retry summary: pytest's own "E   ..." assertion-detail
    line when present (the "F....." dot progress line that's usually first
    carries no information about *why* it failed), falling back to the last
    non-empty line (typically the "N failed, M passed" summary) otherwise.
    """
    lines = [line for line in output.strip().splitlines() if line.strip()]
    if not lines:
        return "(no output)"
    for line in lines:
        if line.lstrip().startswith("E "):
            return line.strip()
    return lines[-1].strip()


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
    dangerous_ops_confirmed: bool = False
    dangerous_ops_confirmed_reasons: tuple[str, ...] = ()
    attempt_failures: tuple[str, ...] = ()
    assertion_gaming_suspected: bool = False
    assertion_gaming_details: str = ""
    weak_test_coverage_suspected: bool = False
    weak_test_coverage_details: str = ""


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
    step_index: int = 1,
    on_progress: Callable[[str], None] | None = None,
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
    anticipated failure mode -- not a bug -- so it's caught here rather than
    propagating as a raw exception (Epic C6). A DiffGenerationError consumes
    one attempt of *this* function's own outer per-step budget, exactly like
    a failed test does: the failure reason is fed back into the next
    attempt's prompt (as a PreviousAttempt with no diff_text of its own, just
    the parse/apply error as failure_output) and the loop continues, rather
    than terminating the step on the first occurrence. Only once the whole
    effective_max_retries budget is exhausted -- whether every attempt hit a
    DiffGenerationError, every attempt failed its tests, or some mix of the
    two -- does the step return needs_human_help=True, with failure_reason
    set only when the *final* attempt was itself a DiffGenerationError (so a
    caller can still tell "died on a malformed diff" apart from "died on a
    failing test" for the attempt that actually ended the loop).

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
        _emit(
            on_progress,
            f"Step {step_index}: blocked -- {step.file} matches a paths.deny pattern in .solvix.yml",
        )
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

    diff: Diff | None = None
    test_result: TestResult | None = None
    previous_attempt: PreviousAttempt | None = None
    confirmed_reasons: list[str] = []
    attempt_failures: list[str] = []
    assertion_gaming_flags: list[str] = []
    last_diff_generation_error: str | None = None

    for attempt in range(1, effective_max_retries + 1):
        _emit(
            on_progress,
            f"Step {step_index}: proposing changes to {step.file} "
            f"(attempt {attempt}/{effective_max_retries})...",
        )
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
            _emit(
                on_progress,
                f"Step {step_index}: diff generation failed (attempt {attempt}): {error}",
            )
            last_diff_generation_error = str(error)
            attempt_failures.append(f"attempt {attempt}: diff generation failed: {error}")
            diff = None
            test_result = None
            previous_attempt = PreviousAttempt(
                diff_text="(no diff was produced -- the previous response could not be "
                "parsed as a unified diff, or did not apply cleanly)",
                failure_output=str(error),
            )
            continue

        last_diff_generation_error = None

        if previous_attempt is not None:
            gaming_message = detect_assertion_gaming(previous_attempt.failure_output, diff.diff_text)
            if gaming_message is not None:
                assertion_gaming_flags.append(f"attempt {attempt}: {gaming_message}")
                _emit(
                    on_progress,
                    f"Step {step_index}: flagged -- suspected assertion-gaming (attempt {attempt}): "
                    f"{gaming_message}",
                )

        if diff.lint_warnings:
            _emit(
                on_progress,
                f"Step {step_index}: lint warnings (attempt {attempt}): "
                f"{'; '.join(diff.lint_warnings)}",
            )
        else:
            _emit(on_progress, f"Step {step_index}: lint clean (attempt {attempt})")

        safety_check = check_dangerous_ops(diff.diff_text, [effective_test_command], config)
        if safety_check.requires_confirmation:
            _emit(
                on_progress,
                f"Step {step_index}: dangerous operation flagged (attempt {attempt}): "
                f"{'; '.join(safety_check.reasons)}",
            )
            approved = confirm_dangerous_ops is not None and confirm_dangerous_ops(safety_check)
            if not approved:
                _emit(
                    on_progress,
                    f"Step {step_index}: dangerous-ops confirmation declined, stopping step",
                )
                return StepResult(
                    success=False,
                    diff=diff,
                    test_result=None,
                    attempts=attempt,
                    needs_human_help=False,
                    requires_confirmation=True,
                    confirmation_reasons=safety_check.reasons,
                    assertion_gaming_suspected=bool(assertion_gaming_flags),
                    assertion_gaming_details="; ".join(assertion_gaming_flags),
                )
            confirmed_reasons.extend(safety_check.reasons)

        _emit(on_progress, f"Step {step_index}: running tests (attempt {attempt})...")
        test_result = run_tests_on_diff(repo_root, diff, test_command=effective_test_command)

        if test_result.passed:
            _emit(on_progress, f"Step {step_index}: tests passed (attempt {attempt})")
            return StepResult(
                success=True,
                diff=diff,
                test_result=test_result,
                attempts=attempt,
                needs_human_help=False,
                dangerous_ops_confirmed=bool(confirmed_reasons),
                dangerous_ops_confirmed_reasons=tuple(confirmed_reasons),
                attempt_failures=tuple(attempt_failures),
                assertion_gaming_suspected=bool(assertion_gaming_flags),
                assertion_gaming_details="; ".join(assertion_gaming_flags),
            )

        _emit(on_progress, f"Step {step_index}: tests failed (attempt {attempt})")
        attempt_failures.append(f"attempt {attempt}: {_summarize_failure_output(test_result.output)}")

        previous_attempt = PreviousAttempt(
            diff_text=diff.diff_text, failure_output=test_result.output
        )

    _emit(
        on_progress,
        f"Step {step_index}: exhausted retries ({effective_max_retries} attempt(s)), needs human help",
    )
    exhausted_failure_reason = (
        f"diff generation failed on the final attempt after exhausting the outer "
        f"per-step retry budget ({effective_max_retries} attempt(s)): {last_diff_generation_error}"
        if last_diff_generation_error is not None
        else None
    )
    return StepResult(
        success=False,
        diff=diff,
        test_result=test_result,
        attempts=effective_max_retries,
        needs_human_help=True,
        failure_reason=exhausted_failure_reason,
        dangerous_ops_confirmed=bool(confirmed_reasons),
        dangerous_ops_confirmed_reasons=tuple(confirmed_reasons),
        attempt_failures=tuple(attempt_failures),
        assertion_gaming_suspected=bool(assertion_gaming_flags),
        assertion_gaming_details="; ".join(assertion_gaming_flags),
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
    on_progress: Callable[[str], None] | None = None,
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

    Each step is proposed and verified against a task-scoped working copy of
    repo_root (Master Document Epic C2), not repo_root itself -- a plain
    copy_repo() at the start of run_task, into which every step's diff gets
    applied (execution.test_runner.apply_diff) immediately after that step
    succeeds. Without this, a later step in the same plan (e.g. a paired
    test step verifying a function an earlier step just added) would have
    its diff proposed and sandbox-tested against the *original*, pre-task
    repo content, missing every earlier step's change -- a test for a
    brand-new function would fail to even import it. repo_root itself is
    never touched by any of this (consistent with execute_step_with_
    verification/run_tests_on_diff's existing contract that only
    execution.patch_applier.apply_to_new_branch, called once at the very
    end after run_task returns, is allowed to write to the real repo); the
    working copy is a task-scoped, disposable stand-in that exists only for
    the duration of this call. file content for each step is read fresh
    from the working copy immediately before that step runs, for the same
    reason.

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
    last_impl_diff: Diff | None = None

    with tempfile.TemporaryDirectory() as scratch_dir:
        working_root: Path | None = None
        if repo_root is not None:
            working_root = Path(scratch_dir) / "repo"
            copy_repo(repo_root, working_root)

        for step_index, step in enumerate(plan.steps, start=1):
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

            file_path = working_root / step.file if working_root is not None else Path(step.file)
            file_content = file_path.read_text() if file_path.exists() else ""

            clamped_by_task_budget = remaining_budget < per_step_cap
            effective_step_cap = min(per_step_cap, remaining_budget)

            step_result = execute_step_with_verification(
                step,
                file_content,
                context,
                complete_fn=complete_fn,
                repo_root=working_root if working_root is not None else repo_root,
                config=config,
                max_retries=effective_step_cap,
                confirm_dangerous_ops=confirm_dangerous_ops,
                step_index=step_index,
                on_progress=on_progress,
            )

            if step_result.success and step_result.diff is not None:
                if is_test_file(step.file):
                    if last_impl_diff is not None:
                        coverage_reason = check_test_coverage_sanity(
                            last_impl_diff.diff_text, step_result.diff.diff_text
                        )
                        if coverage_reason is not None:
                            step_result = replace(
                                step_result,
                                weak_test_coverage_suspected=True,
                                weak_test_coverage_details=coverage_reason,
                            )
                            _emit(
                                on_progress,
                                f"Step {step_index}: flagged -- added test may not meaningfully "
                                f"cover the changed behavior: {coverage_reason}",
                            )
                else:
                    last_impl_diff = step_result.diff

                if working_root is not None:
                    apply_diff(working_root, step_result.diff)

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
