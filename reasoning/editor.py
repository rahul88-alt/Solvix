"""Turns one PlanStep plus the current content of its target file into a
unified diff (Master Document 7.3, Epic C1): the Editor stage.

Unlike the Planner, which reasons about the whole task at once, the Editor
is invoked once per PlanStep and only ever sees that step's target file.
The model is prompted to emit a unified diff only -- never a full-file
rewrite -- so changes stay minimal and reviewable. New files are expressed
as a diff against /dev/null so Diff.diff_text is always a real unified diff
that can be validated the same way (via `patch --dry-run`) regardless of
whether it edits or creates a file.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from reasoning.linter import LintViolation, new_violations, run_linter
from reasoning.llm_client import OllamaUnavailableError, complete
from reasoning.planner import PlanStep
from reasoning.task_input import TaskContext

_SYSTEM_PROMPT = (
    "You are a software editing assistant. Given a task, one plan step "
    "describing a change to a single file, and that file's current content, "
    "produce the change as a unified diff.\n\n"
    "Respond with ONLY a unified diff (--- / +++ headers and @@ hunks), no "
    "prose, no markdown code fences. Never respond with the full rewritten "
    "file -- only the diff hunks for the lines that change. If the step "
    "requires creating a new file, express it as a diff against /dev/null "
    "(--- /dev/null, +++ b/<path>) that adds the new file's content."
)

_DIFF_HEADER_RE = re.compile(
    r"^--- (?P<old>\S+).*\n\+\+\+ (?P<new>\S+).*\n(?:@@.*@@.*\n(?:.*\n?)*)+",
    re.MULTILINE,
)

_CORRECTION_TEMPLATE = (
    "Your previous response was not a valid unified diff that applies "
    "cleanly to the given file content. Problem: {error}\n\n"
    "Previous response:\n{previous}\n\n"
    "Respond again with ONLY a corrected unified diff, no prose, no markdown fences."
)

_LINT_CORRECTION_TEMPLATE = (
    "Your previous diff applies cleanly but introduces new lint violations "
    "that were not present in the original file:\n{violations}\n\n"
    "Previous response:\n{previous}\n\n"
    "Respond again with ONLY a corrected unified diff that makes the same "
    "change without introducing these violations, no prose, no markdown fences."
)


class DiffGenerationError(Exception):
    """Raised when the model's diff output can't be parsed or doesn't apply."""


def _complete_or_raise(
    complete_fn: Callable[[str, list[dict]], str], system: str, messages: list[dict]
) -> str:
    """Call complete_fn, converting an OllamaUnavailableError (SLX-F4: Ollama
    unreachable mid-run, e.g. it was stopped between propose_diff attempts)
    into a DiffGenerationError -- the existing failure mode
    execute_step_with_verification's outer retry loop already knows how to
    absorb (fed back as this attempt's failure, eventually a needs_human_help
    StepResult once retries are exhausted), rather than a raw exception
    escaping propose_diff entirely.
    """
    try:
        return complete_fn(system, messages)
    except OllamaUnavailableError as error:
        raise DiffGenerationError(str(error)) from error


@dataclass(frozen=True)
class Diff:
    target_file: str
    diff_text: str
    is_new_file: bool
    lint_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreviousAttempt:
    """A prior diff for this same step that applied and lint-passed but
    failed the real test suite, plus the actual failure output (Master
    Document 7.3, Epic C4). Passed back into propose_diff so the outer
    test-retry loop in execution.orchestrator can ask for a genuinely
    revised diff instead of a blind "try again".
    """

    diff_text: str
    failure_output: str


def _strip_fences(raw_text: str) -> str:
    text = raw_text.strip()
    fence_match = re.match(r"^```(?:diff|patch)?\n(.*)\n```$", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return text


_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@(.*)$")


def _restore_blank_context_lines(lines: list[str]) -> list[str]:
    """Local models routinely emit unchanged blank lines inside a hunk as
    truly empty lines instead of a single context space, which `patch`
    rejects as malformed. Restore the leading space so a hunk line is
    always one of ' ', '+', '-', or a diff header line.
    """
    fixed = []
    in_hunk = False
    for line in lines:
        if line.startswith(("--- ", "+++ ")):
            in_hunk = False
        elif line.startswith("@@"):
            in_hunk = True
        elif in_hunk and line == "":
            line = " "
        fixed.append(line)
    return fixed


def _recompute_hunk_headers(lines: list[str]) -> list[str]:
    """Local models frequently get the "@@ -a,b +c,d @@" line-count
    arithmetic wrong even when the hunk body itself is fine, which makes
    `patch` bail out as malformed before it ever checks whether the body
    applies. Recompute b/d from the actual body so only genuine content
    mismatches are left for `patch` to catch.
    """
    fixed = []
    i = 0
    while i < len(lines):
        match = _HUNK_HEADER_RE.match(lines[i])
        if match is None:
            fixed.append(lines[i])
            i += 1
            continue

        old_start, new_start, rest = match.groups()
        i += 1
        body = []
        while i < len(lines) and not lines[i].startswith(("@@", "--- ", "+++ ")):
            body.append(lines[i])
            i += 1

        old_count = sum(1 for line in body if line.startswith((" ", "-")))
        new_count = sum(1 for line in body if line.startswith((" ", "+")))
        fixed.append(f"@@ -{old_start},{old_count} +{new_start},{new_count} @@{rest}")
        fixed.extend(body)
    return fixed


def _extract_diff_text(raw_text: str) -> str:
    text = _strip_fences(raw_text)
    match = _DIFF_HEADER_RE.search(text)
    if match is None:
        raise DiffGenerationError(
            f"model response did not contain a unified diff: {raw_text!r}"
        )
    lines = text[match.start() :].strip().splitlines()
    lines = _restore_blank_context_lines(lines)
    lines = _recompute_hunk_headers(lines)
    return "\n".join(lines) + "\n"


def _is_new_file_diff(diff_text: str) -> bool:
    first_lines = diff_text.splitlines()[:1]
    return bool(first_lines) and "/dev/null" in first_lines[0]


def _validate_applies_cleanly(diff_text: str, file_content: str, is_new_file: bool) -> None:
    """Dry-run `patch` against the given file content to confirm the diff
    applies. Raises DiffGenerationError if it doesn't.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        target_path = Path(tmp_dir) / "target"
        if not is_new_file:
            target_path.write_text(file_content)
        # For new files, target_path is deliberately left absent: `patch`
        # creates it fresh from the diff's hunk when given as an explicit
        # positional argument that doesn't yet exist.

        diff_path = Path(tmp_dir) / "change.diff"
        diff_path.write_text(diff_text)

        result = subprocess.run(
            ["patch", "--dry-run", "--forward", str(target_path), str(diff_path)],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            raise DiffGenerationError(
                f"diff did not apply cleanly against file content: {result.stdout}{result.stderr}"
            )


def _apply_diff_to_content(diff_text: str, file_content: str, is_new_file: bool) -> str:
    """Apply diff_text for real (not --dry-run) against file_content and
    return the resulting file content, for feeding to the linter.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        target_path = Path(tmp_dir) / "target"
        if not is_new_file:
            target_path.write_text(file_content)

        diff_path = Path(tmp_dir) / "change.diff"
        diff_path.write_text(diff_text)

        subprocess.run(
            ["patch", "--forward", str(target_path), str(diff_path)],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            check=True,
        )
        return target_path.read_text()


def _lint_target_path(step_file: str) -> str | None:
    """Only Python files are lint-gated at MVP (Master Document 7.1 target
    language); other file types skip the lint check entirely.
    """
    return step_file if step_file.endswith(".py") else None


def _new_lint_violations(
    step: PlanStep,
    file_content: str,
    diff_text: str,
    is_new_file: bool,
    repo_root: str | Path | None,
):
    """Apply diff_text to file_content and return any lint violations it
    introduces that weren't already present in file_content, or () if the
    file isn't lint-gated or the linter isn't available.

    The scratch copies are written inside repo_root (when given) rather
    than an unrelated temp directory, so ruff's normal upward config
    discovery finds the project's own pyproject.toml/ruff.toml -- linting
    with the project's actual rules, not ruff's bare defaults.
    """
    lint_path = _lint_target_path(step.file)
    if lint_path is None:
        return ()

    patched_content = _apply_diff_to_content(diff_text, file_content, is_new_file)

    with tempfile.TemporaryDirectory(dir=repo_root) as tmp_dir:
        lint_scratch_path = Path(tmp_dir) / Path(lint_path).name

        lint_scratch_path.write_text(file_content)
        before_result = run_linter(lint_scratch_path)

        lint_scratch_path.write_text(patched_content)
        after_result = run_linter(lint_scratch_path)

    return new_violations(before_result, after_result)


def _format_violations(violations: tuple[LintViolation, ...]) -> str:
    return "\n".join(f"- line {v.line}: {v.rule} {v.message}" for v in violations)


def _build_user_message(
    step: PlanStep,
    file_content: str,
    task_context: TaskContext,
    previous_attempt: PreviousAttempt | None,
) -> str:
    message = (
        f"Overall task: {task_context.task}\n\n"
        f"Plan step for file {step.file}: {step.description}\n\n"
        f"Current content of {step.file}:\n"
        "-----\n"
        f"{file_content}\n"
        "-----"
    )
    if previous_attempt is not None:
        message += (
            "\n\nA previous attempt at this step applied cleanly but failed the "
            "project's test suite. Do not repeat the same change -- produce a "
            "genuinely revised diff that addresses the actual failure below.\n\n"
            f"Previous diff:\n{previous_attempt.diff_text}\n\n"
            f"Test failure output:\n{previous_attempt.failure_output}"
        )
    return message


def propose_diff(
    step: PlanStep,
    file_content: str,
    context: TaskContext,
    complete_fn: Callable[[str, list[dict]], str] = complete,
    repo_root: str | Path | None = None,
    previous_attempt: PreviousAttempt | None = None,
) -> Diff:
    """Produce a unified diff for a single PlanStep against file_content.

    A response that fails to parse as a unified diff, or that doesn't apply
    cleanly against file_content, triggers one retry with a correction
    prompt before giving up (same pattern as reasoning.planner.generate_plan).

    repo_root, when given, is the real on-disk repo root -- passed through
    to the lint gate so ruff discovers the project's own config instead of
    falling back to bare defaults (Master Document A3: respect project
    conventions).

    previous_attempt, when given, is a prior diff for this same step that
    failed the real test suite (Master Document Epic C4); its diff and
    failure output are included in the prompt so the model can propose a
    genuinely revised diff instead of repeating the same mistake. This is a
    distinct, outer retry loop from the lint-correction retry above, driven
    by execution.orchestrator rather than by propose_diff itself.
    """
    messages = [
        {
            "role": "user",
            "content": _build_user_message(step, file_content, context, previous_attempt),
        }
    ]
    raw_response = _complete_or_raise(complete_fn, _SYSTEM_PROMPT, messages)

    try:
        diff_text = _extract_diff_text(raw_response)
        is_new_file = _is_new_file_diff(diff_text)
        _validate_applies_cleanly(diff_text, file_content, is_new_file)
    except DiffGenerationError as first_error:
        correction = _CORRECTION_TEMPLATE.format(error=first_error, previous=raw_response)
        messages = [
            *messages,
            {"role": "assistant", "content": raw_response},
            {"role": "user", "content": correction},
        ]
        raw_response = _complete_or_raise(complete_fn, _SYSTEM_PROMPT, messages)
        diff_text = _extract_diff_text(raw_response)
        is_new_file = _is_new_file_diff(diff_text)
        _validate_applies_cleanly(diff_text, file_content, is_new_file)

    violations = _new_lint_violations(step, file_content, diff_text, is_new_file, repo_root)
    if violations:
        lint_correction = _LINT_CORRECTION_TEMPLATE.format(
            violations=_format_violations(violations), previous=raw_response
        )
        messages = [
            *messages,
            {"role": "assistant", "content": raw_response},
            {"role": "user", "content": lint_correction},
        ]
        raw_response = _complete_or_raise(complete_fn, _SYSTEM_PROMPT, messages)
        diff_text = _extract_diff_text(raw_response)
        is_new_file = _is_new_file_diff(diff_text)
        _validate_applies_cleanly(diff_text, file_content, is_new_file)

        violations = _new_lint_violations(step, file_content, diff_text, is_new_file, repo_root)
        # A single retry is given to fix lint issues; if the corrected diff
        # still introduces violations, accept it with a warning rather than
        # discarding an otherwise-correct change over an unfixable lint nit.

    return Diff(
        target_file=step.file,
        diff_text=diff_text,
        is_new_file=is_new_file,
        lint_warnings=tuple(f"{v.rule} (line {v.line}): {v.message}" for v in violations),
    )
