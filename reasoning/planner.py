"""Turns a TaskContext into an ordered, reviewable plan (Master Document 7.3,
Epic B3): a list of {file, description of change} steps, plus an approval
gate for plans that touch many files or any sensitive path.

The plan is produced by prompting the model to return JSON only; malformed
output is handled explicitly rather than letting a parse error crash the
caller.

Also home to the Epic C2 test-pairing step: for a task judged to be a
genuine behavior change (_is_behavior_change_task), every step targeting a
non-test .py file gets an extra step appended for its corresponding test
file, if the plan doesn't already have one. This reuses run_task's existing
generic multi-step loop as-is -- see _add_paired_test_steps's docstring.

Also home to check_ambiguity() (Master Document Epic B2): a distinct, cheap
LLM call meant to run before generate_plan, asking whether the task has
multiple materially different valid interpretations or is missing a
critical detail needed to act. See its docstring and _AMBIGUITY_SYSTEM_
PROMPT for how it's designed to avoid over-triggering on tasks that are
merely under-specified in ways that don't matter.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from config import SolvixConfig
from reasoning.llm_client import complete
from reasoning.task_input import TaskContext

_DEFAULT_MAX_FILES = 3
_SENSITIVE_PATH_PREFIXES = ("auth/", "billing/")

# Master Document Epic C2: keywords that mark a task as *not* a genuine
# behavior change (a pure refactor, rename, or doc/comment/formatting
# tidy-up), so it shouldn't get a test step forced onto it just because it
# happens to touch a .py file. This is deliberately a coarse, task-text-wide
# heuristic rather than an attempt at real intent classification -- see
# _is_behavior_change_task's docstring.
_REFACTOR_OR_DOC_KEYWORDS = (
    "refactor",
    "rename",
    "reformat",
    "formatting",
    "docstring",
    "comment",
    "typo",
    "cleanup",
    "clean up",
    "whitespace",
    "lint",
    "documentation",
    "readme",
)

_SYSTEM_PROMPT = (
    "You are a software planning assistant. Given a task description and a "
    "list of relevant files from the repository, produce an ordered plan of "
    "changes needed to complete the task.\n\n"
    "Respond with ONLY a JSON array, no prose, no markdown code fences. Each "
    "element must be an object with exactly two string fields: \"file\" (a "
    "repo-relative path from the provided file list) and \"description\" (a "
    "concise description of the change needed in that file)."
)

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)

_CORRECTION_TEMPLATE = (
    "Your previous response could not be parsed as a JSON array of "
    '{{"file": ..., "description": ...}} objects. Parse error: {error}\n\n'
    "Previous response:\n{previous}\n\n"
    "Respond again with ONLY a corrected JSON array, no prose, no markdown fences."
)


class PlanGenerationError(Exception):
    """Raised when the model's plan output can't be parsed into PlanSteps."""


@dataclass(frozen=True)
class Clarification:
    """A single check_ambiguity question/answer round (Master Document Epic
    B2), carried alongside a TaskContext/Plan through to review.pr_builder so
    a reviewer can see *why* the plan reflects a particular reading of the
    task, not just what the plan ended up being -- the same "any decision
    that shaped the output belongs in the PR" principle Epic D2 already
    applies to plan-approval reasons and dangerous-ops confirmations.
    """

    question: str
    answer: str


@dataclass(frozen=True)
class PlanStep:
    file: str
    description: str


@dataclass(frozen=True)
class Plan:
    steps: list[PlanStep]
    requires_approval: bool
    approval_reasons: tuple[str, ...]


# Master Document Epic B2: system prompt for check_ambiguity, a distinct,
# cheap LLM call that runs BEFORE generate_plan. This is the same
# false-positive risk class as check_dangerous_ops (Epic E2) and
# check_test_coverage_sanity (Epic C2): a local model asked "is this
# ambiguous?" will happily say yes to almost anything if not anchored, since
# nearly every task *could* be more detailed. The prompt draws the line
# explicitly -- "multiple materially different valid interpretations" or "a
# critical missing detail that blocks any reasonable attempt" counts;
# "could be more specific" does not -- and backs that line with worked
# examples in both directions, rather than leaving the model to infer the
# threshold from the word "ambiguous" alone.
_AMBIGUITY_SYSTEM_PROMPT = (
    "You are a scope-clarification checker for an autonomous coding agent. "
    "Given a task description and the relevant files found for it, decide "
    "whether the task is genuinely ambiguous.\n\n"
    "A task is genuinely ambiguous only if EITHER:\n"
    "  (a) it has multiple materially different valid interpretations -- "
    "different implementations would each satisfy the literal wording, but "
    "would do substantively different things, OR\n"
    "  (b) it is missing a critical detail without which no reasonable "
    "implementation could even be attempted.\n\n"
    "Do NOT flag a task as ambiguous just because it could theoretically be "
    "more detailed, more specific, or more thorough -- that is true of "
    "almost every real task and is not a reason to ask a clarifying "
    "question. A task that names a specific function, file, behavior, or "
    "bug is clear enough to proceed even if it doesn't spell out every "
    "implementation detail.\n\n"
    "Examples of CLEAR tasks (do not flag, ambiguous=false, question=null):\n"
    '  - "fix the subtract function in calculator.py to handle negative numbers"\n'
    '  - "add an is_palindrome(s) function to strings.py"\n'
    '  - "add a reverse(text) function to utils/strings.py matching the style of slugify"\n\n'
    "Examples of genuinely AMBIGUOUS tasks (flag, ambiguous=true):\n"
    '  - "improve the calculator" (improve how -- performance, new '
    "operations, error handling, something else?)\n"
    '  - "make strings.py better" (better in what way?)\n'
    '  - "add validation" (validation of what input, using what rules?)\n\n'
    "Respond with ONLY a JSON object, no prose, no markdown fences: "
    '{"ambiguous": true or false, "question": "..." or null}. '
    "If ambiguous, \"question\" must name the specific concrete choice or "
    "missing detail (e.g. \"Which aspect of the calculator should be "
    "improved -- error handling for invalid input, support for additional "
    "operations, or something else?\"), never a generic \"Could you "
    "clarify?\" or \"What do you mean?\". If not ambiguous, \"question\" must be null."
)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_ambiguity_json(raw_text: str) -> Optional[str]:
    """Parse check_ambiguity's expected {"ambiguous": bool, "question": str|null}
    response, returning the question if genuinely ambiguous or None otherwise.

    Fails open (returns None, i.e. "proceed without asking") on any parse
    problem -- malformed JSON, a missing/wrong-typed "ambiguous" field, or an
    ambiguous=true response with no usable question string. This check is a
    convenience gate, not a safety gate like check_dangerous_ops; blocking an
    entire task run because this one classifier call came back malformed
    would be a worse outcome than occasionally missing genuine ambiguity.
    """
    match = _JSON_OBJECT_RE.search(raw_text)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("ambiguous"), bool):
        return None
    if not data["ambiguous"]:
        return None
    question = data.get("question")
    if not isinstance(question, str) or not question.strip():
        return None
    return question.strip()


def _build_user_message(task_context: TaskContext) -> str:
    file_list = "\n".join(
        f"- {f.file_path}" for f in task_context.retrieval.files
    )
    related = "\n".join(
        f"- {f.file_path}" for f in task_context.retrieval.related_files
    )
    parts = [f"Task: {task_context.task}", "", "Relevant files:", file_list or "(none found)"]
    if related:
        parts += ["", "Related files (one-hop imports/callers):", related]
    return "\n".join(parts)


def check_ambiguity(
    task_context: TaskContext,
    complete_fn: Callable[[str, list[dict]], str] = complete,
) -> Optional[str]:
    """Ask the model, in a distinct call before generate_plan runs, whether
    task_context.task has multiple materially different valid
    interpretations or is missing a critical detail needed to act (Master
    Document Epic B2).

    Returns None if the task is clear enough to proceed as-is (the common
    case), or a specific clarifying question if genuinely ambiguous. Never
    raises on a malformed model response -- see _parse_ambiguity_json.
    """
    messages = [{"role": "user", "content": _build_user_message(task_context)}]
    raw_response = complete_fn(_AMBIGUITY_SYSTEM_PROMPT, messages)
    return _parse_ambiguity_json(raw_response)


def _parse_plan_json(raw_text: str) -> list[PlanStep]:
    match = _JSON_ARRAY_RE.search(raw_text)
    if match is None:
        raise PlanGenerationError(f"model response did not contain a JSON array: {raw_text!r}")

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise PlanGenerationError(f"model response was not valid JSON: {e}") from e

    if not isinstance(data, list):
        raise PlanGenerationError(f"expected a JSON array of steps, got {type(data).__name__}")

    steps = []
    for i, item in enumerate(data):
        if not isinstance(item, dict) or "file" not in item or "description" not in item:
            raise PlanGenerationError(
                f"step {i} is missing required \"file\"/\"description\" fields: {item!r}"
            )
        steps.append(PlanStep(file=str(item["file"]), description=str(item["description"])))

    if not steps:
        raise PlanGenerationError("model returned an empty plan")

    return steps


def is_test_file(file_path: str) -> bool:
    """True if file_path already looks like a test file (its own name
    starts with "test_", or it lives under a tests/ directory) -- used both
    here (to avoid pairing a test step onto a step that already targets a
    test file) and by execution.orchestrator (to know which step in a
    completed plan is the "test step" a paired implementation diff should
    be sanity-checked against, Epic C2).
    """
    normalized = file_path.replace("\\", "/")
    return Path(normalized).name.startswith("test_") or "/tests/" in f"/{normalized}"


def _is_behavior_change_task(task_text: str) -> bool:
    """Coarse, deliberately imperfect heuristic (Master Document Epic C2)
    for whether task_text describes a genuine behavior change that
    warrants a paired test step, versus a pure refactor/rename/doc/
    formatting change that doesn't touch observable behavior at all.

    Defaults to True (behavior change) whenever none of
    _REFACTOR_OR_DOC_KEYWORDS appear -- most real tasks ("fix X", "add Y",
    "handle Z") are genuine behavior changes, so the safer default is to
    pair a test and let a human notice an unnecessary one, rather than
    silently skip testing a real change because the heuristic missed it.
    """
    lowered = task_text.lower()
    return not any(keyword in lowered for keyword in _REFACTOR_OR_DOC_KEYWORDS)


def _paired_test_file(source_file: str) -> str:
    """Derive the paired test file path for source_file, following
    sample_repo's own tests/ convention (tests/test_<module>.py, flat --
    see sample_repo/tests/test_calculator.py) rather than mirroring the
    source's own directory structure.
    """
    return f"tests/test_{Path(source_file).stem}.py"


def _add_paired_test_steps(steps: list[PlanStep], task_text: str) -> list[PlanStep]:
    """Append a paired test-file step for each behavior-changing, non-test
    step in steps that doesn't already have a corresponding test step in
    the plan (Master Document Epic C2).

    Whether to pair at all is decided once, from the overall task text
    (_is_behavior_change_task), not per step -- a purely cosmetic task
    shouldn't get test steps forced onto it just because it happens to
    touch a .py file, and a single genuine behavior-change task should get
    every source file it touches paired, not just the first.

    This deliberately does not build any new step-sequencing machinery: it
    just appends ordinary PlanStep entries, so execution.orchestrator.
    run_task's existing generic multi-step loop (Epic E3) runs the paired
    test step exactly like any other step, in order, right after the
    implementation step it was derived from.
    """
    if not _is_behavior_change_task(task_text):
        return steps

    existing_files = {s.file for s in steps}
    augmented = list(steps)
    for step in steps:
        if is_test_file(step.file) or not step.file.endswith(".py"):
            continue
        test_file = _paired_test_file(step.file)
        if test_file in existing_files:
            continue
        augmented.append(
            PlanStep(
                file=test_file,
                description=(
                    f"Add or update a test covering the change in {step.file}: "
                    f"{step.description}"
                ),
            )
        )
        existing_files.add(test_file)
    return augmented


def _check_approval(
    steps: list[PlanStep],
    max_files: int,
    sensitive_prefixes: tuple[str, ...],
) -> tuple[bool, tuple[str, ...]]:
    reasons = []
    touched_files = {s.file for s in steps}
    if len(touched_files) > max_files:
        reasons.append(f"touches {len(touched_files)} files (limit {max_files})")

    sensitive_hits = sorted(
        f for f in touched_files if any(f.startswith(p) for p in sensitive_prefixes)
    )
    if sensitive_hits:
        reasons.append(f"touches sensitive path(s): {', '.join(sensitive_hits)}")

    return bool(reasons), tuple(reasons)


def generate_plan(
    task_context: TaskContext,
    complete_fn: Callable[[str, list[dict]], str] = complete,
    max_files: int = _DEFAULT_MAX_FILES,
    sensitive_paths: tuple[str, ...] = _SENSITIVE_PATH_PREFIXES,
    config: SolvixConfig | None = None,
) -> Plan:
    """Produce an ordered plan for task_context, flagging it for approval if
    it touches more than max_files files or any sensitive path prefix.

    When config is given, config.sensitive_paths (from `.solvix.yml`
    paths.sensitive, always merged with the auth/billing built-in defaults --
    see SolvixConfig.__post_init__) takes precedence over the sensitive_paths
    argument; otherwise sensitive_paths' own default is used, so existing
    callers that don't load a config are unaffected (Master Document Epic F3).

    Local, smaller models are more prone to slightly malformed JSON than
    frontier models, so a single malformed response triggers one retry with
    a correction prompt before giving up.
    """
    messages = [{"role": "user", "content": _build_user_message(task_context)}]
    raw_response = complete_fn(_SYSTEM_PROMPT, messages)

    try:
        steps = _parse_plan_json(raw_response)
    except PlanGenerationError as first_error:
        correction = _CORRECTION_TEMPLATE.format(error=first_error, previous=raw_response)
        messages = [
            *messages,
            {"role": "assistant", "content": raw_response},
            {"role": "user", "content": correction},
        ]
        raw_response = complete_fn(_SYSTEM_PROMPT, messages)
        steps = _parse_plan_json(raw_response)

    steps = _add_paired_test_steps(steps, task_context.task)

    effective_sensitive_paths = (
        config.sensitive_paths if config is not None else sensitive_paths
    )
    requires_approval, reasons = _check_approval(steps, max_files, effective_sensitive_paths)
    return Plan(steps=steps, requires_approval=requires_approval, approval_reasons=reasons)
