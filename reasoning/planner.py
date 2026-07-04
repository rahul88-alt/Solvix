"""Turns a TaskContext into an ordered, reviewable plan (Master Document 7.3,
Epic B3): a list of {file, description of change} steps, plus an approval
gate for plans that touch many files or any sensitive path.

The plan is produced by prompting the model to return JSON only; malformed
output is handled explicitly rather than letting a parse error crash the
caller.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

from config import SolvixConfig
from reasoning.llm_client import complete
from reasoning.task_input import TaskContext

_DEFAULT_MAX_FILES = 3
_SENSITIVE_PATH_PREFIXES = ("auth/", "billing/")

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
class PlanStep:
    file: str
    description: str


@dataclass(frozen=True)
class Plan:
    steps: list[PlanStep]
    requires_approval: bool
    approval_reasons: tuple[str, ...]


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

    effective_sensitive_paths = (
        config.sensitive_paths if config is not None else sensitive_paths
    )
    requires_approval, reasons = _check_approval(steps, max_files, effective_sensitive_paths)
    return Plan(steps=steps, requires_approval=requires_approval, approval_reasons=reasons)
