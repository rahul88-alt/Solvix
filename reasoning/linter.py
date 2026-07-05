"""Runs the project's linter against a file and parses the violations
(Master Document 7.3/A3): used by the Editor to confirm agent-authored
diffs don't introduce new lint violations before they're returned as final.

MVP target is Python via `ruff`, matching the `lint_command` slot in
`.solvix.yml` (Master Document 7.5). Violations are compared by rule code
plus message text rather than raw line number, since a diff can shift
line numbers without actually moving the violation, and ruff's message
text (exact line length, exact unused-import name, ...) stays stable
across such shifts while the line number would not.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LintViolation:
    rule: str
    line: int
    message: str


@dataclass(frozen=True)
class LintResult:
    violations: tuple[LintViolation, ...]
    ran: bool


def run_linter(file_path: str | Path) -> LintResult:
    """Run `ruff check` against file_path and return its violations.

    If ruff isn't available in the environment, returns a LintResult with
    ran=False so callers can skip lint-based gating rather than fail outright.
    """
    try:
        result = subprocess.run(
            ["ruff", "check", "--output-format=json", str(file_path)],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return LintResult(violations=(), ran=False)

    # ruff exits 1 when it finds violations -- only treat other non-zero
    # codes (config errors, crashes) as "couldn't run".
    if result.returncode not in (0, 1):
        return LintResult(violations=(), ran=False)

    try:
        raw_violations = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        return LintResult(violations=(), ran=False)

    violations = tuple(
        LintViolation(
            rule=v["code"],
            line=v["location"]["row"],
            message=v["message"],
        )
        for v in raw_violations
        if v.get("code") is not None
    )
    return LintResult(violations=violations, ran=True)


def new_violations(before: LintResult, after: LintResult) -> tuple[LintViolation, ...]:
    """Return the violations present in `after` but not in `before`, matching
    on rule code + message rather than line number, since ruff's message
    text (e.g. exact line length, exact unused-import name) stays stable
    when a violation merely shifts line due to unrelated edits elsewhere in
    the diff, while raw line number would not.
    """
    if not before.ran or not after.ran:
        return ()

    before_keys = {(v.rule, v.message) for v in before.violations}

    result = []
    for v in after.violations:
        if (v.rule, v.message) in before_keys:
            continue
        result.append(v)
    return tuple(result)
