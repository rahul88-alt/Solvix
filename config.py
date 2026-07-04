"""Loads and validates `.solvix.yml` (Master Document 7.2/7.5, Epic F3): the
single source of truth for which parts of a target repo the agent may touch,
plus the handful of other per-repo settings already wired into the pipeline
(test command, retry limit).

`retries.max_attempts` and `retries.max_task_attempts` are deliberately
distinct knobs, not the same limit reused: max_attempts bounds
execution.orchestrator.execute_step_with_verification's retries within a
single plan step (Epic C4), while max_task_attempts bounds the sum of
attempts across an entire plan's steps in execution.orchestrator.run_task
(Epic E3). A stuck single step must not silently need a change to the
task-wide cap, and vice versa.

Two distinct path mechanisms live here, and they are NOT the same thing with
different names:

- `paths.deny` is a HARD block, enforced by callers (execution.orchestrator)
  before propose_diff() is ever invoked -- a denied file is never sent to the
  LLM at all.
- `paths.sensitive` only flags a plan for human approval (reasoning.planner);
  the change still gets generated, it just can't proceed unreviewed.
  Config-supplied sensitive paths are always additive to the built-in
  defaults (auth/, billing/) -- see SolvixConfig.__post_init__ -- so writing
  `.solvix.yml` to protect one more path can never silently remove
  protection on a path the author didn't mention.

Fields from the Master Document 7.5 schema that aren't wired into any stage
yet (sandbox.*, lint_command, language) are parsed and stored with sensible
defaults so `.solvix.yml` authors can write the full schema now, but nothing
in the pipeline reads them until the story that needs them (e.g. sandbox
settings land with SLX-E1).

`dangerous_ops` (Epic E2) follows the same additive-to-built-in-defaults
rule as `paths.sensitive`: it flags a diff or command for mandatory human
confirmation (execution.orchestrator.check_dangerous_ops) rather than
silently letting a force-push, hard reset, branch deletion, or destructive
SQL statement proceed unreviewed.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path

import yaml

_CONFIG_FILENAME = ".solvix.yml"

_DEFAULT_LANGUAGE = "python"
_DEFAULT_TEST_COMMAND = "pytest -q"
_DEFAULT_LINT_COMMAND = "ruff check ."
_DEFAULT_DENY_PATHS: tuple[str, ...] = ()
_DEFAULT_SENSITIVE_PATHS: tuple[str, ...] = ("auth/", "billing/")
_DEFAULT_MAX_ATTEMPTS = 3
_DEFAULT_MAX_TASK_ATTEMPTS = 10
_DEFAULT_SANDBOX_BASE_IMAGE = "auto"
_DEFAULT_SANDBOX_NETWORK = "install-only"

# Regex patterns (matched case-insensitively against diff text and shell
# commands) flagging operations destructive enough to require explicit human
# confirmation before proceeding (Master Document 7.3/7.6, Epic E2). Like
# _DEFAULT_SENSITIVE_PATHS, these are a floor, not a ceiling -- see
# SolvixConfig.__post_init__.
#
# The SQL patterns (DROP TABLE/DATABASE, TRUNCATE) require a target
# identifier immediately followed by whatever ends a SQL statement in
# practice -- a quote, semicolon, newline, or end of string -- rather than a
# bare keyword match. This is what lets `TRUNCATE orders` inside a real
# `cursor.execute('TRUNCATE orders')` still fire, while `def truncate(text,
# length):`, a `truncate_string` identifier, or the word "truncate" in a
# comment/docstring do not (SLX-E4: those bare-word matches were the
# repeated false positive against sample_repo's own utils/strings.py). The
# optional `IF [NOT] EXISTS` clause is skipped over so the common defensive
# `DROP TABLE IF EXISTS <name>` migration idiom still matches.
_IF_EXISTS = r"(?:IF\s+(?:NOT\s+)?EXISTS\s+)?"
_SQL_TARGET = r"[A-Za-z_][A-Za-z0-9_]*\b\s*(?=[;'\"\n]|$)"
_DEFAULT_DANGEROUS_OPS: tuple[str, ...] = (
    r"git\s+push\b[^\n]*(--force\b|-f\b)",
    r"git\s+reset\s+--hard\b",
    r"git\s+branch\s+.*-D\b",
    r"git\s+push\b[^\n]*--delete\b",
    r"\bDROP\s+TABLE\s+" + _IF_EXISTS + _SQL_TARGET,
    r"\bDROP\s+DATABASE\s+" + _IF_EXISTS + _SQL_TARGET,
    r"\bTRUNCATE\s+(?:TABLE\s+)?" + _IF_EXISTS + _SQL_TARGET,
)


def _matches_pattern(file_path: str, pattern: str) -> bool:
    """True if file_path matches pattern, supporting glob wildcards
    ("secrets/**", ".env*") as well as plain directory prefixes ("auth/")
    with no wildcard at all.
    """
    normalized = file_path.replace("\\", "/")
    if fnmatch.fnmatch(normalized, pattern):
        return True
    if fnmatch.fnmatch(Path(normalized).name, pattern):
        return True
    return normalized.startswith(pattern.rstrip("*"))


def _normalize_prefix(pattern: str) -> str:
    """Strip trailing glob wildcards so a schema-style pattern like
    "auth/**" degrades to the plain directory prefix "auth/" that
    reasoning.planner's existing sensitive-path check already understands.
    """
    return pattern.rstrip("*")


@dataclass(frozen=True)
class SolvixConfig:
    language: str = _DEFAULT_LANGUAGE
    test_command: str = _DEFAULT_TEST_COMMAND
    lint_command: str = _DEFAULT_LINT_COMMAND
    deny_paths: tuple[str, ...] = _DEFAULT_DENY_PATHS
    sensitive_paths: tuple[str, ...] = _DEFAULT_SENSITIVE_PATHS
    dangerous_ops: tuple[str, ...] = _DEFAULT_DANGEROUS_OPS
    max_retries: int = _DEFAULT_MAX_ATTEMPTS
    max_task_attempts: int = _DEFAULT_MAX_TASK_ATTEMPTS
    sandbox_base_image: str = _DEFAULT_SANDBOX_BASE_IMAGE
    sandbox_network: str = _DEFAULT_SANDBOX_NETWORK

    def __post_init__(self) -> None:
        """sensitive_paths and dangerous_ops are always additive to their
        built-in defaults, never a replacement -- a `.solvix.yml` author who
        adds one more sensitive path or dangerous-ops pattern should never
        silently lose protection on the ones they didn't mention. This
        applies regardless of whether the config came from load_config() or
        was constructed directly, so there's one consistent guarantee
        everywhere a SolvixConfig can come from.
        """
        merged_sensitive = tuple(dict.fromkeys((*_DEFAULT_SENSITIVE_PATHS, *self.sensitive_paths)))
        object.__setattr__(self, "sensitive_paths", merged_sensitive)

        merged_dangerous = tuple(dict.fromkeys((*_DEFAULT_DANGEROUS_OPS, *self.dangerous_ops)))
        object.__setattr__(self, "dangerous_ops", merged_dangerous)

    def is_denied(self, file_path: str) -> bool:
        return any(_matches_pattern(file_path, p) for p in self.deny_paths)


def load_config(repo_path: str | Path) -> SolvixConfig:
    """Load SolvixConfig from `.solvix.yml` at the root of repo_path.

    Returns all-defaults SolvixConfig if the file doesn't exist, so a repo
    with no config at all still runs -- nothing forces a user to write
    `.solvix.yml` before their first task.
    """
    config_path = Path(repo_path) / _CONFIG_FILENAME
    if not config_path.exists():
        return SolvixConfig()

    raw = yaml.safe_load(config_path.read_text()) or {}
    paths = raw.get("paths") or {}
    retries = raw.get("retries") or {}
    sandbox = raw.get("sandbox") or {}

    sensitive_raw = paths.get("sensitive", _DEFAULT_SENSITIVE_PATHS)
    dangerous_ops_raw = raw.get("dangerous_ops", _DEFAULT_DANGEROUS_OPS)

    return SolvixConfig(
        language=raw.get("language", _DEFAULT_LANGUAGE),
        test_command=raw.get("test_command", _DEFAULT_TEST_COMMAND),
        lint_command=raw.get("lint_command", _DEFAULT_LINT_COMMAND),
        deny_paths=tuple(paths.get("deny", _DEFAULT_DENY_PATHS)),
        sensitive_paths=tuple(_normalize_prefix(p) for p in sensitive_raw),
        dangerous_ops=tuple(dangerous_ops_raw),
        max_retries=int(retries.get("max_attempts", _DEFAULT_MAX_ATTEMPTS)),
        max_task_attempts=int(retries.get("max_task_attempts", _DEFAULT_MAX_TASK_ATTEMPTS)),
        sandbox_base_image=sandbox.get("base_image", _DEFAULT_SANDBOX_BASE_IMAGE),
        sandbox_network=sandbox.get("network", _DEFAULT_SANDBOX_NETWORK),
    )
