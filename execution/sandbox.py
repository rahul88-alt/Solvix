"""Docker-based sandbox for all code execution (Master Document 7.2/7.3,
Epic E1): isolates generated-diff test runs from the host machine.

`Sandbox` is a context manager: it creates a container bind-mounting only
the caller's (already scratch-copied) repo directory, runs commands inside
it with `docker exec`, and always removes the container on exit --
success, failure, or exception -- so nothing is ever left running.

Containers run with `--network=none`: nothing in the current pipeline
needs network access while executing test/generated code (embeddings and
LLM calls happen outside the sandbox, in the host process). The one
exception is image preparation: the default sandbox image is a minimal
derivative of `python:3.11-slim` with `pytest`/`ruff` pre-installed, built
once via `docker build` (which does need network) and cached by tag --
that is a local image-prep step, not code execution, and every actual
test run still happens fully offline.

`__exit__`-based cleanup only runs if the Python process is alive to run
it -- a crash, `kill -9`, or a laptop sleep interrupting a run can leave a
container behind. `reap_orphans()` removes any `solvix-sandbox-*`
containers left over from such a prior process; `Sandbox.__enter__` calls
it once per process (not on every sandbox creation) as a stopgap until
there's a dedicated CLI entry point that can call it explicitly at true
process startup instead.
"""

from __future__ import annotations

import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_IMAGE = "solvix-sandbox:py311"
_CONTAINER_PREFIX = "solvix-sandbox-"
_DEFAULT_TIMEOUT = 300

_orphans_reaped_this_process = False

_DOCKERFILE = """\
FROM python:3.11-slim
RUN pip install --no-cache-dir pytest ruff
"""


class DockerUnavailableError(RuntimeError):
    """Raised when Docker is not installed or the daemon is not running."""


@dataclass(frozen=True)
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str


def _docker(*args: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, **kwargs)


def ensure_docker_available() -> None:
    """Raise DockerUnavailableError with a clear message if Docker isn't
    usable, rather than letting callers silently fall back to unsandboxed
    execution.
    """
    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, text=True, timeout=10
        )
    except FileNotFoundError as error:
        raise DockerUnavailableError(
            "Docker CLI not found. Install Docker Desktop before running "
            "sandboxed execution."
        ) from error
    except subprocess.TimeoutExpired as error:
        raise DockerUnavailableError(
            "`docker info` timed out. Is Docker Desktop running?"
        ) from error

    if result.returncode != 0:
        raise DockerUnavailableError(
            "Docker daemon is not available (`docker info` failed). Start "
            f"Docker Desktop before running sandboxed execution.\n{result.stderr}"
        )


def reap_orphans() -> list[str]:
    """Remove any solvix-sandbox-* containers left over from a prior
    process that never got to run its Sandbox.__exit__ (crash, kill -9,
    laptop sleep mid-run). Returns the container IDs that were removed.
    """
    list_result = _docker("ps", "-a", "--filter", f"name={_CONTAINER_PREFIX}", "-q")
    ids = [line for line in list_result.stdout.splitlines() if line.strip()]
    if ids:
        _docker("rm", "-f", *ids)
    return ids


def _image_exists(image: str) -> bool:
    return _docker("image", "inspect", image).returncode == 0


def _ensure_default_image_built() -> None:
    if _image_exists(_DEFAULT_IMAGE):
        return
    result = subprocess.run(
        ["docker", "build", "-t", _DEFAULT_IMAGE, "-"],
        input=_DOCKERFILE,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to build sandbox image {_DEFAULT_IMAGE}: {result.stderr}")


class Sandbox:
    """Context manager providing one disposable, network-isolated Docker
    container for a single repo snapshot.

    Usage:
        with Sandbox(repo_path) as sandbox:
            result = sandbox.run("pytest -q")
    """

    def __init__(
        self,
        repo_path: str | Path,
        image: str = _DEFAULT_IMAGE,
        network: str = "none",
        workdir: str = "/workspace",
    ) -> None:
        self.repo_path = str(Path(repo_path).resolve())
        self.image = image
        self.network = network
        self.workdir = workdir
        self._container_name: str | None = None

    def __enter__(self) -> "Sandbox":
        ensure_docker_available()

        global _orphans_reaped_this_process
        if not _orphans_reaped_this_process:
            reap_orphans()
            _orphans_reaped_this_process = True

        if self.image == _DEFAULT_IMAGE:
            _ensure_default_image_built()

        self._container_name = f"{_CONTAINER_PREFIX}{uuid.uuid4().hex[:12]}"
        create_result = _docker(
            "create",
            "--name", self._container_name,
            "--network", self.network,
            "-v", f"{self.repo_path}:{self.workdir}",
            "-w", self.workdir,
            self.image,
            "sleep", "infinity",
        )
        if create_result.returncode != 0:
            self._container_name = None
            raise RuntimeError(f"Failed to create sandbox container: {create_result.stderr}")

        start_result = _docker("start", self._container_name)
        if start_result.returncode != 0:
            self._cleanup()
            raise RuntimeError(f"Failed to start sandbox container: {start_result.stderr}")

        return self

    def run(self, command: str, timeout: int | None = _DEFAULT_TIMEOUT) -> SandboxResult:
        if self._container_name is None:
            raise RuntimeError("Sandbox is not active; use it as a context manager.")

        try:
            result = _docker("exec", self._container_name, "sh", "-c", command, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            _docker("kill", self._container_name)
            return SandboxResult(
                exit_code=-1,
                stdout=error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or ""),
                stderr=(error.stderr or "") + "\nSandbox command timed out.",
            )

        return SandboxResult(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)

    def __exit__(self, exc_type, exc, tb) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        if self._container_name is not None:
            _docker("rm", "-f", self._container_name)
            self._container_name = None
