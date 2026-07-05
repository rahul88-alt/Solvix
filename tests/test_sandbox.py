from pathlib import Path
from unittest.mock import patch

import pytest

import execution.sandbox as sandbox_module
from execution.sandbox import (
    DockerUnavailableError,
    Sandbox,
    SandboxResult,
    _CONTAINER_PREFIX,
    _DEFAULT_IMAGE,
    reap_orphans,
)


@pytest.fixture(autouse=True)
def _reset_orphan_reap_flag():
    sandbox_module._orphans_reaped_this_process = False
    yield
    sandbox_module._orphans_reaped_this_process = False


def _completed(returncode=0, stdout="", stderr=""):
    class _Result:
        def __init__(self):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    return _Result()


@pytest.fixture(autouse=True)
def _fake_docker_available():
    with patch("execution.sandbox.subprocess.run", return_value=_completed(0)) as mock_run:
        yield mock_run


def test_sandbox_creates_and_removes_container_on_normal_exit(tmp_path):
    calls = []

    def fake_docker(*args, **kwargs):
        calls.append(args)
        return _completed(0)

    with patch("execution.sandbox._image_exists", return_value=True), patch(
        "execution.sandbox._docker", side_effect=fake_docker
    ):
        with Sandbox(tmp_path) as sandbox:
            assert sandbox._container_name is not None
            assert sandbox._container_name.startswith(_CONTAINER_PREFIX)
            name = sandbox._container_name

    verbs = [c[0] for c in calls]
    assert "create" in verbs
    assert "start" in verbs
    assert ("rm", "-f", name) in calls
    assert sandbox._container_name is None


def test_sandbox_removes_container_even_when_body_raises(tmp_path):
    calls = []

    def fake_docker(*args, **kwargs):
        calls.append(args)
        return _completed(0)

    with patch("execution.sandbox._image_exists", return_value=True), patch(
        "execution.sandbox._docker", side_effect=fake_docker
    ):
        with pytest.raises(RuntimeError):
            with Sandbox(tmp_path) as sandbox:
                name = sandbox._container_name
                raise RuntimeError("boom")

    assert ("rm", "-f", name) in calls


def test_sandbox_run_returns_result_from_docker_exec(tmp_path):
    def fake_docker(*args, **kwargs):
        if args[0] == "exec":
            return _completed(0, stdout="1 passed", stderr="")
        return _completed(0)

    with patch("execution.sandbox._image_exists", return_value=True), patch(
        "execution.sandbox._docker", side_effect=fake_docker
    ):
        with Sandbox(tmp_path) as sandbox:
            result = sandbox.run("pytest -q")

    assert isinstance(result, SandboxResult)
    assert result.exit_code == 0
    assert result.stdout == "1 passed"


def test_sandbox_uses_network_none_by_default(tmp_path):
    calls = []

    def fake_docker(*args, **kwargs):
        calls.append(args)
        return _completed(0)

    with patch("execution.sandbox._image_exists", return_value=True), patch(
        "execution.sandbox._docker", side_effect=fake_docker
    ):
        with Sandbox(tmp_path):
            pass

    create_call = next(c for c in calls if c[0] == "create")
    assert "--network" in create_call
    assert create_call[create_call.index("--network") + 1] == "none"


def test_sandbox_raises_clear_error_when_docker_unavailable(tmp_path):
    with patch("execution.sandbox.subprocess.run", side_effect=FileNotFoundError()):
        with pytest.raises(DockerUnavailableError):
            with Sandbox(tmp_path):
                pass


def test_reap_orphans_removes_leftover_containers_by_prefix():
    calls = []

    def fake_docker(*args, **kwargs):
        calls.append(args)
        if args[0] == "ps":
            return _completed(0, stdout="abc123\ndef456\n")
        return _completed(0)

    with patch("execution.sandbox._docker", side_effect=fake_docker):
        removed = reap_orphans()

    assert removed == ["abc123", "def456"]
    assert ("ps", "-a", "--filter", f"name={_CONTAINER_PREFIX}", "-q") in calls
    assert ("rm", "-f", "abc123", "def456") in calls


def test_reap_orphans_does_nothing_when_no_leftovers():
    calls = []

    def fake_docker(*args, **kwargs):
        calls.append(args)
        return _completed(0, stdout="")

    with patch("execution.sandbox._docker", side_effect=fake_docker):
        removed = reap_orphans()

    assert removed == []
    assert not any(c[0] == "rm" for c in calls)


def test_sandbox_enter_reaps_orphans_once_per_process(tmp_path):
    reap_calls = []

    def fake_docker(*args, **kwargs):
        if args[0] == "ps":
            reap_calls.append(args)
        return _completed(0)

    with patch("execution.sandbox._image_exists", return_value=True), patch(
        "execution.sandbox._docker", side_effect=fake_docker
    ):
        with Sandbox(tmp_path):
            pass
        with Sandbox(tmp_path):
            pass

    assert len(reap_calls) == 1


def test_sandbox_create_failure_does_not_leave_container_name_set(tmp_path):
    def fake_docker(*args, **kwargs):
        if args[0] == "create":
            return _completed(1, stderr="no such image")
        return _completed(0)

    with patch("execution.sandbox._image_exists", return_value=True), patch(
        "execution.sandbox._docker", side_effect=fake_docker
    ):
        with pytest.raises(RuntimeError):
            with Sandbox(tmp_path):
                pass


def test_sandbox_installs_repo_requirements_before_running_tests(tmp_path):
    """SLX-E5: a target repo with a requirements.txt should get those
    packages installed into a derived sandbox image (built once, offline
    test run after), instead of every test collection failing with
    ModuleNotFoundError against the generic pytest/ruff-only default image.
    """
    (tmp_path / "requirements.txt").write_text("click\nrequests\n")

    build_calls = []

    def fake_subprocess_run(args, **kwargs):
        if args[:2] == ["docker", "build"]:
            build_calls.append(args)
            build_dir = Path(args[-1])
            dockerfile = (build_dir / "Dockerfile").read_text()
            assert "pip install --no-cache-dir -r" in dockerfile
            requirements = (build_dir / "requirements.txt").read_text()
            assert "click" in requirements
            assert "requests" in requirements
        return _completed(0)

    calls = []

    def fake_docker(*args, **kwargs):
        calls.append(args)
        return _completed(0)

    with patch(
        "execution.sandbox._image_exists", side_effect=lambda image: image == _DEFAULT_IMAGE
    ), patch("execution.sandbox._docker", side_effect=fake_docker), patch(
        "execution.sandbox.subprocess.run", side_effect=fake_subprocess_run
    ):
        with Sandbox(tmp_path) as sandbox:
            used_image = sandbox.image

    assert len(build_calls) == 1
    assert used_image != _DEFAULT_IMAGE
    assert used_image.startswith(f"{_DEFAULT_IMAGE}-deps-")

    create_call = next(c for c in calls if c[0] == "create")
    assert used_image in create_call


def test_sandbox_reuses_cached_dependency_image(tmp_path):
    (tmp_path / "requirements.txt").write_text("click\n")

    build_calls = []

    def fake_subprocess_run(args, **kwargs):
        if args[:2] == ["docker", "build"]:
            build_calls.append(args)
        return _completed(0)

    with patch("execution.sandbox._image_exists", return_value=True), patch(
        "execution.sandbox._docker", side_effect=lambda *a, **k: _completed(0)
    ), patch("execution.sandbox.subprocess.run", side_effect=fake_subprocess_run):
        with Sandbox(tmp_path) as sandbox:
            used_image = sandbox.image

    assert build_calls == []
    assert used_image.startswith(f"{_DEFAULT_IMAGE}-deps-")


def test_sandbox_uses_default_image_when_repo_has_no_dependency_file(tmp_path):
    with patch("execution.sandbox._image_exists", return_value=True), patch(
        "execution.sandbox._docker", side_effect=lambda *a, **k: _completed(0)
    ):
        with Sandbox(tmp_path) as sandbox:
            used_image = sandbox.image

    assert used_image == _DEFAULT_IMAGE
