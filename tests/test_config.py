from config import SolvixConfig, load_config


def _write_config(tmp_path, text):
    (tmp_path / ".solvix.yml").write_text(text)
    return tmp_path


def test_load_config_parses_real_yaml_fixture(tmp_path):
    _write_config(
        tmp_path,
        """
        language: python
        test_command: pytest -q --maxfail=1
        lint_command: ruff check .
        paths:
          deny:
            - "secrets/**"
            - ".env*"
          sensitive:
            - "auth/**"
            - "billing/**"
        retries:
          max_attempts: 5
        sandbox:
          base_image: auto
          network: install-only
        """,
    )

    config = load_config(tmp_path)

    assert config.language == "python"
    assert config.test_command == "pytest -q --maxfail=1"
    assert config.lint_command == "ruff check ."
    assert config.deny_paths == ("secrets/**", ".env*")
    assert config.sensitive_paths == ("auth/", "billing/")
    assert config.max_retries == 5
    assert config.sandbox_base_image == "auto"
    assert config.sandbox_network == "install-only"


def test_load_config_returns_defaults_when_no_file_present(tmp_path):
    config = load_config(tmp_path)

    assert config == SolvixConfig()
    assert config.deny_paths == ()
    assert config.sensitive_paths == ("auth/", "billing/")
    assert config.max_retries == 3
    assert config.test_command == "pytest -q"


def test_load_config_applies_defaults_for_missing_sections(tmp_path):
    _write_config(tmp_path, "language: python\n")

    config = load_config(tmp_path)

    assert config.deny_paths == ()
    assert config.sensitive_paths == ("auth/", "billing/")
    assert config.max_retries == 3


def test_is_denied_matches_glob_pattern():
    config = SolvixConfig(deny_paths=("secrets/**", ".env*"))

    assert config.is_denied("secrets/api_key.txt") is True
    assert config.is_denied("secrets/nested/dir/file.py") is True
    assert config.is_denied(".env.production") is True
    assert config.is_denied("app/main.py") is False


def test_is_denied_matches_plain_prefix_without_wildcard():
    config = SolvixConfig(deny_paths=("infra/",))

    assert config.is_denied("infra/deploy.sh") is True
    assert config.is_denied("infrastructure/other.py") is False


def test_sensitive_paths_are_additive_to_builtin_defaults_via_direct_construction():
    config = SolvixConfig(sensitive_paths=("payments/",))

    assert "auth/" in config.sensitive_paths
    assert "billing/" in config.sensitive_paths
    assert "payments/" in config.sensitive_paths


def test_load_config_merges_yaml_sensitive_paths_with_builtin_defaults(tmp_path):
    _write_config(
        tmp_path,
        """
        paths:
          sensitive:
            - "payments/**"
        """,
    )

    config = load_config(tmp_path)

    assert "auth/" in config.sensitive_paths
    assert "billing/" in config.sensitive_paths
    assert "payments/" in config.sensitive_paths


def test_dangerous_ops_defaults_cover_common_destructive_patterns():
    config = SolvixConfig()

    assert any("--force" in p for p in config.dangerous_ops)
    assert any("reset" in p and "hard" in p for p in config.dangerous_ops)
    assert any("DROP" in p for p in config.dangerous_ops)
    assert any("TRUNCATE" in p for p in config.dangerous_ops)


def test_dangerous_ops_are_additive_to_builtin_defaults_via_direct_construction():
    config = SolvixConfig(dangerous_ops=(r"kubectl\s+delete\s+namespace",))

    assert r"kubectl\s+delete\s+namespace" in config.dangerous_ops
    assert any("--force" in p for p in config.dangerous_ops)
    assert any("DROP" in p for p in config.dangerous_ops)


def test_load_config_merges_yaml_dangerous_ops_with_builtin_defaults(tmp_path):
    _write_config(
        tmp_path,
        """
        dangerous_ops:
          - "kubectl\\\\s+delete\\\\s+namespace"
        """,
    )

    config = load_config(tmp_path)

    assert "kubectl\\s+delete\\s+namespace" in config.dangerous_ops
    assert any("--force" in p for p in config.dangerous_ops)
    assert any("TRUNCATE" in p for p in config.dangerous_ops)


def test_sensitive_paths_do_not_duplicate_when_yaml_repeats_a_default(tmp_path):
    _write_config(
        tmp_path,
        """
        paths:
          sensitive:
            - "auth/**"
            - "payments/**"
        """,
    )

    config = load_config(tmp_path)

    assert config.sensitive_paths.count("auth/") == 1
    assert "payments/" in config.sensitive_paths
