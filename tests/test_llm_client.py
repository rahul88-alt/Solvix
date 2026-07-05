from unittest.mock import patch

import pytest
import requests

from reasoning.llm_client import OllamaUnavailableError, complete, ensure_ollama_available


def test_ensure_ollama_available_passes_when_reachable():
    with patch("reasoning.llm_client.requests.get") as mock_get:
        mock_get.return_value = None
        ensure_ollama_available()
    mock_get.assert_called_once()


def test_ensure_ollama_available_raises_clear_error_when_unreachable():
    with patch(
        "reasoning.llm_client.requests.get",
        side_effect=requests.exceptions.ConnectionError(),
    ):
        with pytest.raises(OllamaUnavailableError) as excinfo:
            ensure_ollama_available()

    assert "ollama serve" in str(excinfo.value).lower()


def test_complete_raises_ollama_unavailable_error_on_connection_error_mid_run():
    """SLX-F4: complete() itself must not let a raw
    requests.exceptions.ConnectionError escape -- this is the case where
    Ollama goes down mid-run rather than never having started.
    """
    with patch(
        "reasoning.llm_client.requests.post",
        side_effect=requests.exceptions.ConnectionError(),
    ):
        with pytest.raises(OllamaUnavailableError):
            complete("system prompt", [{"role": "user", "content": "hi"}])
