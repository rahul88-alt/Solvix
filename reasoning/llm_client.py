"""Thin wrapper around the local Ollama OpenAI-compatible endpoint
(Master Document 7.2/7.3). No cloud API, no API key.

Kept deliberately minimal for now: a single complete() call, no tool use.
Swappable behind this function if the underlying LLM provider ever changes.

SLX-F4: ensure_ollama_available() mirrors execution.sandbox.
ensure_docker_available()'s pattern -- a cheap preflight connectivity check
callers should run at process startup, before any real pipeline work, so an
unreachable Ollama endpoint fails fast with a clear message rather than as
a raw requests.exceptions.ConnectionError surfacing from deep inside
check_ambiguity/generate_plan/propose_diff. complete() itself also catches
ConnectionError (for the case Ollama goes down mid-run rather than never
having started), raising the same OllamaUnavailableError so callers have a
single exception type to handle either way.
"""

from __future__ import annotations

import requests

_OLLAMA_HOST = "http://localhost:11434"
_OLLAMA_BASE_URL = f"{_OLLAMA_HOST}/v1"
_MODEL = "qwen2.5-coder:14b"


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama endpoint is not reachable, whether at
    startup (ensure_ollama_available) or mid-run (complete)."""


def ensure_ollama_available(timeout: float = 5.0) -> None:
    """Raise OllamaUnavailableError with a clear, actionable message if the
    local Ollama endpoint isn't reachable, rather than letting the first
    real LLM call fail deep inside the pipeline with a raw
    requests.exceptions.ConnectionError.
    """
    try:
        requests.get(f"{_OLLAMA_HOST}/api/tags", timeout=timeout)
    except requests.exceptions.ConnectionError as error:
        raise OllamaUnavailableError(
            f"Ollama is not available at {_OLLAMA_HOST}. Start it with: ollama serve"
        ) from error


def complete(system: str, messages: list[dict], max_tokens: int = 4096) -> str:
    """Send a system prompt + message history to the local Ollama model and
    return the reply text.
    """
    try:
        response = requests.post(
            f"{_OLLAMA_BASE_URL}/chat/completions",
            json={
                "model": _MODEL,
                "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, *messages],
            },
            timeout=120,
        )
    except requests.exceptions.ConnectionError as error:
        raise OllamaUnavailableError(
            f"lost connection to Ollama at {_OLLAMA_BASE_URL}. Start it with: ollama serve"
        ) from error
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
