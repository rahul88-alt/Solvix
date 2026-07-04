"""Thin wrapper around the local Ollama OpenAI-compatible endpoint
(Master Document 7.2/7.3). No cloud API, no API key.

Kept deliberately minimal for now: a single complete() call, no tool use.
Swappable behind this function if the underlying LLM provider ever changes.
"""

from __future__ import annotations

import requests

_OLLAMA_BASE_URL = "http://localhost:11434/v1"
_MODEL = "qwen2.5-coder:14b"


def complete(system: str, messages: list[dict], max_tokens: int = 4096) -> str:
    """Send a system prompt + message history to the local Ollama model and
    return the reply text.
    """
    response = requests.post(
        f"{_OLLAMA_BASE_URL}/chat/completions",
        json={
            "model": _MODEL,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, *messages],
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
