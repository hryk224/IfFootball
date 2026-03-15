"""Groq LLMClient implementation."""

from __future__ import annotations

from groq import Groq


class GroqClient:
    """LLMClient backed by the Groq API.

    Groq uses an OpenAI-compatible API, but with its own endpoint
    and API key. Separated from OpenAIClient for clarity of
    configuration and default model selection.
    """

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile") -> None:
        self._client = Groq(api_key=api_key)
        self._model = model

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
        )
        return response.choices[0].message.content or ""
