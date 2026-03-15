"""Anthropic LLMClient implementation."""

from __future__ import annotations

from anthropic import Anthropic


class AnthropicClient:
    """LLMClient backed by the Anthropic API.

    The Anthropic API requires a separate system parameter rather than
    a system role in the messages list. This adapter extracts the first
    system message and passes it via the system kwarg.
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514") -> None:
        self._client = Anthropic(api_key=api_key)
        self._model = model

    def complete(self, messages: list[dict[str, str]]) -> str:
        system_text = ""
        user_messages: list[dict[str, str]] = []

        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                user_messages.append(msg)

        response = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=system_text,
            messages=user_messages,  # type: ignore[arg-type]
        )
        block = response.content[0]
        return block.text if hasattr(block, "text") else ""
