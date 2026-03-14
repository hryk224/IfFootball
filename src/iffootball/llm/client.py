"""LLMClient Protocol for provider-swappable LLM access.

Any LLM backend (OpenAI, Anthropic, local model, ...) can be plugged in
by implementing this single-method Protocol.
"""

from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    """Provider-agnostic LLM chat-completion interface.

    Implement this Protocol to plug in any chat-completion backend.
    The knowledge-query layer depends only on this interface, not on
    any specific provider SDK.

    Example implementation (OpenAI):

        class OpenAIClient:
            def __init__(self, model: str) -> None:
                import openai
                self._client = openai.OpenAI()
                self._model = model

            def complete(self, messages: list[dict[str, str]]) -> str:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,  # type: ignore[arg-type]
                )
                return resp.choices[0].message.content or ""
    """

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Send chat messages and return the assistant's response text.

        Args:
            messages: List of role-content dicts, e.g.
                      [{"role": "system", "content": "..."},
                       {"role": "user",   "content": "..."}].
                      Valid roles: "system", "user", "assistant".

        Returns:
            The assistant's response as a plain string.
        """
        ...
