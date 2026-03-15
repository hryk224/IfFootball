"""OpenAI LLMClient implementation.

Supports custom base_url for OpenAI-compatible APIs (e.g., Azure OpenAI,
local inference servers, third-party providers).
"""

from __future__ import annotations

from openai import OpenAI


class OpenAIClient:
    """LLMClient backed by the OpenAI API.

    Args:
        api_key:  OpenAI API key.
        model:    Model name (e.g., "gpt-4o-mini").
        base_url: Optional base URL for OpenAI-compatible endpoints.
                  When None, uses the default OpenAI API endpoint.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
    ) -> None:
        kwargs: dict[str, str] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)  # type: ignore[arg-type]
        self._model = model

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
        )
        return response.choices[0].message.content or ""
