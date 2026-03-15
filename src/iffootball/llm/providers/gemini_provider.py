"""Google Gemini LLMClient implementation."""

from __future__ import annotations

from typing import Any

from google import genai
from google.genai import types


class GeminiClient:
    """LLMClient backed by the Google Gemini API.

    Converts the chat message list to Gemini's content format,
    extracting system instruction from the system role message.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def complete(self, messages: list[dict[str, str]]) -> str:
        system_text = ""
        contents: list[Any] = []

        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part(text=msg["content"])],
                    )
                )

        config = types.GenerateContentConfig(
            system_instruction=system_text if system_text else None,
        )

        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )
        return response.text or ""
