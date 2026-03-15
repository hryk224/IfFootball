"""LLM provider factory.

Resolves a concrete LLMClient implementation based on environment
configuration. Provider selection priority:

  1. LLM_PROVIDER env var (explicit override): "openai" / "anthropic" /
     "gemini" / "groq"
  2. Auto-detect from available API keys:
     OPENAI_API_KEY → OpenAI
     ANTHROPIC_API_KEY → Anthropic
     GOOGLE_API_KEY → Gemini
     GROQ_API_KEY → Groq
  3. None if no provider is configured

Usage:
    from iffootball.llm.providers import create_client

    client = create_client()  # Returns LLMClient or None
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from iffootball.llm.client import LLMClient

# Load .env file so API keys are available from os.environ.
load_dotenv()

# Provider name → (env key for API key, module import path).
_PROVIDERS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
}

# Auto-detect order when LLM_PROVIDER is not set.
_AUTO_DETECT_ORDER = ("openai", "anthropic", "gemini", "groq")

# Provider-specific model env vars (checked before common LLM_MODEL).
_PROVIDER_MODEL_ENVS: dict[str, str] = {
    "openai": "OPENAI_MODEL",
}


def create_client(
    *,
    provider: str | None = None,
    model: str | None = None,
) -> LLMClient | None:
    """Create an LLMClient from environment configuration.

    Model resolution priority:
        1. Explicit `model` argument
        2. Provider-specific env (e.g., OPENAI_MODEL)
        3. Common LLM_MODEL env
        4. Provider default

    Args:
        provider: Explicit provider name. If None, uses LLM_PROVIDER env
                  var, then auto-detects from available API keys.
        model:    Model name override. If None, resolved from env vars.

    Returns:
        A configured LLMClient, or None if no provider is available.
    """
    resolved_provider = _resolve_provider(provider)
    if resolved_provider is None:
        return None

    api_key_env = _PROVIDERS[resolved_provider]
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        return None

    # Model resolution: arg > provider-specific env > common env > default.
    provider_model_env = _PROVIDER_MODEL_ENVS.get(resolved_provider, "")
    resolved_model = (
        model
        or (os.environ.get(provider_model_env, "") if provider_model_env else "")
        or os.environ.get("LLM_MODEL", "")
    )

    if resolved_provider == "openai":
        base_url = os.environ.get("OPENAI_BASE_URL", "")
        return _create_openai(api_key, resolved_model, base_url or None)
    elif resolved_provider == "anthropic":
        return _create_anthropic(api_key, resolved_model)
    elif resolved_provider == "gemini":
        return _create_gemini(api_key, resolved_model)
    elif resolved_provider == "groq":
        return _create_groq(api_key, resolved_model)

    return None


def available_providers() -> list[str]:
    """Return names of providers with API keys configured and SDK importable."""
    result: list[str] = []
    for name in _AUTO_DETECT_ORDER:
        if not os.environ.get(_PROVIDERS[name], ""):
            continue
        # Verify the SDK is actually importable.
        if _can_import(name):
            result.append(name)
    return result


def _can_import(provider: str) -> bool:
    """Check if the provider's SDK is importable."""
    try:
        if provider == "openai":
            import openai  # noqa: F401
        elif provider == "anthropic":
            import anthropic  # noqa: F401
        elif provider == "gemini":
            import google.genai  # noqa: F401
        elif provider == "groq":
            import groq  # noqa: F401
        else:
            return False
        return True
    except ImportError:
        return False


def _resolve_provider(explicit: str | None) -> str | None:
    """Resolve provider name from explicit arg, env var, or auto-detect.

    Only returns a provider that has both an API key and an importable SDK.
    """
    # 1. Explicit argument.
    if explicit and explicit in _PROVIDERS:
        if os.environ.get(_PROVIDERS[explicit], "") and _can_import(explicit):
            return explicit
        return None

    # 2. LLM_PROVIDER env var.
    env_provider = os.environ.get("LLM_PROVIDER", "").lower()
    if env_provider in _PROVIDERS:
        if os.environ.get(_PROVIDERS[env_provider], "") and _can_import(env_provider):
            return env_provider
        return None

    # 3. Auto-detect: first provider with API key AND importable SDK.
    for name in _AUTO_DETECT_ORDER:
        if os.environ.get(_PROVIDERS[name], "") and _can_import(name):
            return name

    return None


# ---------------------------------------------------------------------------
# Provider constructors
# ---------------------------------------------------------------------------


def _create_openai(
    api_key: str, model: str, base_url: str | None = None
) -> LLMClient | None:
    try:
        from iffootball.llm.providers.openai_provider import OpenAIClient

        return OpenAIClient(
            api_key=api_key, model=model or "gpt-4o-mini", base_url=base_url
        )
    except ImportError:
        return None


def _create_anthropic(api_key: str, model: str) -> LLMClient | None:
    try:
        from iffootball.llm.providers.anthropic_provider import AnthropicClient

        return AnthropicClient(api_key=api_key, model=model or "claude-sonnet-4-20250514")
    except ImportError:
        return None


def _create_gemini(api_key: str, model: str) -> LLMClient | None:
    try:
        from iffootball.llm.providers.gemini_provider import GeminiClient

        return GeminiClient(api_key=api_key, model=model or "gemini-2.0-flash")
    except ImportError:
        return None


def _create_groq(api_key: str, model: str) -> LLMClient | None:
    try:
        from iffootball.llm.providers.groq_provider import GroqClient

        return GroqClient(api_key=api_key, model=model or "llama-3.3-70b-versatile")
    except ImportError:
        return None
