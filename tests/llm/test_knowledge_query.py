"""Tests for LLM knowledge query functions."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from iffootball.agents.league import LeagueContext
from iffootball.llm.knowledge_query import (
    _DEFAULT_LEVEL,
    _DEFAULT_PREFERRED_FORMATION,
    _DEFAULT_PROMPT_PATH,
    _DEFAULT_STYLE_STUBBORNNESS,
    query_league_characteristics,
    query_manager_style,
)


# ---------------------------------------------------------------------------
# Fake LLMClient
# ---------------------------------------------------------------------------


class FakeLLMClient:
    """LLMClient stub that returns a fixed response string."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.last_messages: list[dict[str, str]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.last_messages = messages
        return self._response


_FAKE_PROMPT = "You are a football knowledge assistant."


def _json(**kwargs: object) -> str:
    return json.dumps(kwargs)


# ---------------------------------------------------------------------------
# TestQueryManagerStyle
# ---------------------------------------------------------------------------


class TestQueryManagerStyle:
    FORMATIONS = ["4-3-3", "4-2-3-1", "3-5-2", "other"]

    def test_high_stubbornness_maps_to_80(self) -> None:
        client = FakeLLMClient(_json(style_stubbornness="high", preferred_formation="4-2-3-1"))
        result = query_manager_style(client, "Mourinho", self.FORMATIONS, system_prompt=_FAKE_PROMPT)
        assert result.style_stubbornness == pytest.approx(80.0)

    def test_mid_stubbornness_maps_to_50(self) -> None:
        client = FakeLLMClient(_json(style_stubbornness="mid", preferred_formation="4-3-3"))
        result = query_manager_style(client, "Guardiola", self.FORMATIONS, system_prompt=_FAKE_PROMPT)
        assert result.style_stubbornness == pytest.approx(50.0)

    def test_low_stubbornness_maps_to_20(self) -> None:
        client = FakeLLMClient(_json(style_stubbornness="low", preferred_formation="other"))
        result = query_manager_style(client, "Pellegrini", self.FORMATIONS, system_prompt=_FAKE_PROMPT)
        assert result.style_stubbornness == pytest.approx(20.0)

    def test_valid_formation_in_options(self) -> None:
        client = FakeLLMClient(_json(style_stubbornness="high", preferred_formation="4-2-3-1"))
        result = query_manager_style(client, "Manager", self.FORMATIONS, system_prompt=_FAKE_PROMPT)
        assert result.preferred_formation == "4-2-3-1"

    def test_formation_not_in_options_returns_none(self) -> None:
        client = FakeLLMClient(_json(style_stubbornness="mid", preferred_formation="5-3-2"))
        result = query_manager_style(client, "Manager", self.FORMATIONS, system_prompt=_FAKE_PROMPT)
        assert result.preferred_formation is None

    def test_empty_formation_options_returns_none(self) -> None:
        client = FakeLLMClient(_json(style_stubbornness="mid", preferred_formation="4-3-3"))
        result = query_manager_style(client, "Manager", [], system_prompt=_FAKE_PROMPT)
        assert result.preferred_formation is None

    def test_null_preferred_formation_returns_none(self) -> None:
        client = FakeLLMClient(_json(style_stubbornness="high", preferred_formation=None))
        result = query_manager_style(client, "Manager", self.FORMATIONS, system_prompt=_FAKE_PROMPT)
        assert result.preferred_formation is None

    def test_invalid_json_returns_defaults(self) -> None:
        client = FakeLLMClient("not valid json")
        result = query_manager_style(client, "Manager", self.FORMATIONS, system_prompt=_FAKE_PROMPT)
        assert result.style_stubbornness == pytest.approx(_DEFAULT_STYLE_STUBBORNNESS)
        assert result.preferred_formation == _DEFAULT_PREFERRED_FORMATION

    def test_non_dict_json_returns_defaults(self) -> None:
        client = FakeLLMClient('["high", "4-3-3"]')
        result = query_manager_style(client, "Manager", self.FORMATIONS, system_prompt=_FAKE_PROMPT)
        assert result.style_stubbornness == pytest.approx(_DEFAULT_STYLE_STUBBORNNESS)
        assert result.preferred_formation == _DEFAULT_PREFERRED_FORMATION

    def test_invalid_stubbornness_returns_default(self) -> None:
        client = FakeLLMClient(_json(style_stubbornness="very_high", preferred_formation="4-3-3"))
        result = query_manager_style(client, "Manager", self.FORMATIONS, system_prompt=_FAKE_PROMPT)
        assert result.style_stubbornness == pytest.approx(_DEFAULT_STYLE_STUBBORNNESS)

    def test_missing_stubbornness_returns_default(self) -> None:
        client = FakeLLMClient(_json(preferred_formation="4-3-3"))
        result = query_manager_style(client, "Manager", self.FORMATIONS, system_prompt=_FAKE_PROMPT)
        assert result.style_stubbornness == pytest.approx(_DEFAULT_STYLE_STUBBORNNESS)

    def test_system_prompt_in_first_message(self) -> None:
        client = FakeLLMClient(_json(style_stubbornness="mid", preferred_formation="4-3-3"))
        query_manager_style(client, "Manager", self.FORMATIONS, system_prompt="custom_prompt")
        assert client.last_messages[0] == {"role": "system", "content": "custom_prompt"}

    def test_manager_name_in_user_message(self) -> None:
        client = FakeLLMClient(_json(style_stubbornness="mid", preferred_formation="4-3-3"))
        query_manager_style(client, "José Mourinho", self.FORMATIONS, system_prompt=_FAKE_PROMPT)
        user_payload = json.loads(client.last_messages[1]["content"])
        assert user_payload["manager_name"] == "José Mourinho"

    def test_formation_options_in_user_message(self) -> None:
        client = FakeLLMClient(_json(style_stubbornness="mid", preferred_formation="4-3-3"))
        query_manager_style(client, "Manager", self.FORMATIONS, system_prompt=_FAKE_PROMPT)
        user_payload = json.loads(client.last_messages[1]["content"])
        assert user_payload["formation_options"] == self.FORMATIONS

    def test_result_is_frozen(self) -> None:
        client = FakeLLMClient(_json(style_stubbornness="mid", preferred_formation="4-3-3"))
        result = query_manager_style(client, "Manager", self.FORMATIONS, system_prompt=_FAKE_PROMPT)
        with pytest.raises((AttributeError, TypeError)):
            result.style_stubbornness = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestQueryLeagueCharacteristics
# ---------------------------------------------------------------------------


class TestQueryLeagueCharacteristics:
    def test_valid_all_high(self) -> None:
        client = FakeLLMClient(_json(
            pressing_level="high",
            physicality_level="high",
            tactical_complexity="high",
        ))
        result = query_league_characteristics(client, "Premier League", system_prompt=_FAKE_PROMPT)
        assert result.pressing_level == "high"
        assert result.physicality_level == "high"
        assert result.tactical_complexity == "high"

    def test_valid_mixed_levels(self) -> None:
        client = FakeLLMClient(_json(
            pressing_level="high",
            physicality_level="mid",
            tactical_complexity="low",
        ))
        result = query_league_characteristics(client, "League", system_prompt=_FAKE_PROMPT)
        assert result.pressing_level == "high"
        assert result.physicality_level == "mid"
        assert result.tactical_complexity == "low"

    def test_all_valid_level_values(self) -> None:
        for level in ("high", "mid", "low"):
            client = FakeLLMClient(_json(
                pressing_level=level,
                physicality_level=level,
                tactical_complexity=level,
            ))
            result = query_league_characteristics(client, "League", system_prompt=_FAKE_PROMPT)
            assert result.pressing_level == level

    def test_invalid_json_returns_defaults(self) -> None:
        client = FakeLLMClient("not json")
        result = query_league_characteristics(client, "League", system_prompt=_FAKE_PROMPT)
        assert result.pressing_level == _DEFAULT_LEVEL
        assert result.physicality_level == _DEFAULT_LEVEL
        assert result.tactical_complexity == _DEFAULT_LEVEL

    def test_non_dict_json_returns_defaults(self) -> None:
        client = FakeLLMClient('["high", "low"]')
        result = query_league_characteristics(client, "League", system_prompt=_FAKE_PROMPT)
        assert result.pressing_level == _DEFAULT_LEVEL

    def test_invalid_level_returns_default_for_that_field(self) -> None:
        client = FakeLLMClient(_json(
            pressing_level="extreme",
            physicality_level="mid",
            tactical_complexity="low",
        ))
        result = query_league_characteristics(client, "League", system_prompt=_FAKE_PROMPT)
        assert result.pressing_level == _DEFAULT_LEVEL
        assert result.physicality_level == "mid"
        assert result.tactical_complexity == "low"

    def test_missing_field_returns_default(self) -> None:
        client = FakeLLMClient(_json(pressing_level="high"))
        result = query_league_characteristics(client, "League", system_prompt=_FAKE_PROMPT)
        assert result.physicality_level == _DEFAULT_LEVEL
        assert result.tactical_complexity == _DEFAULT_LEVEL

    def test_system_prompt_in_first_message(self) -> None:
        client = FakeLLMClient(_json(
            pressing_level="mid", physicality_level="mid", tactical_complexity="mid"
        ))
        query_league_characteristics(client, "League", system_prompt="custom")
        assert client.last_messages[0] == {"role": "system", "content": "custom"}

    def test_league_name_in_user_message(self) -> None:
        client = FakeLLMClient(_json(
            pressing_level="mid", physicality_level="mid", tactical_complexity="mid"
        ))
        query_league_characteristics(client, "La Liga", system_prompt=_FAKE_PROMPT)
        user_payload = json.loads(client.last_messages[1]["content"])
        assert user_payload["league_name"] == "La Liga"

    def test_result_is_frozen(self) -> None:
        client = FakeLLMClient(_json(
            pressing_level="mid", physicality_level="mid", tactical_complexity="mid"
        ))
        result = query_league_characteristics(client, "League", system_prompt=_FAKE_PROMPT)
        with pytest.raises((AttributeError, TypeError)):
            result.pressing_level = "high"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestPromptFile
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TestLeagueContext
# ---------------------------------------------------------------------------


class TestLeagueContext:
    def test_is_frozen(self) -> None:
        ctx = LeagueContext(competition_id=2, season_id=27, name="Premier League")
        with pytest.raises((AttributeError, TypeError)):
            ctx.pressing_level = "high"  # type: ignore[misc]

    def test_default_hypothesis_fields_are_none(self) -> None:
        ctx = LeagueContext(competition_id=2, season_id=27, name="Premier League")
        assert ctx.pressing_level is None
        assert ctx.physicality_level is None
        assert ctx.tactical_complexity is None

    def test_replace_updates_hypothesis_fields(self) -> None:
        ctx = LeagueContext(competition_id=2, season_id=27, name="Premier League")
        updated = replace(ctx, pressing_level="high", physicality_level="mid")
        assert updated.pressing_level == "high"
        assert updated.physicality_level == "mid"
        assert updated.tactical_complexity is None  # untouched

    def test_replace_does_not_mutate_original(self) -> None:
        ctx = LeagueContext(competition_id=2, season_id=27, name="Premier League")
        replace(ctx, pressing_level="high")
        assert ctx.pressing_level is None


# ---------------------------------------------------------------------------
# TestPromptFile
# ---------------------------------------------------------------------------


class TestPromptFile:
    def test_default_prompt_file_exists(self) -> None:
        assert _DEFAULT_PROMPT_PATH.exists(), (
            f"System prompt not found: {_DEFAULT_PROMPT_PATH}. "
            "Create prompts/knowledge_query_v1.md before running tests."
        )

    def test_missing_prompt_file_raises(self) -> None:
        from pathlib import Path

        from iffootball.llm.knowledge_query import _load_system_prompt

        with pytest.raises(FileNotFoundError):
            _load_system_prompt(Path("/nonexistent/path/prompt.md"))
