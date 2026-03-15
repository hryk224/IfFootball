"""Tests for LLM action explanation."""

from __future__ import annotations

import json

from iffootball.agents.manager import ManagerAgent
from iffootball.agents.player import BroadPosition, PlayerAgent, RoleFamily
from iffootball.llm.action_explanation import (
    ActionExplanationResult,
    _DEFAULT_EXPLANATION,
    _DEFAULT_LABEL,
    _build_user_payload,
    _parse_response,
    explain_action,
)
from iffootball.simulation.turning_point import ActionDistribution, SimContext


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


_FAKE_PROMPT = "You are a football simulation analyst."


def _json(**kwargs: object) -> str:
    return json.dumps(kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_player() -> PlayerAgent:
    return PlayerAgent(
        player_id=7,
        player_name="Test Player",
        position_name="Right Wing",
        role_family=RoleFamily.WINGER,
        broad_position=BroadPosition.MF,
        pace=70.0,
        passing=60.0,
        shooting=55.0,
        pressing=50.0,
        defending=40.0,
        physicality=50.0,
        consistency=50.0,
        current_form=0.35,
        fatigue=0.20,
        tactical_understanding=0.22,
        manager_trust=0.38,
        bench_streak=3,
    )


def _make_context() -> SimContext:
    manager = ManagerAgent(
        manager_name="New Manager",
        team_name="Team A",
        competition_id=1,
        season_id=1,
        tenure_match_ids=frozenset(),
        pressing_intensity=55.0,
        possession_preference=0.55,
        counter_tendency=0.45,
        preferred_formation="4-3-3",
    )
    return SimContext(
        current_week=14,
        matches_since_appointment=3,
        manager=manager,
        recent_points=(0, 1, 3, 0, 1),
    )


def _make_distribution() -> ActionDistribution:
    return ActionDistribution({"adapt": 0.3, "resist": 0.6, "transfer": 0.1})


# ---------------------------------------------------------------------------
# Tests: _parse_response
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_valid_response(self) -> None:
        raw = _json(
            explanation="Player resisted due to low trust.",
            label="analysis",
            confidence_note="Based on rule-based model",
        )
        result = _parse_response(raw)
        assert result.explanation == "Player resisted due to low trust."
        assert result.label == "analysis"
        assert result.confidence_note == "Based on rule-based model"

    def test_missing_confidence_note(self) -> None:
        raw = _json(explanation="Adapted well.", label="data")
        result = _parse_response(raw)
        assert result.explanation == "Adapted well."
        assert result.label == "data"
        assert result.confidence_note == ""

    def test_invalid_label_falls_back(self) -> None:
        raw = _json(explanation="Some text.", label="unknown")
        result = _parse_response(raw)
        assert result.label == _DEFAULT_LABEL

    def test_empty_explanation_falls_back(self) -> None:
        raw = _json(explanation="", label="analysis")
        result = _parse_response(raw)
        assert result.explanation == _DEFAULT_EXPLANATION

    def test_missing_explanation_falls_back(self) -> None:
        raw = _json(label="analysis")
        result = _parse_response(raw)
        assert result.explanation == _DEFAULT_EXPLANATION

    def test_invalid_json_falls_back(self) -> None:
        result = _parse_response("not json at all")
        assert result.explanation == _DEFAULT_EXPLANATION
        assert result.label == _DEFAULT_LABEL

    def test_non_dict_json_falls_back(self) -> None:
        result = _parse_response("[1, 2, 3]")
        assert result.explanation == _DEFAULT_EXPLANATION

    def test_all_valid_labels_accepted(self) -> None:
        for label in ("data", "analysis", "hypothesis"):
            raw = _json(explanation="Test.", label=label)
            result = _parse_response(raw)
            assert result.label == label

    def test_whitespace_trimmed(self) -> None:
        raw = _json(
            explanation="  Some explanation.  ",
            label="analysis",
            confidence_note="  note  ",
        )
        result = _parse_response(raw)
        assert result.explanation == "Some explanation."
        assert result.confidence_note == "note"


# ---------------------------------------------------------------------------
# Tests: _build_user_payload
# ---------------------------------------------------------------------------


class TestBuildUserPayload:
    def test_contains_required_keys(self) -> None:
        payload = _build_user_payload(
            _make_player(),
            _make_context(),
            "resist",
            _make_distribution(),
            ["bench_streak"],
        )
        assert "player" in payload
        assert "turning_points" in payload
        assert "sampled_action" in payload
        assert "action_distribution" in payload
        assert "context" in payload
        assert "source_types" in payload

    def test_player_fields(self) -> None:
        payload = _build_user_payload(
            _make_player(),
            _make_context(),
            "resist",
            _make_distribution(),
            ["bench_streak"],
        )
        p = payload["player"]
        assert p["name"] == "Test Player"
        assert p["position"] == "Right Wing"
        assert p["bench_streak"] == 3
        assert 0.0 <= p["current_form"] <= 1.0

    def test_source_types_present(self) -> None:
        payload = _build_user_payload(
            _make_player(),
            _make_context(),
            "resist",
            _make_distribution(),
            ["bench_streak"],
        )
        st = payload["source_types"]
        assert st["action_distribution"] == "rule_based_model"
        assert st["form_fatigue_trust"] == "simulation_output"

    def test_serializable_to_json(self) -> None:
        payload = _build_user_payload(
            _make_player(),
            _make_context(),
            "resist",
            _make_distribution(),
            ["bench_streak"],
        )
        # Must not raise.
        serialized = json.dumps(payload, ensure_ascii=False)
        assert isinstance(serialized, str)


# ---------------------------------------------------------------------------
# Tests: explain_action (integration with FakeLLMClient)
# ---------------------------------------------------------------------------


class TestExplainAction:
    def test_returns_result(self) -> None:
        response = _json(
            explanation="Low trust after 3 benchings caused resistance.",
            label="analysis",
            confidence_note="",
        )
        client = FakeLLMClient(response)
        result = explain_action(
            client,
            _make_player(),
            _make_context(),
            "resist",
            _make_distribution(),
            ["bench_streak"],
            system_prompt=_FAKE_PROMPT,
        )
        assert isinstance(result, ActionExplanationResult)
        assert result.label == "analysis"
        assert "resistance" in result.explanation.lower()

    def test_system_user_role_separation(self) -> None:
        client = FakeLLMClient(_json(explanation="Ok.", label="data"))
        explain_action(
            client,
            _make_player(),
            _make_context(),
            "adapt",
            _make_distribution(),
            ["low_understanding"],
            system_prompt=_FAKE_PROMPT,
        )
        messages = client.last_messages
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_player_name_not_in_system_prompt(self) -> None:
        client = FakeLLMClient(_json(explanation="Ok.", label="data"))
        explain_action(
            client,
            _make_player(),
            _make_context(),
            "adapt",
            _make_distribution(),
            ["low_understanding"],
            system_prompt=_FAKE_PROMPT,
        )
        system_content = client.last_messages[0]["content"]
        assert "Test Player" not in system_content

    def test_player_name_in_user_message(self) -> None:
        client = FakeLLMClient(_json(explanation="Ok.", label="data"))
        explain_action(
            client,
            _make_player(),
            _make_context(),
            "adapt",
            _make_distribution(),
            ["low_understanding"],
            system_prompt=_FAKE_PROMPT,
        )
        user_content = client.last_messages[1]["content"]
        assert "Test Player" in user_content

    def test_fallback_on_bad_response(self) -> None:
        client = FakeLLMClient("not valid json")
        result = explain_action(
            client,
            _make_player(),
            _make_context(),
            "resist",
            _make_distribution(),
            ["bench_streak"],
            system_prompt=_FAKE_PROMPT,
        )
        assert result.explanation == _DEFAULT_EXPLANATION
        assert result.label == _DEFAULT_LABEL

    def test_loads_default_prompt_file(self) -> None:
        """Verify the default prompt file exists and is loadable."""
        from iffootball.llm.action_explanation import _load_system_prompt

        prompt = _load_system_prompt()
        assert len(prompt) > 0
        assert "action" in prompt.lower()
