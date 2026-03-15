"""Tests for natural language input structuring."""

from __future__ import annotations

import json

from iffootball.llm.input_structuring import (
    StructuredInput,
    _parse_response,
    structure_input,
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


_FAKE_PROMPT = "You are a football simulation input parser."


def _json(**kwargs: object) -> str:
    return json.dumps(kwargs)


# ---------------------------------------------------------------------------
# Tests: _parse_response — manager_change
# ---------------------------------------------------------------------------


class TestParseManagerChange:
    def test_valid_full_response(self) -> None:
        raw = _json(
            trigger_type="manager_change",
            team_name="Manchester United",
            outgoing_manager_name="Erik ten Hag",
            incoming_manager_name="José Mourinho",
            transition_type="mid_season",
            applied_at=10,
        )
        result = _parse_response(raw)
        assert result.parse_success
        assert result.trigger_type == "manager_change"
        assert result.team_name == "Manchester United"
        assert result.params["incoming_manager_name"] == "José Mourinho"
        assert result.params["outgoing_manager_name"] == "Erik ten Hag"
        assert result.params["transition_type"] == "mid_season"
        assert result.params["applied_at"] == 10

    def test_null_outgoing_allowed(self) -> None:
        raw = _json(
            trigger_type="manager_change",
            team_name="Chelsea",
            outgoing_manager_name=None,
            incoming_manager_name="New Manager",
            transition_type="mid_season",
            applied_at=None,
        )
        result = _parse_response(raw)
        assert result.parse_success
        assert result.params["outgoing_manager_name"] is None
        assert result.params["applied_at"] is None

    def test_null_transition_type_defaults_to_mid_season(self) -> None:
        raw = _json(
            trigger_type="manager_change",
            team_name="Chelsea",
            outgoing_manager_name=None,
            incoming_manager_name="New Manager",
            transition_type=None,
            applied_at=10,
        )
        result = _parse_response(raw)
        assert result.parse_success
        assert result.params["transition_type"] == "mid_season"

    def test_missing_incoming_name_fails(self) -> None:
        raw = _json(
            trigger_type="manager_change",
            team_name="Arsenal",
            outgoing_manager_name="Old Manager",
            transition_type="mid_season",
            applied_at=5,
        )
        result = _parse_response(raw)
        assert not result.parse_success
        assert "incoming_manager_name" in result.error_message

    def test_invalid_transition_type_fails(self) -> None:
        raw = _json(
            trigger_type="manager_change",
            team_name="Liverpool",
            outgoing_manager_name="Old",
            incoming_manager_name="New",
            transition_type="post_season",
            applied_at=10,
        )
        result = _parse_response(raw)
        assert not result.parse_success
        assert "transition_type" in result.error_message

    def test_pre_season_transition_accepted(self) -> None:
        raw = _json(
            trigger_type="manager_change",
            team_name="Spurs",
            outgoing_manager_name=None,
            incoming_manager_name="New Coach",
            transition_type="pre_season",
            applied_at=0,
        )
        result = _parse_response(raw)
        assert result.parse_success
        assert result.params["transition_type"] == "pre_season"


# ---------------------------------------------------------------------------
# Tests: _parse_response — player_transfer_in
# ---------------------------------------------------------------------------


class TestParseTransferIn:
    def test_valid_full_response(self) -> None:
        raw = _json(
            trigger_type="player_transfer_in",
            team_name="Manchester City",
            player_name="Kylian Mbappé",
            expected_role="starter",
            applied_at=15,
        )
        result = _parse_response(raw)
        assert result.parse_success
        assert result.trigger_type == "player_transfer_in"
        assert result.team_name == "Manchester City"
        assert result.params["player_name"] == "Kylian Mbappé"
        assert result.params["expected_role"] == "starter"

    def test_rotation_role_accepted(self) -> None:
        raw = _json(
            trigger_type="player_transfer_in",
            team_name="Barcelona",
            player_name="Some Player",
            expected_role="rotation",
            applied_at=None,
        )
        result = _parse_response(raw)
        assert result.parse_success
        assert result.params["expected_role"] == "rotation"

    def test_null_role_defaults_to_starter(self) -> None:
        raw = _json(
            trigger_type="player_transfer_in",
            team_name="PSG",
            player_name="Some Player",
            expected_role=None,
            applied_at=5,
        )
        result = _parse_response(raw)
        assert result.parse_success
        assert result.params["expected_role"] == "starter"

    def test_missing_player_name_fails(self) -> None:
        raw = _json(
            trigger_type="player_transfer_in",
            team_name="Real Madrid",
            expected_role="starter",
            applied_at=10,
        )
        result = _parse_response(raw)
        assert not result.parse_success
        assert "player_name" in result.error_message

    def test_invalid_role_fails(self) -> None:
        raw = _json(
            trigger_type="player_transfer_in",
            team_name="Bayern",
            player_name="Player X",
            expected_role="bench_warmer",
            applied_at=5,
        )
        result = _parse_response(raw)
        assert not result.parse_success
        assert "expected_role" in result.error_message


# ---------------------------------------------------------------------------
# Tests: _parse_response — error cases
# ---------------------------------------------------------------------------


class TestParseErrors:
    def test_invalid_json(self) -> None:
        result = _parse_response("not json")
        assert not result.parse_success
        assert "JSON" in result.error_message

    def test_non_dict_json(self) -> None:
        result = _parse_response("[1, 2, 3]")
        assert not result.parse_success

    def test_null_trigger_type(self) -> None:
        raw = _json(trigger_type=None, error="Not a valid scenario.")
        result = _parse_response(raw)
        assert not result.parse_success
        assert "Not a valid scenario" in result.error_message

    def test_unknown_trigger_type(self) -> None:
        raw = _json(trigger_type="formation_change", team_name="Team")
        result = _parse_response(raw)
        assert not result.parse_success
        assert "Unknown trigger_type" in result.error_message

    def test_missing_team_name(self) -> None:
        raw = _json(
            trigger_type="manager_change",
            incoming_manager_name="New Manager",
            transition_type="mid_season",
            applied_at=10,
        )
        result = _parse_response(raw)
        assert not result.parse_success
        assert "team_name" in result.error_message

    def test_applied_at_string_fails(self) -> None:
        raw = _json(
            trigger_type="manager_change",
            team_name="Arsenal",
            outgoing_manager_name="Old",
            incoming_manager_name="New",
            transition_type="mid_season",
            applied_at="10",
        )
        result = _parse_response(raw)
        assert not result.parse_success
        assert "applied_at" in result.error_message

    def test_applied_at_float_fails(self) -> None:
        raw = _json(
            trigger_type="manager_change",
            team_name="Arsenal",
            outgoing_manager_name="Old",
            incoming_manager_name="New",
            transition_type="mid_season",
            applied_at=10.5,
        )
        result = _parse_response(raw)
        assert not result.parse_success
        assert "applied_at" in result.error_message

    def test_empty_team_name(self) -> None:
        raw = _json(
            trigger_type="manager_change",
            team_name="",
            incoming_manager_name="New Manager",
            transition_type="mid_season",
            applied_at=10,
        )
        result = _parse_response(raw)
        assert not result.parse_success
        assert "team_name" in result.error_message


# ---------------------------------------------------------------------------
# Tests: structure_input (integration with FakeLLMClient)
# ---------------------------------------------------------------------------


class TestStructureInput:
    def test_returns_structured_input(self) -> None:
        response = _json(
            trigger_type="manager_change",
            team_name="Chelsea",
            outgoing_manager_name="Current",
            incoming_manager_name="Incoming",
            transition_type="mid_season",
            applied_at=12,
        )
        client = FakeLLMClient(response)
        result = structure_input(client, "What if Chelsea hired Incoming?",
                                 system_prompt=_FAKE_PROMPT)
        assert isinstance(result, StructuredInput)
        assert result.parse_success
        assert result.team_name == "Chelsea"

    def test_system_user_role_separation(self) -> None:
        response = _json(
            trigger_type="manager_change",
            team_name="T",
            outgoing_manager_name=None,
            incoming_manager_name="M",
            transition_type="mid_season",
            applied_at=1,
        )
        client = FakeLLMClient(response)
        structure_input(client, "test input", system_prompt=_FAKE_PROMPT)
        messages = client.last_messages
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_user_text_in_user_message_only(self) -> None:
        response = _json(
            trigger_type="manager_change",
            team_name="T",
            outgoing_manager_name=None,
            incoming_manager_name="M",
            transition_type="mid_season",
            applied_at=1,
        )
        client = FakeLLMClient(response)
        structure_input(client, "Mourinho to United",
                        system_prompt=_FAKE_PROMPT)
        assert "Mourinho" not in client.last_messages[0]["content"]
        assert "Mourinho" in client.last_messages[1]["content"]

    def test_bad_llm_response_returns_failure(self) -> None:
        client = FakeLLMClient("garbage output")
        result = structure_input(client, "test", system_prompt=_FAKE_PROMPT)
        assert not result.parse_success
        assert result.error_message != ""

    def test_loads_default_prompt_file(self) -> None:
        from iffootball.llm.input_structuring import _load_system_prompt

        prompt = _load_system_prompt()
        assert len(prompt) > 0
        assert "trigger" in prompt.lower()
