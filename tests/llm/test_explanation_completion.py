"""Tests for LLM explanation completion."""

from __future__ import annotations

import json

from iffootball.llm.explanation_completion import (
    _merge_response,
    _parse_llm_response,
    _skeleton_to_json,
    complete_skeleton,
)
from iffootball.simulation.structured_explanation import (
    CausalStep,
    DifferenceHighlight,
    EvidenceItem,
    PlayerImpactChange,
    PlayerImpactSummary,
    ScenarioDescriptor,
    StructuredExplanation,
)


# ---------------------------------------------------------------------------
# Fake LLMClient
# ---------------------------------------------------------------------------


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self._response = response
        self.last_messages: list[dict[str, str]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.last_messages = messages
        return self._response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_skeleton() -> StructuredExplanation:
    return StructuredExplanation(
        scenario=ScenarioDescriptor(
            trigger_type="manager_change",
            team_name="Arsenal",
            detail={
                "outgoing_manager": "Arteta",
                "incoming_manager": "Xabi Alonso",
            },
        ),
        highlights=(
            DifferenceHighlight(
                metric_name="total_points_mean",
                value_a=50.0,
                value_b=45.0,
                diff=-5.0,
                interpretations=(
                    EvidenceItem(
                        statement="",
                        label="data",
                        source="simulation_output",
                    ),
                ),
            ),
        ),
        causal_chain=(
            CausalStep(
                step_id="cs-001",
                cause="",
                effect="",
                affected_agent="Player A",
                event_type="form_drop",
                evidence=(
                    EvidenceItem(
                        statement="",
                        label="data",
                        source="simulation_output",
                    ),
                ),
                depth=1,
            ),
        ),
        player_impacts=(
            PlayerImpactSummary(
                player_name="Player A",
                impact_score=0.3,
                changes=(
                    PlayerImpactChange(
                        axis="form",
                        diff=-0.15,
                        interpretation=EvidenceItem(
                            statement="",
                            label="data",
                            source="simulation_output",
                        ),
                    ),
                ),
                related_step_ids=("cs-001",),
            ),
        ),
        confidence_notes=("Chain reaches depth 3.",),
    )


def _make_filled_response(skeleton: StructuredExplanation) -> str:
    """Create a valid LLM response with statements filled."""
    data = json.loads(_skeleton_to_json(skeleton))
    # Fill statements.
    data["highlights"][0]["interpretations"][0]["statement"] = "Points decreased by 5."
    data["causal_chain"][0]["cause"] = "Manager change disrupted tactics"
    data["causal_chain"][0]["effect"] = "Player form dropped"
    data["causal_chain"][0]["evidence"][0]["statement"] = "Form decreased significantly"
    data["player_impacts"][0]["changes"][0]["interpretation"]["statement"] = (
        "Form declined due to tactical confusion"
    )
    data["confidence_notes"][0] = "Causal chain reaches depth 3."
    return json.dumps(data, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tests: _parse_llm_response
# ---------------------------------------------------------------------------


class TestParseLlmResponse:
    def test_valid_json(self) -> None:
        result = _parse_llm_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_code_fence_stripped(self) -> None:
        result = _parse_llm_response('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_invalid_json_returns_none(self) -> None:
        assert _parse_llm_response("not json") is None

    def test_non_dict_returns_none(self) -> None:
        assert _parse_llm_response("[1, 2, 3]") is None


# ---------------------------------------------------------------------------
# Tests: _skeleton_to_json
# ---------------------------------------------------------------------------


class TestSkeletonToJson:
    def test_roundtrip_parseable(self) -> None:
        skeleton = _make_skeleton()
        serialized = _skeleton_to_json(skeleton)
        parsed = json.loads(serialized)
        assert parsed["scenario"]["trigger_type"] == "manager_change"
        assert parsed["causal_chain"][0]["step_id"] == "cs-001"

    def test_empty_statements_present(self) -> None:
        skeleton = _make_skeleton()
        serialized = _skeleton_to_json(skeleton)
        parsed = json.loads(serialized)
        assert parsed["causal_chain"][0]["cause"] == ""
        assert parsed["causal_chain"][0]["effect"] == ""


# ---------------------------------------------------------------------------
# Tests: _merge_response
# ---------------------------------------------------------------------------


class TestMergeResponse:
    def test_statements_filled(self) -> None:
        skeleton = _make_skeleton()
        filled_json = _make_filled_response(skeleton)
        filled_data = json.loads(filled_json)
        result = _merge_response(skeleton, filled_data)

        assert result.highlights[0].interpretations[0].statement == "Points decreased by 5."
        assert result.causal_chain[0].cause == "Manager change disrupted tactics"
        assert result.causal_chain[0].effect == "Player form dropped"

    def test_structural_fields_preserved(self) -> None:
        skeleton = _make_skeleton()
        filled_json = _make_filled_response(skeleton)
        filled_data = json.loads(filled_json)

        # Tamper with structural fields — label/source should come from skeleton.
        filled_data["causal_chain"][0]["evidence"][0]["label"] = "hypothesis"
        filled_data["causal_chain"][0]["evidence"][0]["source"] = "llm_knowledge"

        result = _merge_response(skeleton, filled_data)

        # Structural fields come from skeleton, not LLM.
        assert result.causal_chain[0].step_id == "cs-001"
        assert result.causal_chain[0].evidence[0].label == "data"
        assert result.causal_chain[0].evidence[0].source == "simulation_output"

    def test_causal_chain_merge_by_step_id(self) -> None:
        """LLM reorders causal chain — merge still matches by step_id."""
        skeleton = _make_skeleton()
        filled_data = {
            "causal_chain": [
                {
                    "step_id": "cs-001",
                    "cause": "Correct cause for cs-001",
                    "effect": "Correct effect for cs-001",
                    "evidence": [{"statement": "Evidence for cs-001"}],
                }
            ]
        }
        result = _merge_response(skeleton, filled_data)
        assert result.causal_chain[0].cause == "Correct cause for cs-001"

    def test_causal_chain_tampered_step_id_not_merged(self) -> None:
        """LLM tampers step_id — that entry is not matched, skeleton unchanged."""
        skeleton = _make_skeleton()
        filled_data = {
            "causal_chain": [
                {
                    "step_id": "TAMPERED",
                    "cause": "Should not appear",
                    "effect": "Should not appear",
                }
            ]
        }
        result = _merge_response(skeleton, filled_data)
        assert result.causal_chain[0].cause == ""  # skeleton default

    def test_highlights_merge_by_metric_name(self) -> None:
        """LLM reorders highlights — merge by metric_name."""
        skeleton = _make_skeleton()
        filled_data = {
            "highlights": [
                {
                    "metric_name": "total_points_mean",
                    "interpretations": [
                        {"statement": "Points decreased by 5."}
                    ],
                }
            ]
        }
        result = _merge_response(skeleton, filled_data)
        assert (
            result.highlights[0].interpretations[0].statement
            == "Points decreased by 5."
        )

    def test_highlights_tampered_metric_not_merged(self) -> None:
        """LLM returns unknown metric_name — skeleton unchanged."""
        skeleton = _make_skeleton()
        filled_data = {
            "highlights": [
                {
                    "metric_name": "UNKNOWN",
                    "interpretations": [
                        {"statement": "Should not appear"}
                    ],
                }
            ]
        }
        result = _merge_response(skeleton, filled_data)
        assert result.highlights[0].interpretations[0].statement == ""

    def test_player_impact_merge_by_name(self) -> None:
        """LLM reorders player impacts — merge by player_name."""
        skeleton = _make_skeleton()
        filled_data = {
            "player_impacts": [
                {
                    "player_name": "Player A",
                    "changes": [
                        {
                            "interpretation": {
                                "statement": "Correct interpretation"
                            }
                        }
                    ],
                }
            ]
        }
        result = _merge_response(skeleton, filled_data)
        assert (
            result.player_impacts[0].changes[0].interpretation.statement
            == "Correct interpretation"
        )

    def test_scenario_preserved(self) -> None:
        skeleton = _make_skeleton()
        filled_data = {"scenario": {"trigger_type": "TAMPERED"}}
        result = _merge_response(skeleton, filled_data)
        assert result.scenario.trigger_type == "manager_change"

    def test_confidence_notes_rewording(self) -> None:
        skeleton = _make_skeleton()
        filled_data = {"confidence_notes": ["Refined note wording."]}
        result = _merge_response(skeleton, filled_data)
        assert result.confidence_notes[0] == "Refined note wording."

    def test_confidence_notes_count_preserved(self) -> None:
        skeleton = _make_skeleton()
        # LLM tries to add extra notes.
        filled_data = {"confidence_notes": ["Note 1", "Extra note"]}
        result = _merge_response(skeleton, filled_data)
        assert len(result.confidence_notes) == 1

    def test_player_impact_axis_preserved(self) -> None:
        skeleton = _make_skeleton()
        filled_data = {
            "player_impacts": [
                {
                    "player_name": "Player A",
                    "changes": [
                        {
                            "axis": "TAMPERED",
                            "diff": 999,
                            "interpretation": {
                                "statement": "Filled text",
                                "label": "hypothesis",
                                "source": "llm_knowledge",
                            },
                        }
                    ],
                }
            ]
        }
        result = _merge_response(skeleton, filled_data)
        pi = result.player_impacts[0]
        assert pi.changes[0].axis == "form"
        assert pi.changes[0].diff == -0.15
        assert pi.changes[0].interpretation.label == "data"
        assert pi.changes[0].interpretation.statement == "Filled text"


# ---------------------------------------------------------------------------
# Tests: complete_skeleton
# ---------------------------------------------------------------------------


class TestCompleteSkeleton:
    def test_fills_statements(self) -> None:
        skeleton = _make_skeleton()
        response = _make_filled_response(skeleton)
        client = FakeLLMClient(response)
        result = complete_skeleton(
            client, skeleton, system_prompt="Test prompt"
        )
        assert result.causal_chain[0].cause == "Manager change disrupted tactics"

    def test_empty_response_returns_skeleton(self) -> None:
        skeleton = _make_skeleton()
        client = FakeLLMClient("")
        result = complete_skeleton(
            client, skeleton, system_prompt="Test prompt"
        )
        assert result is skeleton

    def test_invalid_json_returns_skeleton(self) -> None:
        skeleton = _make_skeleton()
        client = FakeLLMClient("not valid json at all")
        result = complete_skeleton(
            client, skeleton, system_prompt="Test prompt"
        )
        assert result is skeleton

    def test_system_user_message_structure(self) -> None:
        skeleton = _make_skeleton()
        response = _make_filled_response(skeleton)
        client = FakeLLMClient(response)
        complete_skeleton(client, skeleton, system_prompt="Test prompt")
        assert len(client.last_messages) == 2
        assert client.last_messages[0]["role"] == "system"
        assert client.last_messages[1]["role"] == "user"

    def test_user_message_is_valid_json(self) -> None:
        skeleton = _make_skeleton()
        response = _make_filled_response(skeleton)
        client = FakeLLMClient(response)
        complete_skeleton(client, skeleton, system_prompt="Test prompt")
        parsed = json.loads(client.last_messages[1]["content"])
        assert isinstance(parsed, dict)
        assert "scenario" in parsed
