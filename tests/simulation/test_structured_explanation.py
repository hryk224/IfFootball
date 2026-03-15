"""Tests for structured_explanation schema and validation."""

from __future__ import annotations

import pytest

from iffootball.simulation.structured_explanation import (
    CausalStep,
    DifferenceHighlight,
    EvidenceItem,
    PlayerImpactChange,
    PlayerImpactSummary,
    ScenarioDescriptor,
    StructuredExplanation,
    generate_confidence_note_drafts,
    infer_label,
)


# ---------------------------------------------------------------------------
# infer_label
# ---------------------------------------------------------------------------


class TestInferLabel:
    def test_simulation_output_always_data(self) -> None:
        assert infer_label("simulation_output", depth=1) == "data"
        assert infer_label("simulation_output", depth=4) == "data"

    def test_rule_based_shallow_is_analysis(self) -> None:
        assert infer_label("rule_based_model", depth=1) == "analysis"
        assert infer_label("rule_based_model", depth=2) == "analysis"

    def test_rule_based_deep_is_hypothesis(self) -> None:
        assert infer_label("rule_based_model", depth=3) == "hypothesis"

    def test_llm_knowledge_always_hypothesis(self) -> None:
        assert infer_label("llm_knowledge", depth=1) == "hypothesis"


# ---------------------------------------------------------------------------
# ScenarioDescriptor validation
# ---------------------------------------------------------------------------


class TestScenarioDescriptor:
    def test_valid_manager_change(self) -> None:
        sd = ScenarioDescriptor(
            trigger_type="manager_change",
            team_name="Arsenal",
            detail={
                "outgoing_manager": "Arteta",
                "incoming_manager": "Xabi Alonso",
            },
        )
        assert sd.trigger_type == "manager_change"
        assert sd.detail["incoming_manager"] == "Xabi Alonso"

    def test_valid_transfer_in(self) -> None:
        sd = ScenarioDescriptor(
            trigger_type="player_transfer_in",
            team_name="Arsenal",
            detail={
                "player_name": "Florian Wirtz",
                "expected_role": "starter",
            },
        )
        assert sd.trigger_type == "player_transfer_in"

    def test_missing_required_key_raises(self) -> None:
        with pytest.raises(ValueError, match="missing required detail keys"):
            ScenarioDescriptor(
                trigger_type="manager_change",
                team_name="Arsenal",
                detail={"outgoing_manager": "Arteta"},
            )

    def test_unknown_trigger_type_no_validation(self) -> None:
        # Unknown types have no required keys, so no error.
        sd = ScenarioDescriptor(
            trigger_type="unknown",
            team_name="Team A",
            detail={},
        )
        assert sd.trigger_type == "unknown"

    def test_extra_keys_allowed(self) -> None:
        sd = ScenarioDescriptor(
            trigger_type="manager_change",
            team_name="Arsenal",
            detail={
                "outgoing_manager": "Arteta",
                "incoming_manager": "Xabi Alonso",
                "transition_type": "mid_season",
            },
        )
        assert "transition_type" in sd.detail


# ---------------------------------------------------------------------------
# EvidenceItem
# ---------------------------------------------------------------------------


class TestEvidenceItem:
    def test_empty_statement_allowed(self) -> None:
        ev = EvidenceItem(statement="", label="data", source="simulation_output")
        assert ev.statement == ""

    def test_fields_preserved(self) -> None:
        ev = EvidenceItem(
            statement="Form dropped by 0.15",
            label="data",
            source="simulation_output",
        )
        assert ev.label == "data"
        assert ev.source == "simulation_output"


# ---------------------------------------------------------------------------
# DifferenceHighlight
# ---------------------------------------------------------------------------


class TestDifferenceHighlight:
    def test_multiple_interpretations(self) -> None:
        dh = DifferenceHighlight(
            metric_name="total_points_mean",
            value_a=55.0,
            value_b=48.0,
            diff=-7.0,
            interpretations=(
                EvidenceItem(
                    statement="Points dropped",
                    label="data",
                    source="simulation_output",
                ),
                EvidenceItem(
                    statement="Transition period impact",
                    label="analysis",
                    source="rule_based_model",
                ),
            ),
        )
        assert len(dh.interpretations) == 2
        assert dh.interpretations[0].label == "data"
        assert dh.interpretations[1].label == "analysis"


# ---------------------------------------------------------------------------
# CausalStep
# ---------------------------------------------------------------------------


class TestCausalStep:
    def test_skeleton_step(self) -> None:
        step = CausalStep(
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
        )
        assert step.step_id == "cs-001"
        assert step.cause == ""
        assert step.affected_agent == "Player A"


# ---------------------------------------------------------------------------
# PlayerImpactChange / PlayerImpactSummary
# ---------------------------------------------------------------------------


class TestPlayerImpact:
    def test_change_with_axis(self) -> None:
        change = PlayerImpactChange(
            axis="form",
            diff=-0.15,
            interpretation=EvidenceItem(
                statement="",
                label="data",
                source="simulation_output",
            ),
        )
        assert change.axis == "form"
        assert change.diff == -0.15

    def test_summary_with_related_steps(self) -> None:
        summary = PlayerImpactSummary(
            player_name="Player A",
            impact_score=0.35,
            changes=(
                PlayerImpactChange(
                    axis="form",
                    diff=-0.1,
                    interpretation=EvidenceItem(
                        statement="", label="data", source="simulation_output"
                    ),
                ),
                PlayerImpactChange(
                    axis="trust",
                    diff=-0.05,
                    interpretation=EvidenceItem(
                        statement="", label="data", source="simulation_output"
                    ),
                ),
            ),
            related_step_ids=("cs-001", "cs-003"),
        )
        assert len(summary.changes) == 2
        assert summary.changes[0].axis == "form"
        assert summary.related_step_ids == ("cs-001", "cs-003")


# ---------------------------------------------------------------------------
# generate_confidence_note_drafts
# ---------------------------------------------------------------------------


class TestConfidenceNoteDrafts:
    def test_empty_chain_returns_empty(self) -> None:
        assert generate_confidence_note_drafts(()) == []

    def test_shallow_chain_no_depth_warning(self) -> None:
        chain = (
            CausalStep(
                step_id="cs-001",
                cause="",
                effect="",
                affected_agent="P",
                event_type="form_drop",
                evidence=(
                    EvidenceItem(
                        statement="",
                        label="data",
                        source="simulation_output",
                    ),
                ),
                depth=2,
            ),
        )
        notes = generate_confidence_note_drafts(chain)
        assert not any("depth" in n.lower() for n in notes)

    def test_deep_chain_triggers_depth_warning(self) -> None:
        chain = (
            CausalStep(
                step_id="cs-001",
                cause="",
                effect="",
                affected_agent="P",
                event_type="form_drop",
                evidence=(
                    EvidenceItem(
                        statement="",
                        label="hypothesis",
                        source="rule_based_model",
                    ),
                ),
                depth=4,
            ),
        )
        notes = generate_confidence_note_drafts(chain)
        assert any("depth" in n.lower() for n in notes)

    def test_high_non_simulation_ratio_triggers_warning(self) -> None:
        chain = tuple(
            CausalStep(
                step_id=f"cs-{i:03d}",
                cause="",
                effect="",
                affected_agent="P",
                event_type="form_drop",
                evidence=(
                    EvidenceItem(
                        statement="",
                        label="hypothesis",
                        source="llm_knowledge",
                    ),
                ),
                depth=1,
            )
            for i in range(3)
        )
        notes = generate_confidence_note_drafts(chain)
        assert any("100%" in n for n in notes)

    def test_all_simulation_output_no_source_warning(self) -> None:
        chain = (
            CausalStep(
                step_id="cs-001",
                cause="",
                effect="",
                affected_agent="P",
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
        )
        notes = generate_confidence_note_drafts(chain)
        assert not any("%" in n for n in notes)


# ---------------------------------------------------------------------------
# StructuredExplanation (top-level)
# ---------------------------------------------------------------------------


class TestStructuredExplanation:
    def test_minimal_skeleton(self) -> None:
        se = StructuredExplanation(
            scenario=ScenarioDescriptor(
                trigger_type="manager_change",
                team_name="Arsenal",
                detail={
                    "outgoing_manager": "Arteta",
                    "incoming_manager": "Xabi Alonso",
                },
            ),
            highlights=(),
            causal_chain=(),
            player_impacts=(),
            confidence_notes=(),
        )
        assert se.scenario.team_name == "Arsenal"
        assert se.highlights == ()
