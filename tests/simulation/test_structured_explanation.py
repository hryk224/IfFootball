"""Tests for structured_explanation schema and validation."""

from __future__ import annotations

import pytest

from iffootball.simulation.structured_explanation import (
    SYSTEM_LIMITATIONS,
    CausalStep,
    DifferenceHighlight,
    EvidenceItem,
    LimitationCategory,
    LimitationItem,
    LimitationsDisclosure,
    PlayerImpactChange,
    PlayerImpactSummary,
    ScenarioDescriptor,
    StructuredExplanation,
    generate_scenario_limitations,
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
# EvidenceItem / DifferenceHighlight / CausalStep / PlayerImpact
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


class TestCausalStep:
    def test_skeleton_step(self) -> None:
        step = CausalStep(
            step_id="cs-001",
            cause="",
            effect="",
            affected_agent="Player A",
            event_type="form_drop",
            evidence=(
                EvidenceItem(statement="", label="data", source="simulation_output"),
            ),
            depth=1,
        )
        assert step.step_id == "cs-001"
        assert step.cause == ""


class TestPlayerImpact:
    def test_change_with_axis(self) -> None:
        change = PlayerImpactChange(
            axis="form",
            diff=-0.15,
            interpretation=EvidenceItem(
                statement="", label="data", source="simulation_output"
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
            ),
            related_step_ids=("cs-001", "cs-003"),
        )
        assert summary.related_step_ids == ("cs-001", "cs-003")


# ---------------------------------------------------------------------------
# LimitationItem / LimitationsDisclosure
# ---------------------------------------------------------------------------


class TestLimitationItem:
    def test_system_limitation(self) -> None:
        item = LimitationItem(
            category=LimitationCategory.MODEL_BOUNDARY,
            message_en="Match uses Poisson model.",
            message_ja="試合は Poisson モデルを使用。",
            severity="warning",
        )
        assert item.category == LimitationCategory.MODEL_BOUNDARY
        assert item.related_step_ids == ()

    def test_scenario_limitation_with_step_ids(self) -> None:
        item = LimitationItem(
            category=LimitationCategory.CHAIN_DEPTH,
            message_en="Deep chain.",
            message_ja="深い連鎖。",
            severity="warning",
            related_step_ids=("cs-003", "cs-004"),
        )
        assert item.related_step_ids == ("cs-003", "cs-004")


class TestLimitationsDisclosure:
    def test_two_layer_structure(self) -> None:
        disclosure = LimitationsDisclosure(
            system=(
                LimitationItem(
                    category=LimitationCategory.MODEL_BOUNDARY,
                    message_en="System limit.",
                    message_ja="システム制約。",
                    severity="warning",
                ),
            ),
            scenario=(
                LimitationItem(
                    category=LimitationCategory.CHAIN_DEPTH,
                    message_en="Scenario limit.",
                    message_ja="シナリオ制約。",
                    severity="warning",
                ),
            ),
        )
        assert len(disclosure.system) == 1
        assert len(disclosure.scenario) == 1


class TestSystemLimitations:
    def test_all_have_both_languages(self) -> None:
        for item in SYSTEM_LIMITATIONS:
            assert item.message_en
            assert item.message_ja

    def test_at_least_five_items(self) -> None:
        assert len(SYSTEM_LIMITATIONS) >= 5

    def test_all_have_category(self) -> None:
        for item in SYSTEM_LIMITATIONS:
            assert isinstance(item.category, LimitationCategory)


# ---------------------------------------------------------------------------
# generate_scenario_limitations
# ---------------------------------------------------------------------------


class TestScenarioLimitations:
    def test_empty_chain_returns_empty(self) -> None:
        assert generate_scenario_limitations(()) == ()

    def test_shallow_chain_no_depth_warning(self) -> None:
        chain = (
            CausalStep(
                step_id="cs-001",
                cause="",
                effect="",
                affected_agent="P",
                event_type="form_drop",
                evidence=(
                    EvidenceItem(statement="", label="data", source="simulation_output"),
                ),
                depth=2,
            ),
        )
        items = generate_scenario_limitations(chain)
        assert not any(i.category == LimitationCategory.CHAIN_DEPTH for i in items)

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
                        statement="", label="hypothesis", source="rule_based_model"
                    ),
                ),
                depth=4,
            ),
        )
        items = generate_scenario_limitations(chain)
        depth_items = [i for i in items if i.category == LimitationCategory.CHAIN_DEPTH]
        assert len(depth_items) == 1
        assert depth_items[0].severity == "warning"
        assert "cs-001" in depth_items[0].related_step_ids

    def test_high_non_simulation_ratio(self) -> None:
        chain = tuple(
            CausalStep(
                step_id=f"cs-{i:03d}",
                cause="",
                effect="",
                affected_agent="P",
                event_type="form_drop",
                evidence=(
                    EvidenceItem(
                        statement="", label="hypothesis", source="llm_knowledge"
                    ),
                ),
                depth=1,
            )
            for i in range(3)
        )
        items = generate_scenario_limitations(chain)
        est_items = [
            i for i in items if i.category == LimitationCategory.ESTIMATION_DEPENDENCY
        ]
        assert len(est_items) == 1
        assert "100%" in est_items[0].message_en

    def test_all_simulation_output_no_estimation_warning(self) -> None:
        chain = (
            CausalStep(
                step_id="cs-001",
                cause="",
                effect="",
                affected_agent="P",
                event_type="form_drop",
                evidence=(
                    EvidenceItem(statement="", label="data", source="simulation_output"),
                ),
                depth=1,
            ),
        )
        items = generate_scenario_limitations(chain)
        assert not any(
            i.category == LimitationCategory.ESTIMATION_DEPENDENCY for i in items
        )

    def test_bilingual_messages(self) -> None:
        chain = (
            CausalStep(
                step_id="cs-001",
                cause="",
                effect="",
                affected_agent="P",
                event_type="form_drop",
                evidence=(
                    EvidenceItem(
                        statement="", label="hypothesis", source="rule_based_model"
                    ),
                ),
                depth=4,
            ),
        )
        items = generate_scenario_limitations(chain)
        for item in items:
            assert item.message_en
            assert item.message_ja


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
            limitations=LimitationsDisclosure(system=(), scenario=()),
        )
        assert se.scenario.team_name == "Arsenal"
        assert se.highlights == ()
