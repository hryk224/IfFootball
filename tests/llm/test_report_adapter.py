"""Tests for the StructuredExplanation -> ReportInput adapter."""

from __future__ import annotations

from iffootball.llm.report_adapter import structured_to_report_input
from iffootball.simulation.report_planner import DisplayContext, plan_report
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
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_player_impact(
    name: str,
    form_diff: float = -0.1,
    fatigue_diff: float = 0.02,
    understanding_diff: float = -0.25,
    trust_diff: float = -0.05,
    impact_score: float = 0.3,
) -> PlayerImpactSummary:
    """Build a PlayerImpactSummary with standard 4 axes."""
    return PlayerImpactSummary(
        player_name=name,
        impact_score=impact_score,
        changes=(
            PlayerImpactChange(
                axis="form",
                diff=form_diff,
                interpretation=EvidenceItem(
                    statement=f"Form diff for {name}",
                    label="data",
                    source="simulation_output",
                ),
            ),
            PlayerImpactChange(
                axis="fatigue",
                diff=fatigue_diff,
                interpretation=EvidenceItem(
                    statement=f"Fatigue diff for {name}",
                    label="data",
                    source="simulation_output",
                ),
            ),
            PlayerImpactChange(
                axis="understanding",
                diff=understanding_diff,
                interpretation=EvidenceItem(
                    statement="Tactical reset",
                    label="data",
                    source="simulation_output",
                ),
            ),
            PlayerImpactChange(
                axis="trust",
                diff=trust_diff,
                interpretation=EvidenceItem(
                    statement=f"Trust diff for {name}",
                    label="data",
                    source="simulation_output",
                ),
            ),
        ),
        related_step_ids=(),
    )


def _make_multi_player_explanation() -> StructuredExplanation:
    """Build explanation with 3 players sharing understanding_diff=-0.25."""
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
                        statement="Points decreased",
                        label="data",
                        source="simulation_output",
                    ),
                ),
            ),
        ),
        causal_chain=(),
        player_impacts=(
            _make_player_impact("Player A", form_diff=-0.15, trust_diff=-0.08, impact_score=0.4),
            _make_player_impact("Player B", form_diff=0.03, trust_diff=0.12, impact_score=0.35),
            _make_player_impact("Player C", form_diff=-0.13, trust_diff=-0.04, impact_score=0.3),
        ),
        limitations=LimitationsDisclosure(system=SYSTEM_LIMITATIONS, scenario=()),
    )


def _make_explanation() -> StructuredExplanation:
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
                        statement="Points decreased by 5",
                        label="data",
                        source="simulation_output",
                    ),
                ),
            ),
            DifferenceHighlight(
                metric_name="form_drop",
                value_a=0.0,
                value_b=2.5,
                diff=2.5,
                interpretations=(
                    EvidenceItem(
                        statement="Form drops increased",
                        label="data",
                        source="simulation_output",
                    ),
                ),
            ),
        ),
        causal_chain=(
            CausalStep(
                step_id="cs-001",
                cause="Manager change disrupted tactics",
                effect="Form dropped for key players",
                affected_agent="Player A",
                event_type="form_drop",
                evidence=(
                    EvidenceItem(
                        statement="Form decreased by 0.15",
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
                            statement="Form declined",
                            label="data",
                            source="simulation_output",
                        ),
                    ),
                    PlayerImpactChange(
                        axis="fatigue",
                        diff=0.05,
                        interpretation=EvidenceItem(
                            statement="Slight fatigue increase",
                            label="data",
                            source="simulation_output",
                        ),
                    ),
                    PlayerImpactChange(
                        axis="understanding",
                        diff=-0.25,
                        interpretation=EvidenceItem(
                            statement="Tactical reset",
                            label="data",
                            source="simulation_output",
                        ),
                    ),
                    PlayerImpactChange(
                        axis="trust",
                        diff=-0.1,
                        interpretation=EvidenceItem(
                            statement="Trust declined",
                            label="data",
                            source="simulation_output",
                        ),
                    ),
                ),
                related_step_ids=("cs-001",),
            ),
        ),
        limitations=LimitationsDisclosure(
            system=SYSTEM_LIMITATIONS,
            scenario=(
                LimitationItem(
                    category=LimitationCategory.CHAIN_DEPTH,
                    message_en="Causal chain reaches depth 3.",
                    message_ja="因果連鎖が深さ 3 に達しています。",
                    severity="warning",
                    related_step_ids=("cs-001",),
                ),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStructuredToReportInput:
    def test_trigger_description_manager_change(self) -> None:
        ri = structured_to_report_input(_make_explanation(), n_runs=10)
        assert "Arteta" in ri.trigger_description
        assert "Xabi Alonso" in ri.trigger_description

    def test_trigger_description_transfer(self) -> None:
        exp = StructuredExplanation(
            scenario=ScenarioDescriptor(
                trigger_type="player_transfer_in",
                team_name="Arsenal",
                detail={"player_name": "Wirtz", "expected_role": "starter"},
            ),
            highlights=(),
            causal_chain=(),
            player_impacts=(),
            limitations=LimitationsDisclosure(system=(), scenario=()),
        )
        ri = structured_to_report_input(exp, n_runs=5)
        assert "Wirtz" in ri.trigger_description
        assert "starter" in ri.trigger_description

    def test_points_extracted(self) -> None:
        ri = structured_to_report_input(_make_explanation(), n_runs=10)
        assert ri.points_mean_a == 50.0
        assert ri.points_mean_b == 45.0
        assert ri.points_mean_diff == -5.0

    def test_cascade_diff_extracted(self) -> None:
        ri = structured_to_report_input(_make_explanation(), n_runs=10)
        assert "form_drop" in ri.cascade_count_diff
        assert ri.cascade_count_diff["form_drop"] == 2.5
        assert "total_points_mean" not in ri.cascade_count_diff

    def test_player_impacts_mapped(self) -> None:
        ri = structured_to_report_input(_make_explanation(), n_runs=10)
        assert len(ri.player_impacts) == 1
        pi = ri.player_impacts[0]
        assert pi.player_name == "Player A"
        assert pi.form_diff == -0.15
        assert pi.fatigue_diff == 0.05
        assert pi.understanding_diff == -0.25
        assert pi.trust_diff == -0.1

    def test_action_explanations_always_empty(self) -> None:
        # CausalStep does not map to ActionExplanationEntry contract.
        ri = structured_to_report_input(_make_explanation(), n_runs=10)
        assert ri.action_explanations == []

    def test_n_runs_passed_through(self) -> None:
        ri = structured_to_report_input(_make_explanation(), n_runs=20)
        assert ri.n_runs == 20

    def test_limitations_includes_system_and_scenario_warnings(self) -> None:
        ri = structured_to_report_input(_make_explanation(), n_runs=10)
        # System limitations (5) + scenario warnings (1 depth warning).
        assert len(ri.limitations) >= 6
        # Scenario warning is included.
        assert any("depth" in lim.lower() for lim in ri.limitations)

    def test_limitations_override(self) -> None:
        custom = ["Custom limitation 1", "Custom limitation 2"]
        ri = structured_to_report_input(
            _make_explanation(), n_runs=10, limitations=custom
        )
        assert ri.limitations == custom

    def test_plan_adds_display_hints(self) -> None:
        exp = _make_explanation()
        plan = plan_report(exp, DisplayContext.STANDARD)
        ri = structured_to_report_input(exp, plan=plan, n_runs=10)

        assert ri.display_hints is not None
        assert "summary" in ri.display_hints.section_order
        assert len(ri.display_hints.featured_players) <= 3
        assert ri.display_hints.show_limitations_info is False

    def test_plan_filters_players(self) -> None:
        exp = _make_explanation()
        plan = plan_report(exp, DisplayContext.COMPACT)
        ri = structured_to_report_input(exp, plan=plan, n_runs=10)

        # Compact limits to 1 player.
        assert len(ri.player_impacts) == 1

    def test_plan_analyst_includes_info_limitations(self) -> None:
        exp = _make_explanation()
        plan = plan_report(exp, DisplayContext.ANALYST)
        ri = structured_to_report_input(exp, plan=plan, n_runs=10)

        # Analyst includes info-severity limitations.
        # System (5) + scenario warning (1) + scenario info would be included.
        # The fixture has 1 scenario warning. No scenario info in _make_explanation.
        # But plan.include_info = True, so the filter allows info items.
        assert ri.display_hints is not None
        assert ri.display_hints.show_limitations_info is True

    def test_no_plan_gives_none_display_hints(self) -> None:
        ri = structured_to_report_input(_make_explanation(), n_runs=10)
        assert ri.display_hints is None

    def test_plan_preserves_step_ids_in_hints(self) -> None:
        exp = _make_explanation()
        plan = plan_report(exp, DisplayContext.STANDARD)
        ri = structured_to_report_input(exp, plan=plan, n_runs=10)

        assert ri.display_hints is not None
        assert ri.display_hints.expanded_step_ids == plan.expanded_step_ids
        assert ri.display_hints.collapsed_step_ids == plan.collapsed_step_ids

    # ------------------------------------------------------------------
    # Shared reset detection and axis filtering tests
    # ------------------------------------------------------------------

    def test_shared_reset_detected_in_meta(self) -> None:
        exp = _make_multi_player_explanation()
        plan = plan_report(exp, DisplayContext.STANDARD)
        ri = structured_to_report_input(exp, plan=plan, n_runs=10)

        assert ri.player_impact_meta is not None
        assert "understanding" in ri.player_impact_meta.shared_resets
        assert ri.player_impact_meta.shared_resets["understanding"] == -0.25

    def test_shared_reset_removed_from_individual_changes(self) -> None:
        exp = _make_multi_player_explanation()
        plan = plan_report(exp, DisplayContext.STANDARD)
        ri = structured_to_report_input(exp, plan=plan, n_runs=10)

        assert ri.player_impact_details is not None
        for player in ri.player_impact_details:
            axes = [c.axis for c in player.changes]
            assert "understanding" not in axes, (
                f"{player.player_name} still has understanding in changes"
            )

    def test_max_two_axes_per_player(self) -> None:
        exp = _make_multi_player_explanation()
        plan = plan_report(exp, DisplayContext.STANDARD)
        ri = structured_to_report_input(exp, plan=plan, n_runs=10)

        assert ri.player_impact_details is not None
        for player in ri.player_impact_details:
            assert len(player.changes) <= 2, (
                f"{player.player_name} has {len(player.changes)} axes, expected <= 2"
            )

    def test_axes_sorted_by_absolute_diff(self) -> None:
        exp = _make_multi_player_explanation()
        plan = plan_report(exp, DisplayContext.STANDARD)
        ri = structured_to_report_input(exp, plan=plan, n_runs=10)

        assert ri.player_impact_details is not None
        for player in ri.player_impact_details:
            if len(player.changes) >= 2:
                assert abs(player.changes[0].diff) >= abs(player.changes[1].diff)

    def test_no_shared_reset_when_values_differ(self) -> None:
        """When understanding_diff differs across players, it is not shared."""
        exp = StructuredExplanation(
            scenario=ScenarioDescriptor(
                trigger_type="manager_change",
                team_name="Arsenal",
                detail={"outgoing_manager": "A", "incoming_manager": "B"},
            ),
            highlights=(),
            causal_chain=(),
            player_impacts=(
                _make_player_impact("Player A", understanding_diff=-0.25),
                _make_player_impact("Player B", understanding_diff=-0.10),
            ),
            limitations=LimitationsDisclosure(system=(), scenario=()),
        )
        ri = structured_to_report_input(exp, n_runs=10)

        # understanding differs across players, so it should NOT be shared.
        if ri.player_impact_meta is not None:
            assert "understanding" not in ri.player_impact_meta.shared_resets
        # understanding should remain in individual changes.
        assert ri.player_impact_details is not None
        for player in ri.player_impact_details:
            axes = [c.axis for c in player.changes]
            assert "understanding" in axes

    def test_no_meta_when_no_players(self) -> None:
        exp = StructuredExplanation(
            scenario=ScenarioDescriptor(
                trigger_type="manager_change",
                team_name="Arsenal",
                detail={"outgoing_manager": "A", "incoming_manager": "B"},
            ),
            highlights=(),
            causal_chain=(),
            player_impacts=(),
            limitations=LimitationsDisclosure(system=(), scenario=()),
        )
        ri = structured_to_report_input(exp, n_runs=5)
        assert ri.player_impact_meta is None

    def test_empty_explanation(self) -> None:
        exp = StructuredExplanation(
            scenario=ScenarioDescriptor(
                trigger_type="manager_change",
                team_name="Arsenal",
                detail={
                    "outgoing_manager": "A",
                    "incoming_manager": "B",
                },
            ),
            highlights=(),
            causal_chain=(),
            player_impacts=(),
            limitations=LimitationsDisclosure(system=(), scenario=()),
        )
        ri = structured_to_report_input(exp, n_runs=5)
        assert ri.points_mean_a == 0.0
        assert ri.player_impacts == []
        assert ri.action_explanations == []
