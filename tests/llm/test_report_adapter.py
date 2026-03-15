"""Tests for the StructuredExplanation -> ReportInput adapter."""

from __future__ import annotations

from iffootball.llm.report_adapter import structured_to_report_input
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
# Helpers
# ---------------------------------------------------------------------------


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
        confidence_notes=("Chain reaches depth 3.",),
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
            confidence_notes=(),
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

    def test_limitations_uses_defaults(self) -> None:
        ri = structured_to_report_input(_make_explanation(), n_runs=10)
        # Should use DEFAULT_LIMITATIONS, not confidence_notes.
        assert "Chain reaches depth 3." not in ri.limitations
        assert len(ri.limitations) >= 3  # DEFAULT_LIMITATIONS has 5 entries

    def test_limitations_override(self) -> None:
        custom = ["Custom limitation 1", "Custom limitation 2"]
        ri = structured_to_report_input(
            _make_explanation(), n_runs=10, limitations=custom
        )
        assert ri.limitations == custom

    def test_limitations_ja(self) -> None:
        ri = structured_to_report_input(
            _make_explanation(), n_runs=10, lang="ja"
        )
        # Japanese limitations should be present.
        assert len(ri.limitations) >= 3

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
            confidence_notes=(),
        )
        ri = structured_to_report_input(exp, n_runs=5)
        assert ri.points_mean_a == 0.0
        assert ri.player_impacts == []
        assert ri.action_explanations == []
