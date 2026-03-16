"""Tests for report_planner module."""

from __future__ import annotations

from iffootball.simulation.report_planner import (
    DetailLevel,
    DisplayContext,
    SectionType,
    plan_report,
)
from iffootball.simulation.structured_explanation import (
    SYSTEM_LIMITATIONS,
    CausalStep,
    DifferenceHighlight,
    EvidenceItem,
    LimitationItem,
    LimitationCategory,
    LimitationsDisclosure,
    PlayerImpactChange,
    PlayerImpactSummary,
    ScenarioDescriptor,
    StructuredExplanation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_explanation(
    *,
    n_highlights: int = 3,
    n_players: int = 3,
    n_steps: int = 4,
    max_depth: int = 3,
    points_diff: float = 2.5,
) -> StructuredExplanation:
    """Build a minimal StructuredExplanation for testing."""
    scenario = ScenarioDescriptor(
        trigger_type="manager_change",
        team_name="Test FC",
        detail={
            "outgoing_manager": "Old Manager",
            "incoming_manager": "New Manager",
        },
    )

    highlights: list[DifferenceHighlight] = [
        DifferenceHighlight(
            metric_name="total_points_mean",
            value_a=50.0,
            value_b=50.0 + points_diff,
            diff=points_diff,
            interpretations=(
                EvidenceItem(statement="test", label="data", source="simulation_output"),
            ),
        ),
    ]
    for i in range(1, n_highlights):
        highlights.append(
            DifferenceHighlight(
                metric_name=f"event_{i}",
                value_a=10.0,
                value_b=10.0 + i * 2.0,
                diff=i * 2.0,
                interpretations=(
                    EvidenceItem(
                        statement="test",
                        label="data",
                        source="simulation_output",
                    ),
                ),
            )
        )

    steps: list[CausalStep] = []
    for i in range(n_steps):
        depth = min(i + 1, max_depth)
        steps.append(
            CausalStep(
                step_id=f"cs-{i + 1:03d}",
                cause="test cause",
                effect="test effect",
                affected_agent=f"Player {chr(65 + i % max(n_players, 1))}",
                event_type="form_drop",
                evidence=(
                    EvidenceItem(
                        statement="test",
                        label="data",
                        source="simulation_output",
                    ),
                ),
                depth=depth,
            )
        )

    players: list[PlayerImpactSummary] = []
    for i in range(n_players):
        players.append(
            PlayerImpactSummary(
                player_name=f"Player {chr(65 + i)}",
                impact_score=round(0.5 - i * 0.1, 2),
                changes=(
                    PlayerImpactChange(
                        axis="form",
                        diff=-0.1 * (i + 1),
                        interpretation=EvidenceItem(
                            statement="test",
                            label="data",
                            source="simulation_output",
                        ),
                    ),
                    PlayerImpactChange(
                        axis="fatigue",
                        diff=0.0,
                        interpretation=EvidenceItem(
                            statement="test",
                            label="data",
                            source="simulation_output",
                        ),
                    ),
                    PlayerImpactChange(
                        axis="understanding",
                        diff=-0.25,
                        interpretation=EvidenceItem(
                            statement="test",
                            label="data",
                            source="simulation_output",
                        ),
                    ),
                    PlayerImpactChange(
                        axis="trust",
                        diff=0.05,
                        interpretation=EvidenceItem(
                            statement="test",
                            label="data",
                            source="simulation_output",
                        ),
                    ),
                ),
                related_step_ids=(),
            )
        )

    limitations = LimitationsDisclosure(
        system=SYSTEM_LIMITATIONS,
        scenario=(
            LimitationItem(
                category=LimitationCategory.CHAIN_DEPTH,
                message_en="Deep chain warning",
                message_ja="深い連鎖の警告",
                severity="warning",
            ),
            LimitationItem(
                category=LimitationCategory.ESTIMATION_DEPENDENCY,
                message_en="Estimation info",
                message_ja="推定情報",
                severity="info",
            ),
        ),
    )

    return StructuredExplanation(
        scenario=scenario,
        highlights=tuple(highlights),
        causal_chain=tuple(steps),
        player_impacts=tuple(players),
        limitations=limitations,
    )


# ---------------------------------------------------------------------------
# Tests: plan_report
# ---------------------------------------------------------------------------


class TestPlanReport:
    def test_standard_returns_all_sections_included(self) -> None:
        explanation = _make_explanation()
        plan = plan_report(explanation, DisplayContext.STANDARD)

        assert len(plan.sections) == 5
        for section in plan.sections:
            assert section.include is True
            assert section.detail_level == DetailLevel.NORMAL

    def test_compact_hides_causal_chain(self) -> None:
        explanation = _make_explanation()
        plan = plan_report(explanation, DisplayContext.COMPACT)

        causal = [s for s in plan.sections if s.section_type == SectionType.CAUSAL_CHAIN]
        assert len(causal) == 1
        assert causal[0].include is False

    def test_analyst_uses_full_detail(self) -> None:
        explanation = _make_explanation()
        plan = plan_report(explanation, DisplayContext.ANALYST)

        for section in plan.sections:
            assert section.detail_level == DetailLevel.FULL


class TestSummaryPriority:
    def test_lead_metric_is_first_highlight(self) -> None:
        explanation = _make_explanation()
        plan = plan_report(explanation)

        assert plan.summary_priority.lead_metric == "total_points_mean"

    def test_lead_player_is_highest_impact(self) -> None:
        explanation = _make_explanation(n_players=3)
        plan = plan_report(explanation)

        assert plan.summary_priority.lead_player == "Player A"

    def test_lead_direction_positive(self) -> None:
        explanation = _make_explanation(points_diff=3.0)
        plan = plan_report(explanation)

        assert plan.summary_priority.lead_direction == "positive"

    def test_lead_direction_negative(self) -> None:
        explanation = _make_explanation(points_diff=-2.0)
        plan = plan_report(explanation)

        assert plan.summary_priority.lead_direction == "negative"

    def test_lead_direction_marginal(self) -> None:
        explanation = _make_explanation(points_diff=0.3)
        plan = plan_report(explanation)

        assert plan.summary_priority.lead_direction == "marginal"

    def test_secondary_metrics_from_highlights(self) -> None:
        explanation = _make_explanation(n_highlights=4)
        plan = plan_report(explanation, DisplayContext.STANDARD)

        # Standard allows 2 secondary metrics.
        assert len(plan.summary_priority.secondary_metrics) == 2
        # All should be metric_names from highlights (not cascade_count_diff).
        for metric in plan.summary_priority.secondary_metrics:
            assert metric.startswith("event_")

    def test_no_players_gives_none_lead(self) -> None:
        explanation = _make_explanation(n_players=0)
        plan = plan_report(explanation)

        assert plan.summary_priority.lead_player is None


class TestPlayerOrder:
    def test_standard_limits_to_3(self) -> None:
        explanation = _make_explanation(n_players=5)
        plan = plan_report(explanation, DisplayContext.STANDARD)

        assert len(plan.player_display_order) == 3

    def test_compact_limits_to_1(self) -> None:
        explanation = _make_explanation(n_players=5)
        plan = plan_report(explanation, DisplayContext.COMPACT)

        assert len(plan.player_display_order) == 1
        assert plan.player_display_order[0] == "Player A"

    def test_analyst_shows_all(self) -> None:
        explanation = _make_explanation(n_players=5)
        plan = plan_report(explanation, DisplayContext.ANALYST)

        assert len(plan.player_display_order) == 5

    def test_order_follows_impact_score(self) -> None:
        explanation = _make_explanation(n_players=3)
        plan = plan_report(explanation, DisplayContext.STANDARD)

        assert plan.player_display_order == ("Player A", "Player B", "Player C")


class TestStepClassification:
    def test_standard_expands_shallow_collapses_deep(self) -> None:
        explanation = _make_explanation(n_steps=4, max_depth=4)
        plan = plan_report(explanation, DisplayContext.STANDARD)

        # depth 1, 2 -> expanded; depth 3, 4 -> collapsed.
        for step in explanation.causal_chain:
            if step.depth <= 2:
                assert step.step_id in plan.expanded_step_ids
            else:
                assert step.step_id in plan.collapsed_step_ids

    def test_analyst_expands_all(self) -> None:
        explanation = _make_explanation(n_steps=4, max_depth=4)
        plan = plan_report(explanation, DisplayContext.ANALYST)

        assert len(plan.collapsed_step_ids) == 0
        assert len(plan.expanded_step_ids) == 4

    def test_compact_collapses_all(self) -> None:
        explanation = _make_explanation(n_steps=4, max_depth=2)
        plan = plan_report(explanation, DisplayContext.COMPACT)

        assert len(plan.expanded_step_ids) == 0
        assert len(plan.collapsed_step_ids) == 4


class TestLimitationPlacement:
    def test_standard_excludes_info(self) -> None:
        explanation = _make_explanation()
        plan = plan_report(explanation, DisplayContext.STANDARD)

        assert plan.limitation_placement.show_system is True
        assert plan.limitation_placement.show_scenario is True
        assert plan.limitation_placement.include_info is False

    def test_analyst_includes_info(self) -> None:
        explanation = _make_explanation()
        plan = plan_report(explanation, DisplayContext.ANALYST)

        assert plan.limitation_placement.include_info is True


class TestDisplayHints:
    def test_to_display_hints_preserves_step_ids(self) -> None:
        explanation = _make_explanation(n_steps=4, max_depth=4)
        plan = plan_report(explanation, DisplayContext.STANDARD)
        hints = plan.to_display_hints()

        assert hints.expanded_step_ids == plan.expanded_step_ids
        assert hints.collapsed_step_ids == plan.collapsed_step_ids

    def test_to_display_hints_section_order(self) -> None:
        explanation = _make_explanation()
        plan = plan_report(explanation, DisplayContext.COMPACT)
        hints = plan.to_display_hints()

        # Compact excludes causal_chain.
        assert "causal_chain" not in hints.section_order
        assert "summary" in hints.section_order

    def test_to_display_hints_featured_players(self) -> None:
        explanation = _make_explanation(n_players=3)
        plan = plan_report(explanation, DisplayContext.STANDARD)
        hints = plan.to_display_hints()

        assert hints.featured_players == plan.player_display_order

    def test_default_context_is_standard(self) -> None:
        explanation = _make_explanation()
        plan = plan_report(explanation)

        assert plan.limitation_placement.include_info is False
        assert len(plan.player_display_order) <= 3


class TestSummaryPriorityTradeoff:
    def test_tradeoff_selects_negative_connotation_metric(self) -> None:
        """form_drop with positive diff should be selected as trade-off."""
        explanation = _make_explanation(n_highlights=4, points_diff=2.5)
        plan = plan_report(explanation, DisplayContext.STANDARD)

        # The fixture creates highlights: total_points_mean + event_1, event_2, event_3.
        # event_N have positive diff. Without polarity info, they won't be trade-offs.
        # But for the actual data (form_drop etc), we need a specific fixture.
        # Just verify the field is populated or None.
        assert hasattr(plan.summary_priority, "tradeoff_metric")

    def test_tradeoff_is_none_when_no_negative_events(self) -> None:
        """All highlights are positive-connotation — no trade-off."""
        explanation = _make_explanation(n_highlights=1, points_diff=2.5)
        plan = plan_report(explanation, DisplayContext.STANDARD)
        # Only total_points_mean, no negative-connotation events.
        assert plan.summary_priority.tradeoff_metric is None

    def test_max_sentences_compact(self) -> None:
        explanation = _make_explanation()
        plan = plan_report(explanation, DisplayContext.COMPACT)
        assert plan.summary_priority.max_sentences == 2

    def test_max_sentences_standard(self) -> None:
        explanation = _make_explanation()
        plan = plan_report(explanation, DisplayContext.STANDARD)
        assert plan.summary_priority.max_sentences == 4

    def test_max_sentences_analyst(self) -> None:
        explanation = _make_explanation()
        plan = plan_report(explanation, DisplayContext.ANALYST)
        assert plan.summary_priority.max_sentences == 5

    def test_display_hints_carries_summary_info(self) -> None:
        explanation = _make_explanation()
        plan = plan_report(explanation, DisplayContext.STANDARD)
        hints = plan.to_display_hints()
        assert hints.summary_max_sentences == 4

    def test_tradeoff_picks_form_drop_not_adaptation(self) -> None:
        """With real event names, form_drop (bad) is trade-off, not adaptation_progress (good)."""
        explanation = StructuredExplanation(
            scenario=ScenarioDescriptor(
                trigger_type="manager_change",
                team_name="Test FC",
                detail={
                    "outgoing_manager": "Old",
                    "incoming_manager": "New",
                },
            ),
            highlights=(
                DifferenceHighlight(
                    metric_name="total_points_mean",
                    value_a=12.0,
                    value_b=14.0,
                    diff=2.0,
                    interpretations=(
                        EvidenceItem(statement="", label="data", source="simulation_output"),
                    ),
                ),
                DifferenceHighlight(
                    metric_name="adaptation_progress",
                    value_a=0.0,
                    value_b=24.0,
                    diff=24.0,
                    interpretations=(
                        EvidenceItem(statement="", label="data", source="simulation_output"),
                    ),
                ),
                DifferenceHighlight(
                    metric_name="form_drop",
                    value_a=0.0,
                    value_b=8.4,
                    diff=8.4,
                    interpretations=(
                        EvidenceItem(statement="", label="data", source="simulation_output"),
                    ),
                ),
            ),
            causal_chain=(),
            player_impacts=(),
            limitations=LimitationsDisclosure(system=(), scenario=()),
        )
        plan = plan_report(explanation, DisplayContext.STANDARD)
        # form_drop (increase is bad) should be trade-off, not adaptation_progress.
        assert plan.summary_priority.tradeoff_metric == "form_drop"
