"""Tests for next scenario suggestion generation."""

from __future__ import annotations

from iffootball.simulation.comparison import (
    AggregatedResult,
    ComparisonResult,
    DeltaMetrics,
)
from iffootball.simulation.scenario_suggestions import (
    ScenarioSuggestion,
    generate_scenario_suggestions,
    render_suggestions_markdown,
)
from iffootball.simulation.structured_explanation import (
    EvidenceItem,
    LimitationsDisclosure,
    PlayerImpactChange,
    PlayerImpactSummary,
    ScenarioDescriptor,
    StructuredExplanation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_comparison(
    points_diff: float = 0.5,
    tactical_confusion: float = 20.0,
) -> ComparisonResult:
    cascade_b = {"tactical_confusion": tactical_confusion, "form_drop": 10.0}
    cascade_a: dict[str, float] = {}
    return ComparisonResult(
        no_change=AggregatedResult(
            n_runs=20,
            total_points_mean=12.0,
            total_points_median=12.0,
            total_points_std=3.0,
            cascade_event_counts=cascade_a,
            run_results=(),
        ),
        with_change=AggregatedResult(
            n_runs=20,
            total_points_mean=12.0 + points_diff,
            total_points_median=12.0 + points_diff,
            total_points_std=3.0,
            cascade_event_counts=cascade_b,
            run_results=(),
        ),
        delta=DeltaMetrics(
            points_mean_diff=points_diff,
            points_median_diff=points_diff,
            cascade_count_diff={"tactical_confusion": tactical_confusion},
        ),
    )


def _make_explanation(
    top_impact: float = 0.8,
    second_impact: float = 0.5,
) -> StructuredExplanation:
    def _pi(name: str, score: float) -> PlayerImpactSummary:
        return PlayerImpactSummary(
            player_name=name,
            impact_score=score,
            changes=(
                PlayerImpactChange(
                    axis="form",
                    diff=-0.1,
                    interpretation=EvidenceItem(
                        statement="", label="data", source="simulation_output"
                    ),
                ),
            ),
            related_step_ids=(),
        )

    return StructuredExplanation(
        scenario=ScenarioDescriptor(
            trigger_type="manager_change",
            team_name="Team A",
            detail={
                "outgoing_manager": "Manager A",
                "incoming_manager": "Manager B",
            },
        ),
        highlights=(),
        causal_chain=(),
        player_impacts=(
            _pi("Player Top", top_impact),
            _pi("Player Second", second_impact),
        ),
        limitations=LimitationsDisclosure(system=(), scenario=()),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGenerateScenarioSuggestions:
    def test_small_diff_triggers_alternative_manager(self) -> None:
        comparison = _make_comparison(points_diff=0.3)
        explanation = _make_explanation()
        suggestions = generate_scenario_suggestions(comparison, explanation)
        categories = [s.category for s in suggestions]
        assert "alternative_manager" in categories

    def test_large_diff_no_alternative_manager(self) -> None:
        comparison = _make_comparison(points_diff=2.0)
        explanation = _make_explanation()
        suggestions = generate_scenario_suggestions(comparison, explanation)
        categories = [s.category for s in suggestions]
        assert "alternative_manager" not in categories

    def test_high_confusion_triggers_trigger_timing(self) -> None:
        comparison = _make_comparison(tactical_confusion=25.0)
        explanation = _make_explanation()
        suggestions = generate_scenario_suggestions(comparison, explanation)
        categories = [s.category for s in suggestions]
        assert "trigger_timing" in categories

    def test_low_confusion_no_trigger_timing(self) -> None:
        comparison = _make_comparison(tactical_confusion=5.0)
        explanation = _make_explanation()
        suggestions = generate_scenario_suggestions(comparison, explanation)
        categories = [s.category for s in suggestions]
        assert "trigger_timing" not in categories

    def test_dominant_player_triggers_focus_player(self) -> None:
        comparison = _make_comparison()
        explanation = _make_explanation(top_impact=1.0, second_impact=0.5)
        suggestions = generate_scenario_suggestions(comparison, explanation)
        categories = [s.category for s in suggestions]
        assert "focus_player" in categories

    def test_balanced_players_no_focus_player(self) -> None:
        comparison = _make_comparison()
        explanation = _make_explanation(top_impact=0.8, second_impact=0.7)
        suggestions = generate_scenario_suggestions(comparison, explanation)
        categories = [s.category for s in suggestions]
        assert "focus_player" not in categories

    def test_max_one_per_category(self) -> None:
        comparison = _make_comparison(points_diff=0.1, tactical_confusion=25.0)
        explanation = _make_explanation(top_impact=1.0, second_impact=0.5)
        suggestions = generate_scenario_suggestions(comparison, explanation)
        categories = [s.category for s in suggestions]
        assert len(categories) == len(set(categories))

    def test_max_three_suggestions(self) -> None:
        comparison = _make_comparison(points_diff=0.1, tactical_confusion=25.0)
        explanation = _make_explanation(top_impact=1.0, second_impact=0.5)
        suggestions = generate_scenario_suggestions(comparison, explanation)
        assert len(suggestions) <= 3

    def test_ordered_by_priority(self) -> None:
        comparison = _make_comparison(points_diff=0.1, tactical_confusion=25.0)
        explanation = _make_explanation(top_impact=1.0, second_impact=0.5)
        suggestions = generate_scenario_suggestions(comparison, explanation)
        priorities = [s.priority for s in suggestions]
        assert priorities == sorted(priorities)

    def test_signal_source_is_populated(self) -> None:
        comparison = _make_comparison(points_diff=0.3)
        explanation = _make_explanation()
        suggestions = generate_scenario_suggestions(comparison, explanation)
        for s in suggestions:
            assert s.signal_source != ""


class TestRenderSuggestionsMarkdown:
    def test_empty_returns_empty(self) -> None:
        assert render_suggestions_markdown(()) == ""

    def test_contains_next_steps_heading(self) -> None:
        comparison = _make_comparison(points_diff=0.3)
        explanation = _make_explanation()
        suggestions = generate_scenario_suggestions(comparison, explanation)
        md = render_suggestions_markdown(suggestions)
        assert "## Next Steps" in md

    def test_numbered_list(self) -> None:
        comparison = _make_comparison(points_diff=0.1, tactical_confusion=25.0)
        explanation = _make_explanation(top_impact=1.0, second_impact=0.5)
        suggestions = generate_scenario_suggestions(comparison, explanation)
        md = render_suggestions_markdown(suggestions)
        assert "1." in md
        assert "2." in md
