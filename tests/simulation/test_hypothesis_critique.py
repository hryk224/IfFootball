"""Tests for hypothesis generation with mandatory critique."""

from __future__ import annotations

import pytest

from iffootball.simulation.comparison import (
    AggregatedResult,
    ComparisonResult,
    DeltaMetrics,
)
from iffootball.simulation.hypothesis_critique import (
    CritiquePoint,
    HypothesisCritique,
    generate_hypothesis_critiques,
    render_critiques_markdown,
)
from iffootball.simulation.structured_explanation import (
    DifferenceHighlight,
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
    points_std: float = 3.0,
    tactical_confusion: float = 20.0,
    form_drop: float = 30.0,
) -> ComparisonResult:
    cascade_b = {
        "tactical_confusion": tactical_confusion,
        "form_drop": form_drop,
    }
    return ComparisonResult(
        no_change=AggregatedResult(
            n_runs=20,
            total_points_mean=12.0,
            total_points_median=12.0,
            total_points_std=points_std,
            cascade_event_counts={},
            run_results=(),
        ),
        with_change=AggregatedResult(
            n_runs=20,
            total_points_mean=12.0 + points_diff,
            total_points_median=12.0 + points_diff,
            total_points_std=points_std,
            cascade_event_counts=cascade_b,
            run_results=(),
        ),
        delta=DeltaMetrics(
            points_mean_diff=points_diff,
            points_median_diff=points_diff,
            cascade_count_diff={},
        ),
    )


def _make_explanation(
    top_impact: float = 0.8,
    second_impact: float = 0.5,
    has_highlights: bool = True,
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

    highlights = ()
    if has_highlights:
        highlights = (
            DifferenceHighlight(
                metric_name="total_points_mean",
                value_a=12.0,
                value_b=12.5,
                diff=0.5,
                interpretations=(
                    EvidenceItem(
                        statement="", label="data", source="simulation_output"
                    ),
                ),
            ),
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
        highlights=highlights,
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


class TestGenerateHypothesisCritiques:
    def test_generates_main_diff_hypothesis(self) -> None:
        comparison = _make_comparison(points_diff=0.5)
        explanation = _make_explanation()
        critiques = generate_hypothesis_critiques(comparison, explanation)
        claims = [hc.claim for hc in critiques]
        assert any("total_points_mean" in c for c in claims)

    def test_generates_cascade_hypothesis(self) -> None:
        comparison = _make_comparison(tactical_confusion=25.0)
        explanation = _make_explanation()
        critiques = generate_hypothesis_critiques(comparison, explanation)
        claims = [hc.claim for hc in critiques]
        assert any("tactical_confusion" in c or "form_drop" in c for c in claims)

    def test_no_cascade_hypothesis_when_low(self) -> None:
        comparison = _make_comparison(tactical_confusion=5.0, form_drop=5.0)
        explanation = _make_explanation()
        critiques = generate_hypothesis_critiques(comparison, explanation)
        claims = [hc.claim for hc in critiques]
        assert not any("suppresses" in c for c in claims)

    def test_generates_player_concentration_hypothesis(self) -> None:
        comparison = _make_comparison()
        explanation = _make_explanation(top_impact=1.0, second_impact=0.5)
        critiques = generate_hypothesis_critiques(comparison, explanation)
        claims = [hc.claim for hc in critiques]
        assert any("Player Top" in c for c in claims)

    def test_no_player_hypothesis_when_balanced(self) -> None:
        comparison = _make_comparison()
        explanation = _make_explanation(top_impact=0.8, second_impact=0.7)
        critiques = generate_hypothesis_critiques(comparison, explanation)
        claims = [hc.claim for hc in critiques]
        assert not any("dominate" in c for c in claims)

    def test_max_three_hypotheses(self) -> None:
        comparison = _make_comparison(
            points_diff=0.3, tactical_confusion=25.0, form_drop=40.0
        )
        explanation = _make_explanation(top_impact=1.0, second_impact=0.5)
        critiques = generate_hypothesis_critiques(comparison, explanation)
        assert len(critiques) <= 3

    def test_all_have_at_least_one_critique(self) -> None:
        comparison = _make_comparison(
            points_diff=0.3, tactical_confusion=25.0
        )
        explanation = _make_explanation(top_impact=1.0, second_impact=0.5)
        critiques = generate_hypothesis_critiques(comparison, explanation)
        for hc in critiques:
            assert len(hc.critiques) >= 1

    def test_hypothesis_ids_are_unique(self) -> None:
        comparison = _make_comparison(
            points_diff=0.3, tactical_confusion=25.0
        )
        explanation = _make_explanation(top_impact=1.0, second_impact=0.5)
        critiques = generate_hypothesis_critiques(comparison, explanation)
        ids = [hc.hypothesis_id for hc in critiques]
        assert len(ids) == len(set(ids))

    def test_label_is_weakest(self) -> None:
        """Main diff hypothesis mixes data + analysis; should adopt analysis."""
        comparison = _make_comparison()
        explanation = _make_explanation()
        critiques = generate_hypothesis_critiques(comparison, explanation)
        main = [hc for hc in critiques if "total_points_mean" in hc.claim]
        assert main
        assert main[0].label == "analysis"

    def test_no_highlights_skips_main_hypothesis(self) -> None:
        comparison = _make_comparison()
        explanation = _make_explanation(has_highlights=False)
        critiques = generate_hypothesis_critiques(comparison, explanation)
        claims = [hc.claim for hc in critiques]
        assert not any("total_points_mean" in c for c in claims)

    def test_alternative_critique_when_diff_within_std(self) -> None:
        comparison = _make_comparison(points_diff=0.5, points_std=3.0)
        explanation = _make_explanation()
        critiques = generate_hypothesis_critiques(comparison, explanation)
        main = [hc for hc in critiques if "total_points_mean" in hc.claim]
        if main:
            aspects = [cp.aspect for cp in main[0].critiques]
            assert "alternative" in aspects


class TestHypothesisCritiqueValidation:
    def test_empty_critiques_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one critique"):
            HypothesisCritique(
                hypothesis_id="hyp-001",
                claim="test",
                supporting_evidence="test",
                label="data",
                related_step_ids=(),
                critiques=(),
            )


class TestRenderCritiquesMarkdown:
    def test_empty_returns_empty(self) -> None:
        assert render_critiques_markdown(()) == ""

    def test_contains_heading(self) -> None:
        comparison = _make_comparison()
        explanation = _make_explanation()
        critiques = generate_hypothesis_critiques(comparison, explanation)
        md = render_critiques_markdown(critiques)
        assert "## Hypothesis Review" in md

    def test_contains_label(self) -> None:
        comparison = _make_comparison()
        explanation = _make_explanation()
        critiques = generate_hypothesis_critiques(comparison, explanation)
        md = render_critiques_markdown(critiques)
        assert "[analysis]" in md or "[data]" in md or "[hypothesis]" in md
