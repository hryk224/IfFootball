"""Tests for report-level risk audit."""

from __future__ import annotations

from iffootball.agents.player import SampleTier
from iffootball.simulation.comparison import (
    AggregatedResult,
    ComparisonResult,
    DeltaMetrics,
)
from iffootball.simulation.risk_audit import (
    RiskFlag,
    generate_risk_audit,
    render_risk_audit_markdown,
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
    points_std: float = 3.0,
    form_drop: float = 30.0,
    trust_decline: float = 30.0,
) -> ComparisonResult:
    cascade_b = {"form_drop": form_drop, "trust_decline": trust_decline}
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
    top_tier: SampleTier = SampleTier.FULL,
    second_tier: SampleTier = SampleTier.FULL,
) -> StructuredExplanation:
    def _pi(name: str, score: float, tier: SampleTier) -> PlayerImpactSummary:
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
            sample_tier=tier,
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
            _pi("Player Top", 0.8, top_tier),
            _pi("Player Second", 0.5, second_tier),
            _pi("Player Third", 0.3, SampleTier.FULL),
        ),
        limitations=LimitationsDisclosure(system=(), scenario=()),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGenerateRiskAudit:
    def test_overstatement_high_when_very_small_ratio(self) -> None:
        comparison = _make_comparison(points_diff=0.3, points_std=3.0)
        flags = generate_risk_audit(comparison, _make_explanation())
        overstatement = [f for f in flags if f.category == "overstatement"]
        assert len(overstatement) == 1
        assert overstatement[0].severity == "high"

    def test_overstatement_medium_when_moderate_ratio(self) -> None:
        comparison = _make_comparison(points_diff=1.0, points_std=3.0)
        flags = generate_risk_audit(comparison, _make_explanation())
        overstatement = [f for f in flags if f.category == "overstatement"]
        assert len(overstatement) == 1
        assert overstatement[0].severity == "medium"

    def test_no_overstatement_when_large_ratio(self) -> None:
        comparison = _make_comparison(points_diff=3.0, points_std=3.0)
        flags = generate_risk_audit(comparison, _make_explanation())
        categories = [f.category for f in flags]
        assert "overstatement" not in categories

    def test_unstable_basis_high_when_very_high_cascade(self) -> None:
        comparison = _make_comparison(form_drop=50.0, trust_decline=50.0)
        flags = generate_risk_audit(comparison, _make_explanation())
        unstable = [f for f in flags if f.category == "unstable_basis"]
        assert len(unstable) == 1
        assert unstable[0].severity == "high"

    def test_unstable_basis_medium_when_moderate_cascade(self) -> None:
        comparison = _make_comparison(form_drop=25.0, trust_decline=25.0)
        flags = generate_risk_audit(comparison, _make_explanation())
        unstable = [f for f in flags if f.category == "unstable_basis"]
        assert len(unstable) == 1
        assert unstable[0].severity == "medium"

    def test_no_unstable_basis_when_low_cascade(self) -> None:
        comparison = _make_comparison(form_drop=10.0, trust_decline=10.0)
        flags = generate_risk_audit(comparison, _make_explanation())
        categories = [f.category for f in flags]
        assert "unstable_basis" not in categories

    def test_data_reliability_high_when_top1_partial(self) -> None:
        flags = generate_risk_audit(
            _make_comparison(),
            _make_explanation(top_tier=SampleTier.PARTIAL),
        )
        reliability = [f for f in flags if f.category == "data_reliability"]
        assert len(reliability) == 1
        assert reliability[0].severity == "high"

    def test_data_reliability_medium_when_top3_partial(self) -> None:
        flags = generate_risk_audit(
            _make_comparison(),
            _make_explanation(second_tier=SampleTier.PARTIAL),
        )
        reliability = [f for f in flags if f.category == "data_reliability"]
        assert len(reliability) == 1
        assert reliability[0].severity == "medium"

    def test_no_data_reliability_when_all_full(self) -> None:
        flags = generate_risk_audit(_make_comparison(), _make_explanation())
        categories = [f.category for f in flags]
        assert "data_reliability" not in categories

    def test_max_one_per_category(self) -> None:
        comparison = _make_comparison(
            points_diff=0.1, points_std=3.0,
            form_drop=50.0, trust_decline=50.0,
        )
        explanation = _make_explanation(top_tier=SampleTier.PARTIAL)
        flags = generate_risk_audit(comparison, explanation)
        categories = [f.category for f in flags]
        assert len(categories) == len(set(categories))

    def test_max_three_flags(self) -> None:
        comparison = _make_comparison(
            points_diff=0.1, points_std=3.0,
            form_drop=50.0, trust_decline=50.0,
        )
        explanation = _make_explanation(top_tier=SampleTier.PARTIAL)
        flags = generate_risk_audit(comparison, explanation)
        assert len(flags) <= 3

    def test_ordered_by_severity_then_category(self) -> None:
        comparison = _make_comparison(
            points_diff=0.1, points_std=3.0,
            form_drop=50.0, trust_decline=50.0,
        )
        explanation = _make_explanation(top_tier=SampleTier.PARTIAL)
        flags = generate_risk_audit(comparison, explanation)
        severities = [f.severity for f in flags]
        # All high should come before all medium.
        high_indices = [i for i, s in enumerate(severities) if s == "high"]
        medium_indices = [i for i, s in enumerate(severities) if s == "medium"]
        if high_indices and medium_indices:
            assert max(high_indices) < min(medium_indices)

    def test_signal_source_populated(self) -> None:
        comparison = _make_comparison(points_diff=0.1)
        flags = generate_risk_audit(comparison, _make_explanation())
        for f in flags:
            assert f.signal_source != ""


class TestRenderRiskAuditMarkdown:
    def test_empty_returns_empty(self) -> None:
        assert render_risk_audit_markdown(()) == ""

    def test_contains_heading(self) -> None:
        comparison = _make_comparison(points_diff=0.1)
        flags = generate_risk_audit(comparison, _make_explanation())
        md = render_risk_audit_markdown(flags)
        assert "## Risk Review" in md

    def test_contains_severity_icon(self) -> None:
        comparison = _make_comparison(points_diff=0.1)
        flags = generate_risk_audit(comparison, _make_explanation())
        md = render_risk_audit_markdown(flags)
        assert "\u26a0" in md  # warning icon
