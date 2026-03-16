"""Tests for validation_signals module."""

from __future__ import annotations

from iffootball.simulation.structured_explanation import (
    CausalStep,
    DifferenceHighlight,
    EvidenceItem,
    LimitationsDisclosure,
    PlayerImpactChange,
    PlayerImpactSummary,
    ScenarioDescriptor,
    StructuredExplanation,
    ValidationSignal,
)
from iffootball.simulation.validation_signals import (
    generate_validation_signals,
    render_signals_markdown,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_explanation(
    *,
    n_steps: int = 3,
    n_players: int = 2,
) -> StructuredExplanation:
    """Build a minimal StructuredExplanation for signal testing."""
    scenario = ScenarioDescriptor(
        trigger_type="manager_change",
        team_name="Test FC",
        detail={
            "outgoing_manager": "Old Manager",
            "incoming_manager": "New Manager",
        },
    )

    steps: list[CausalStep] = []
    event_types = ["tactical_confusion", "form_drop", "adaptation_progress"]
    agents = ["Player A", "Player B", "Player C"]
    for i in range(n_steps):
        steps.append(
            CausalStep(
                step_id=f"cs-{i + 1:03d}",
                cause="test cause",
                effect="test effect",
                affected_agent=agents[i % len(agents)],
                event_type=event_types[i % len(event_types)],
                evidence=(
                    EvidenceItem(
                        statement="test",
                        label="data",
                        source="simulation_output",
                    ),
                ),
                depth=min(i + 1, 3),
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
                            statement="test", label="data", source="simulation_output"
                        ),
                    ),
                    PlayerImpactChange(
                        axis="trust",
                        diff=0.12 if i == 0 else -0.08,
                        interpretation=EvidenceItem(
                            statement="test", label="data", source="simulation_output"
                        ),
                    ),
                ),
                related_step_ids=(f"cs-{i + 1:03d}",),
            )
        )

    return StructuredExplanation(
        scenario=scenario,
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
        ),
        causal_chain=tuple(steps),
        player_impacts=tuple(players),
        limitations=LimitationsDisclosure(system=(), scenario=()),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGenerateValidationSignals:
    def test_returns_signals_for_known_event_types(self) -> None:
        exp = _make_explanation(n_steps=3)
        signals = generate_validation_signals(exp)

        assert len(signals) > 0
        assert len(signals) <= 3

    def test_max_signals_respected(self) -> None:
        exp = _make_explanation(n_steps=5)
        signals = generate_validation_signals(exp, max_signals=2)

        assert len(signals) <= 2

    def test_confidence_from_depth(self) -> None:
        exp = _make_explanation(n_steps=3)
        signals = generate_validation_signals(exp)

        # First step is depth 1 -> high confidence.
        depth_1_signals = [s for s in signals if s.confidence == "high"]
        assert len(depth_1_signals) >= 1

    def test_signals_have_required_fields(self) -> None:
        exp = _make_explanation()
        signals = generate_validation_signals(exp)

        for signal in signals:
            assert signal.metric
            assert signal.observation_window
            assert signal.metric_direction in ("increase", "decrease", "stable")
            assert signal.hypothesis_support in ("supports", "contradicts")
            assert signal.reason
            assert signal.confidence in ("high", "medium", "low")

    def test_empty_causal_chain_returns_empty(self) -> None:
        exp = _make_explanation(n_steps=0, n_players=0)
        signals = generate_validation_signals(exp)

        assert signals == ()

    def test_sorted_by_confidence(self) -> None:
        exp = _make_explanation(n_steps=3)
        signals = generate_validation_signals(exp)

        confidence_order = {"high": 0, "medium": 1, "low": 2}
        for i in range(len(signals) - 1):
            assert (
                confidence_order[signals[i].confidence]
                <= confidence_order[signals[i + 1].confidence]
            )

    def test_related_step_id_links_to_causal_chain(self) -> None:
        exp = _make_explanation(n_steps=3)
        signals = generate_validation_signals(exp)
        step_ids = {s.step_id for s in exp.causal_chain}

        for signal in signals:
            if signal.related_step_id is not None:
                assert signal.related_step_id in step_ids


class TestRenderSignalsMarkdown:
    def test_renders_section_heading(self) -> None:
        signals = (
            ValidationSignal(
                metric="PPDA",
                observation_window="first 3 matches",
                metric_direction="increase",
                hypothesis_support="supports",
                reason="Test reason",
                related_step_id="cs-001",
                confidence="high",
            ),
        )
        md = render_signals_markdown(signals)
        assert "## What to Watch" in md
        assert "PPDA" in md
        assert "high confidence" in md

    def test_empty_signals_returns_empty(self) -> None:
        assert render_signals_markdown(()) == ""

    def test_confidence_icons(self) -> None:
        signals = (
            ValidationSignal(
                metric="A", observation_window="w", metric_direction="increase",
                hypothesis_support="supports", reason="r",
                related_step_id=None, confidence="high",
            ),
            ValidationSignal(
                metric="B", observation_window="w", metric_direction="decrease",
                hypothesis_support="supports", reason="r",
                related_step_id=None, confidence="medium",
            ),
            ValidationSignal(
                metric="C", observation_window="w", metric_direction="stable",
                hypothesis_support="supports", reason="r",
                related_step_id=None, confidence="low",
            ),
        )
        md = render_signals_markdown(signals)
        assert "●" in md   # high
        assert "◐" in md   # medium
        assert "○" in md   # low


class TestPlannerIntegration:
    def test_standard_plan_includes_signals(self) -> None:
        from iffootball.simulation.report_planner import (
            DisplayContext,
            plan_report,
        )

        exp = _make_explanation()
        plan = plan_report(exp, DisplayContext.STANDARD)

        assert len(plan.validation_signals) > 0
        assert "what_to_watch" in plan.to_display_hints().section_order

    def test_compact_plan_excludes_signals(self) -> None:
        from iffootball.simulation.report_planner import (
            DisplayContext,
            plan_report,
        )

        exp = _make_explanation()
        plan = plan_report(exp, DisplayContext.COMPACT)

        assert len(plan.validation_signals) == 0
        assert "what_to_watch" not in plan.to_display_hints().section_order
