"""Tests for skeleton_builder module."""

from __future__ import annotations

import pytest

from iffootball.agents.trigger import ManagerChangeTrigger, TransferInTrigger
from iffootball.simulation.cascade_tracker import CascadeEvent
from iffootball.simulation.comparison import (
    AggregatedResult,
    ComparisonResult,
    DeltaMetrics,
)
from iffootball.simulation.engine import SimulationResult
from iffootball.simulation.skeleton_builder import build_skeleton
from iffootball.visualization.player_impact import PlayerImpact


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cascade_event(
    *,
    week: int = 5,
    event_type: str = "form_drop",
    affected_agent: str = "Player A",
    cause_chain: tuple[str, ...] = (),
    magnitude: float = 0.3,
    depth: int = 1,
) -> CascadeEvent:
    return CascadeEvent(
        week=week,
        event_type=event_type,
        affected_agent=affected_agent,
        cause_chain=cause_chain,
        magnitude=magnitude,
        depth=depth,
    )


def _make_simulation_result(
    cascade_events: list[CascadeEvent] | None = None,
) -> SimulationResult:
    return SimulationResult(
        match_results=[],
        cascade_events=cascade_events or [],
        final_squad=[],
        final_manager=None,  # type: ignore[arg-type]
    )


def _make_comparison(
    events_b: list[CascadeEvent] | None = None,
    n_runs: int = 1,
    points_a: float = 50.0,
    points_b: float = 45.0,
) -> ComparisonResult:
    events = events_b or []
    run_a = _make_simulation_result()
    run_b = _make_simulation_result(events)

    agg_a = AggregatedResult(
        n_runs=n_runs,
        total_points_mean=points_a,
        total_points_median=points_a,
        total_points_std=2.0,
        cascade_event_counts={},
        run_results=tuple([run_a] * n_runs),
    )

    # Count events for branch B.
    event_counts: dict[str, float] = {}
    for ev in events:
        event_counts[ev.event_type] = event_counts.get(ev.event_type, 0) + 1.0
    for k in event_counts:
        event_counts[k] /= n_runs

    agg_b = AggregatedResult(
        n_runs=n_runs,
        total_points_mean=points_b,
        total_points_median=points_b,
        total_points_std=3.0,
        cascade_event_counts=event_counts,
        run_results=tuple([run_b] * n_runs),
    )

    delta = DeltaMetrics(
        points_mean_diff=points_b - points_a,
        points_median_diff=points_b - points_a,
        cascade_count_diff=event_counts,
    )

    return ComparisonResult(no_change=agg_a, with_change=agg_b, delta=delta)


def _make_player_impact(
    *,
    player_name: str = "Player A",
    impact_score: float = 0.3,
    form_diff: float = -0.1,
) -> PlayerImpact:
    return PlayerImpact(
        player_id=1,
        player_name=player_name,
        impact_score=impact_score,
        mean_form_a=0.7,
        mean_form_b=0.7 + form_diff,
        mean_fatigue_a=0.4,
        mean_fatigue_b=0.4,
        mean_understanding_a=0.6,
        mean_understanding_b=0.6,
        mean_trust_a=0.8,
        mean_trust_b=0.8,
    )


def _make_trigger() -> ManagerChangeTrigger:
    return ManagerChangeTrigger(
        outgoing_manager_name="Original Manager",
        incoming_manager_name="New Manager",
        transition_type="mid_season",
        applied_at=10,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildSkeleton:
    def test_scenario_descriptor_from_manager_change(self) -> None:
        comparison = _make_comparison()
        trigger = _make_trigger()
        result = build_skeleton(comparison, trigger, "Arsenal", [])

        assert result.scenario.trigger_type == "manager_change"
        assert result.scenario.team_name == "Arsenal"
        assert result.scenario.detail["outgoing_manager"] == "Original Manager"
        assert result.scenario.detail["incoming_manager"] == "New Manager"

    def test_scenario_descriptor_from_transfer_in(self) -> None:
        comparison = _make_comparison()
        trigger = TransferInTrigger(
            player_name="Florian Wirtz",
            expected_role="starter",
            applied_at=10,
        )
        result = build_skeleton(comparison, trigger, "Arsenal", [])

        assert result.scenario.trigger_type == "player_transfer_in"
        assert result.scenario.detail["player_name"] == "Florian Wirtz"

    def test_highlights_include_points_diff(self) -> None:
        comparison = _make_comparison(points_a=50.0, points_b=45.0)
        result = build_skeleton(comparison, _make_trigger(), "Team A", [])

        assert len(result.highlights) >= 1
        points_hl = result.highlights[0]
        assert points_hl.metric_name == "total_points_mean"
        assert points_hl.diff == -5.0
        assert points_hl.interpretations[0].label == "data"
        assert points_hl.interpretations[0].source == "simulation_output"
        # Statement is empty in skeleton.
        assert points_hl.interpretations[0].statement == ""

    def test_highlights_include_cascade_events(self) -> None:
        events = [
            _make_cascade_event(event_type="form_drop", magnitude=0.4),
            _make_cascade_event(event_type="trust_decline", magnitude=0.2),
        ]
        comparison = _make_comparison(events_b=events)
        result = build_skeleton(comparison, _make_trigger(), "Team A", [])

        # Points + 2 event types = 3 highlights.
        assert len(result.highlights) == 3

    def test_causal_chain_from_cascade_events(self) -> None:
        events = [
            _make_cascade_event(
                event_type="form_drop",
                affected_agent="Player A",
                depth=1,
                magnitude=0.4,
            ),
            _make_cascade_event(
                event_type="trust_decline",
                affected_agent="Player B",
                cause_chain=("form_drop",),
                depth=2,
                magnitude=0.2,
            ),
        ]
        comparison = _make_comparison(events_b=events)
        result = build_skeleton(comparison, _make_trigger(), "Team A", [])

        assert len(result.causal_chain) == 2
        assert result.causal_chain[0].step_id == "cs-001"
        assert result.causal_chain[0].affected_agent == "Player A"
        assert result.causal_chain[0].event_type == "form_drop"
        assert result.causal_chain[0].depth == 1
        assert result.causal_chain[0].cause == ""  # skeleton
        assert result.causal_chain[0].effect == ""  # skeleton

        assert result.causal_chain[1].step_id == "cs-002"
        assert result.causal_chain[1].affected_agent == "Player B"
        assert result.causal_chain[1].depth == 2

    def test_causal_chain_label_inference(self) -> None:
        events = [
            _make_cascade_event(depth=1, magnitude=0.4),
            _make_cascade_event(
                event_type="trust_decline",
                affected_agent="Player B",
                depth=3,
                magnitude=0.2,
            ),
        ]
        comparison = _make_comparison(events_b=events)
        result = build_skeleton(comparison, _make_trigger(), "Team A", [])

        # depth 1 -> simulation_output -> data
        assert result.causal_chain[0].evidence[0].label == "data"
        assert result.causal_chain[0].evidence[0].source == "simulation_output"
        # depth 3 -> rule_based_model -> hypothesis
        assert result.causal_chain[1].evidence[0].label == "hypothesis"
        assert result.causal_chain[1].evidence[0].source == "rule_based_model"

    def test_player_impacts_with_axis_info(self) -> None:
        impacts = [_make_player_impact(form_diff=-0.15)]
        comparison = _make_comparison()
        result = build_skeleton(comparison, _make_trigger(), "Team A", impacts)

        assert len(result.player_impacts) == 1
        pi = result.player_impacts[0]
        assert pi.player_name == "Player A"
        assert len(pi.changes) == 4  # form, fatigue, understanding, trust

        form_change = pi.changes[0]
        assert form_change.axis == "form"
        assert form_change.diff == -0.15
        assert form_change.interpretation.statement == ""  # skeleton

    def test_player_impacts_related_step_ids(self) -> None:
        events = [
            _make_cascade_event(
                affected_agent="Player A",
                depth=1,
                magnitude=0.4,
            ),
        ]
        impacts = [_make_player_impact(player_name="Player A")]
        comparison = _make_comparison(events_b=events)
        result = build_skeleton(comparison, _make_trigger(), "Team A", impacts)

        pi = result.player_impacts[0]
        assert len(pi.related_step_ids) == 1
        assert pi.related_step_ids[0] == "cs-001"

    def test_player_impacts_no_related_steps(self) -> None:
        impacts = [_make_player_impact(player_name="Player X")]
        comparison = _make_comparison()
        result = build_skeleton(comparison, _make_trigger(), "Team A", impacts)

        assert result.player_impacts[0].related_step_ids == ()

    def test_confidence_notes_generated(self) -> None:
        events = [
            _make_cascade_event(depth=4, magnitude=0.3),
        ]
        comparison = _make_comparison(events_b=events)
        result = build_skeleton(comparison, _make_trigger(), "Team A", [])

        assert len(result.confidence_notes) >= 1
        assert any("depth" in n.lower() for n in result.confidence_notes)

    def test_empty_events_produce_empty_chain(self) -> None:
        comparison = _make_comparison(events_b=[])
        result = build_skeleton(comparison, _make_trigger(), "Team A", [])

        assert result.causal_chain == ()
        assert result.confidence_notes == ()

    def test_deduplication_same_path(self) -> None:
        # Same event_type + agent + depth + cause_chain from multiple runs.
        events = [
            _make_cascade_event(
                affected_agent="Player A", depth=1, magnitude=0.3
            ),
            _make_cascade_event(
                affected_agent="Player A", depth=1, magnitude=0.5
            ),
        ]
        comparison = _make_comparison(events_b=events)
        result = build_skeleton(comparison, _make_trigger(), "Team A", [])

        # Deduplicated to 1 step (keeps higher magnitude).
        assert len(result.causal_chain) == 1
        assert result.causal_chain[0].affected_agent == "Player A"

    def test_deduplication_preserves_different_causal_paths(self) -> None:
        # Same event_type + agent + depth, but different cause_chain.
        events = [
            _make_cascade_event(
                affected_agent="Player A",
                event_type="form_drop",
                depth=2,
                cause_chain=("trust_decline",),
                magnitude=0.3,
            ),
            _make_cascade_event(
                affected_agent="Player A",
                event_type="form_drop",
                depth=2,
                cause_chain=("tactical_confusion",),
                magnitude=0.4,
            ),
        ]
        comparison = _make_comparison(events_b=events)
        result = build_skeleton(comparison, _make_trigger(), "Team A", [])

        # Different causal paths preserved as separate steps.
        assert len(result.causal_chain) == 2

    def test_unsupported_trigger_raises(self) -> None:
        comparison = _make_comparison()

        class FakeTrigger:
            trigger_type = "fake"

        with pytest.raises(TypeError, match="Unsupported trigger type"):
            build_skeleton(
                comparison, FakeTrigger(), "Team A", []  # type: ignore[arg-type]
            )
