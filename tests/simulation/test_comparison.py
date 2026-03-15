"""Tests for Branch A/B comparison."""

from __future__ import annotations

import pytest

from iffootball.agents.fixture import Fixture, FixtureList, OpponentStrength
from iffootball.agents.manager import ManagerAgent
from iffootball.agents.player import BroadPosition, PlayerAgent, RoleFamily
from iffootball.agents.team import TeamBaseline
from iffootball.agents.trigger import ManagerChangeTrigger
from iffootball.config import (
    ActionDistributionConfig,
    AdaptationConfig,
    MatchConfig,
    ManagerTurningPointConfig,
    PlayerTurningPointConfig,
    SimulationRules,
    TurningPointConfig,
)
from iffootball.simulation.comparison import (
    ComparisonResult,
    run_comparison,
)
from iffootball.simulation.turning_point import RuleBasedHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_433_POSITIONS = [
    ("Goalkeeper", RoleFamily.GOALKEEPER, BroadPosition.GK),
    ("Right Back", RoleFamily.FULL_BACK, BroadPosition.DF),
    ("Right Center Back", RoleFamily.CENTER_BACK, BroadPosition.DF),
    ("Left Center Back", RoleFamily.CENTER_BACK, BroadPosition.DF),
    ("Left Back", RoleFamily.FULL_BACK, BroadPosition.DF),
    ("Center Defensive Midfield", RoleFamily.DEFENSIVE_MIDFIELDER, BroadPosition.MF),
    ("Right Center Midfield", RoleFamily.CENTRAL_MIDFIELDER, BroadPosition.MF),
    ("Left Center Midfield", RoleFamily.CENTRAL_MIDFIELDER, BroadPosition.MF),
    ("Right Wing", RoleFamily.WINGER, BroadPosition.MF),
    ("Left Wing", RoleFamily.WINGER, BroadPosition.MF),
    ("Center Forward", RoleFamily.FORWARD, BroadPosition.FW),
]


def _rules() -> SimulationRules:
    return SimulationRules(
        adaptation=AdaptationConfig(
            base_fatigue_increase=0.05,
            base_fatigue_recovery=0.03,
            tactical_understanding_gain=0.04,
            fatigue_penalty_weight=0.5,
            trust_increase_on_start=0.02,
            trust_decrease_on_bench=0.01,
        ),
        turning_points=TurningPointConfig(
            player=PlayerTurningPointConfig(
                bench_streak_threshold=3,
                tactical_understanding_low=0.40,
                short_term_window=4,
                trust_low=0.40,
            ),
            manager=ManagerTurningPointConfig(
                job_security_warning=0.30,
                job_security_critical=0.10,
                style_stubbornness_threshold=80,
            ),
            action_distribution=ActionDistributionConfig(
                bench_streak_low_trust={"resist": 0.6, "adapt": 0.3, "transfer": 0.1},
                low_understanding={"adapt": 0.5, "resist": 0.4, "transfer": 0.1},
                default={"adapt": 0.8, "resist": 0.2, "transfer": 0.0},
            ),
        ),
        match=MatchConfig(home_advantage_factor=1.1),
    )


def _make_squad() -> list[PlayerAgent]:
    squad: list[PlayerAgent] = []
    for i, (pos_name, role, broad) in enumerate(_433_POSITIONS):
        squad.append(
            PlayerAgent(
                player_id=i + 1,
                player_name=f"Player {i + 1}",
                position_name=pos_name,
                role_family=role,
                broad_position=broad,
                pace=50.0,
                passing=50.0,
                shooting=50.0,
                pressing=50.0,
                defending=50.0,
                physicality=50.0,
                consistency=50.0,
            )
        )
    # 3 subs
    squad.append(PlayerAgent(12, "Player 12", "Center Back", RoleFamily.CENTER_BACK, BroadPosition.DF, 50, 50, 50, 50, 50, 50, 50))
    squad.append(PlayerAgent(13, "Player 13", "Center Defensive Midfield", RoleFamily.DEFENSIVE_MIDFIELDER, BroadPosition.MF, 50, 50, 50, 50, 50, 50, 50))
    squad.append(PlayerAgent(14, "Player 14", "Center Forward", RoleFamily.FORWARD, BroadPosition.FW, 50, 50, 50, 50, 50, 50, 50))
    return squad


def _make_manager() -> ManagerAgent:
    return ManagerAgent(
        manager_name="Original Manager",
        team_name="Team A",
        competition_id=1,
        season_id=1,
        tenure_match_ids=frozenset(),
        pressing_intensity=55.0,
        possession_preference=0.55,
        counter_tendency=0.45,
        preferred_formation="4-3-3",
    )


def _make_team() -> TeamBaseline:
    return TeamBaseline(
        team_name="Team A",
        competition_id=1,
        season_id=1,
        played_match_ids=frozenset({1, 2}),
        xg_for_per90=1.5,
        xg_against_per90=1.0,
        ppda=10.0,
        progressive_passes_per90=50.0,
        possession_pct=0.55,
        league_position=5,
        points_to_safety=10,
        points_to_title=-5,
        matches_remaining=3,
    )


def _make_fixture_list() -> FixtureList:
    return FixtureList(
        team_name="Team A",
        trigger_week=10,
        fixtures=(
            Fixture(match_week=11, opponent_name="Opp A", is_home=True),
            Fixture(match_week=12, opponent_name="Opp B", is_home=False),
            Fixture(match_week=13, opponent_name="Opp C", is_home=True),
        ),
    )


def _make_opponent_strengths() -> dict[str, OpponentStrength]:
    return {
        "Opp A": OpponentStrength("Opp A", 1.2, 1.1, 1500.0),
        "Opp B": OpponentStrength("Opp B", 1.0, 1.3, 1450.0),
        "Opp C": OpponentStrength("Opp C", 1.4, 0.9, 1520.0),
    }


def _make_trigger() -> ManagerChangeTrigger:
    return ManagerChangeTrigger(
        outgoing_manager_name="Original Manager",
        incoming_manager_name="New Manager",
        transition_type="mid_season",
        applied_at=10,
    )


def _run(n_runs: int = 3, seed: int = 42) -> ComparisonResult:
    return run_comparison(
        team=_make_team(),
        squad=_make_squad(),
        manager=_make_manager(),
        fixture_list=_make_fixture_list(),
        opponent_strengths=_make_opponent_strengths(),
        rules=_rules(),
        handler=RuleBasedHandler(_rules()),
        trigger=_make_trigger(),
        n_runs=n_runs,
        rng_seed=seed,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunComparison:
    def test_returns_comparison_result(self) -> None:
        result = _run(n_runs=1)
        assert isinstance(result, ComparisonResult)

    def test_n_runs_count(self) -> None:
        result = _run(n_runs=5)
        assert result.no_change.n_runs == 5
        assert result.with_change.n_runs == 5
        assert len(result.no_change.run_results) == 5
        assert len(result.with_change.run_results) == 5

    def test_reproducible_with_same_seed(self) -> None:
        r1 = _run(n_runs=3, seed=123)
        r2 = _run(n_runs=3, seed=123)
        assert r1.no_change.total_points_mean == r2.no_change.total_points_mean
        assert r1.with_change.total_points_mean == r2.with_change.total_points_mean
        assert r1.delta.points_mean_diff == r2.delta.points_mean_diff

    def test_different_seeds_may_differ(self) -> None:
        results = [_run(n_runs=3, seed=s) for s in range(10)]
        means = [r.no_change.total_points_mean for r in results]
        assert len(set(means)) > 1

    def test_trigger_affects_branch_b(self) -> None:
        result = _run(n_runs=5)
        # Branch B has a manager change; manager name should be "New Manager"
        for sr in result.with_change.run_results:
            assert sr.final_manager.manager_name == "New Manager"
        # Branch A keeps original manager
        for sr in result.no_change.run_results:
            assert sr.final_manager.manager_name == "Original Manager"

    def test_match_results_per_run(self) -> None:
        result = _run(n_runs=2)
        for sr in result.no_change.run_results:
            assert len(sr.match_results) == 3  # 3 fixtures

    def test_points_stats_valid(self) -> None:
        result = _run(n_runs=10)
        agg = result.no_change
        assert agg.total_points_mean >= 0
        assert agg.total_points_median >= 0
        assert agg.total_points_std >= 0


class TestDeltaMetrics:
    def test_delta_is_b_minus_a(self) -> None:
        result = _run(n_runs=5)
        diff = result.delta.points_mean_diff
        expected = (
            result.with_change.total_points_mean
            - result.no_change.total_points_mean
        )
        assert diff == pytest.approx(expected)

    def test_cascade_count_diff_union(self) -> None:
        """cascade_count_diff should include all event types from both branches."""
        result = _run(n_runs=10)
        all_types_a = set(result.no_change.cascade_event_counts)
        all_types_b = set(result.with_change.cascade_event_counts)
        expected_union = all_types_a | all_types_b
        assert set(result.delta.cascade_count_diff) == expected_union

    def test_cascade_diff_values(self) -> None:
        result = _run(n_runs=5)
        for et, diff in result.delta.cascade_count_diff.items():
            val_a = result.no_change.cascade_event_counts.get(et, 0.0)
            val_b = result.with_change.cascade_event_counts.get(et, 0.0)
            assert diff == pytest.approx(val_b - val_a)


class TestInputIsolation:
    def test_original_squad_not_mutated(self) -> None:
        """The original squad passed to run_comparison should not be mutated."""
        squad = _make_squad()
        original_forms = [p.current_form for p in squad]
        _run_with_squad(squad, n_runs=3)
        after_forms = [p.current_form for p in squad]
        assert original_forms == after_forms


def _run_with_squad(
    squad: list[PlayerAgent], n_runs: int
) -> ComparisonResult:
    return run_comparison(
        team=_make_team(),
        squad=squad,
        manager=_make_manager(),
        fixture_list=_make_fixture_list(),
        opponent_strengths=_make_opponent_strengths(),
        rules=_rules(),
        handler=RuleBasedHandler(_rules()),
        trigger=_make_trigger(),
        n_runs=n_runs,
        rng_seed=42,
    )
