"""Tests for player impact ranking."""

from __future__ import annotations

from iffootball.agents.fixture import Fixture, FixtureList, OpponentStrength
from iffootball.agents.manager import ManagerAgent
from iffootball.agents.player import BroadPosition, PlayerAgent, RoleFamily
from iffootball.agents.team import TeamBaseline
from iffootball.agents.trigger import ManagerChangeTrigger
from iffootball.config import (
    AdaptationConfig,
    ManagerTurningPointConfig,
    MatchConfig,
    PlayerTurningPointConfig,
    SimulationRules,
    TurningPointConfig,
)
from iffootball.simulation.comparison import ComparisonResult, run_comparison
from iffootball.simulation.turning_point import RuleBasedHandler
from iffootball.visualization.player_impact import (
    PlayerImpact,
    rank_player_impact,
)

# ---------------------------------------------------------------------------
# Helpers (same fixtures/squad as test_radar_data for consistency)
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
    squad.append(PlayerAgent(12, "Player 12", "Center Back", RoleFamily.CENTER_BACK, BroadPosition.DF, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0))
    squad.append(PlayerAgent(13, "Player 13", "Center Defensive Midfield", RoleFamily.DEFENSIVE_MIDFIELDER, BroadPosition.MF, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0))
    squad.append(PlayerAgent(14, "Player 14", "Center Forward", RoleFamily.FORWARD, BroadPosition.FW, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0))
    return squad


def _make_rules() -> SimulationRules:
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
        ),
        match=MatchConfig(home_advantage_factor=1.1),
    )


def _make_comparison() -> ComparisonResult:
    rules = _make_rules()
    incoming = ManagerAgent(
        manager_name="New Manager",
        team_name="Other",
        competition_id=2,
        season_id=2,
        tenure_match_ids=frozenset(),
        pressing_intensity=70.0,
        possession_preference=0.65,
        counter_tendency=0.35,
        preferred_formation="3-5-2",
    )
    trigger = ManagerChangeTrigger(
        outgoing_manager_name="Original Manager",
        incoming_manager_name="New Manager",
        transition_type="mid_season",
        applied_at=10,
        incoming_profile=incoming,
    )
    return run_comparison(
        team=TeamBaseline(
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
        ),
        squad=_make_squad(),
        manager=ManagerAgent(
            manager_name="Original Manager",
            team_name="Team A",
            competition_id=1,
            season_id=1,
            tenure_match_ids=frozenset(),
            pressing_intensity=55.0,
            possession_preference=0.55,
            counter_tendency=0.45,
            preferred_formation="4-3-3",
        ),
        fixture_list=FixtureList(
            team_name="Team A",
            trigger_week=10,
            fixtures=(
                Fixture(match_week=11, opponent_name="Opp A", is_home=True),
                Fixture(match_week=12, opponent_name="Opp B", is_home=False),
                Fixture(match_week=13, opponent_name="Opp C", is_home=True),
            ),
        ),
        opponent_strengths={
            "Opp A": OpponentStrength("Opp A", 1.2, 1.1, 1500.0),
            "Opp B": OpponentStrength("Opp B", 1.0, 1.3, 1450.0),
            "Opp C": OpponentStrength("Opp C", 1.4, 0.9, 1520.0),
        },
        rules=rules,
        handler=RuleBasedHandler(rules),
        trigger=trigger,
        n_runs=5,
        rng_seed=42,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRankPlayerImpact:
    def test_returns_list_of_player_impact(self) -> None:
        comparison = _make_comparison()
        results = rank_player_impact(comparison, top_n=5)
        assert isinstance(results, list)
        for item in results:
            assert isinstance(item, PlayerImpact)

    def test_respects_top_n(self) -> None:
        comparison = _make_comparison()
        results = rank_player_impact(comparison, top_n=3)
        assert len(results) <= 3

    def test_sorted_by_descending_impact(self) -> None:
        comparison = _make_comparison()
        results = rank_player_impact(comparison, top_n=10)
        for i in range(len(results) - 1):
            assert results[i].impact_score >= results[i + 1].impact_score

    def test_tiebreaker_by_player_id(self) -> None:
        comparison = _make_comparison()
        results = rank_player_impact(comparison, top_n=14)
        # Among players with equal impact, player_id should be ascending.
        for i in range(len(results) - 1):
            if results[i].impact_score == results[i + 1].impact_score:
                assert results[i].player_id < results[i + 1].player_id

    def test_impact_score_non_negative(self) -> None:
        comparison = _make_comparison()
        results = rank_player_impact(comparison, top_n=14)
        for item in results:
            assert item.impact_score >= 0.0

    def test_mean_states_in_valid_range(self) -> None:
        comparison = _make_comparison()
        results = rank_player_impact(comparison, top_n=5)
        for item in results:
            assert 0.0 <= item.mean_form_a <= 1.0
            assert 0.0 <= item.mean_form_b <= 1.0
            assert 0.0 <= item.mean_fatigue_a <= 1.0
            assert 0.0 <= item.mean_fatigue_b <= 1.0
            assert 0.0 <= item.mean_understanding_a <= 1.0
            assert 0.0 <= item.mean_understanding_b <= 1.0
            assert 0.0 <= item.mean_trust_a <= 1.0
            assert 0.0 <= item.mean_trust_b <= 1.0

    def test_has_nonzero_impact_with_trigger(self) -> None:
        comparison = _make_comparison()
        results = rank_player_impact(comparison, top_n=1)
        # With a manager change trigger, at least the top player
        # should show some state difference.
        assert len(results) >= 1
        assert results[0].impact_score > 0.0

    def test_player_id_used_for_matching(self) -> None:
        comparison = _make_comparison()
        results = rank_player_impact(comparison, top_n=14)
        ids = [r.player_id for r in results]
        # All IDs should be unique.
        assert len(ids) == len(set(ids))
