"""Tests for the weekly simulation engine."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from iffootball.agents.fixture import Fixture, FixtureList, OpponentStrength
from iffootball.agents.manager import ManagerAgent
from iffootball.agents.player import BroadPosition, PlayerAgent, RoleFamily
from iffootball.agents.team import TeamBaseline
from iffootball.agents.trigger import ManagerChangeTrigger, TransferInTrigger
from iffootball.config import (
    AdaptationConfig,
    ManagerTurningPointConfig,
    MatchConfig,
    PlayerTurningPointConfig,
    SimulationRules,
    TurningPointConfig,
)
from iffootball.simulation.engine import Simulation, SimulationResult
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
        ),
        match=MatchConfig(home_advantage_factor=1.1),
    )


def _make_player(
    player_id: int,
    pos_name: str,
    role: RoleFamily,
    broad: BroadPosition,
) -> PlayerAgent:
    return PlayerAgent(
        player_id=player_id,
        player_name=f"Player {player_id}",
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


def _make_squad() -> list[PlayerAgent]:
    squad: list[PlayerAgent] = []
    for i, (pos_name, role, broad) in enumerate(_433_POSITIONS):
        squad.append(_make_player(i + 1, pos_name, role, broad))
    # 3 subs
    squad.append(_make_player(12, "Center Back", RoleFamily.CENTER_BACK, BroadPosition.DF))
    squad.append(_make_player(13, "Center Defensive Midfield", RoleFamily.DEFENSIVE_MIDFIELDER, BroadPosition.MF))
    squad.append(_make_player(14, "Center Forward", RoleFamily.FORWARD, BroadPosition.FW))
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


def _make_simulation(seed: int = 42) -> Simulation:
    return Simulation(
        team=_make_team(),
        squad=_make_squad(),
        manager=_make_manager(),
        fixture_list=_make_fixture_list(),
        opponent_strengths=_make_opponent_strengths(),
        rules=_rules(),
        handler=RuleBasedHandler(_rules()),
        rng=np.random.default_rng(seed),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSimulationRun:
    def test_returns_simulation_result(self) -> None:
        sim = _make_simulation()
        result = sim.run()
        assert isinstance(result, SimulationResult)

    def test_match_results_count(self) -> None:
        sim = _make_simulation()
        result = sim.run()
        assert len(result.match_results) == 3  # 3 fixtures

    def test_match_results_valid(self) -> None:
        sim = _make_simulation()
        result = sim.run()
        for mr in result.match_results:
            assert mr.home_goals >= 0
            assert mr.away_goals >= 0
            assert mr.points_earned in (0, 1, 3)

    def test_reproducible_with_same_seed(self) -> None:
        r1 = _make_simulation(seed=123).run()
        r2 = _make_simulation(seed=123).run()
        assert len(r1.match_results) == len(r2.match_results)
        for m1, m2 in zip(r1.match_results, r2.match_results):
            assert m1 == m2

    def test_different_seeds_may_differ(self) -> None:
        results = [_make_simulation(seed=s).run() for s in range(20)]
        points = [
            tuple(mr.points_earned for mr in r.match_results) for r in results
        ]
        # Over 20 seeds, expect some variation
        assert len(set(points)) > 1

    def test_fatigue_increases_for_starters(self) -> None:
        sim = _make_simulation()
        result = sim.run()
        # At least some players should have increased fatigue after 3 matches
        any_increased = any(p.fatigue > 0.0 for p in result.final_squad)
        assert any_increased

    def test_final_squad_and_manager_returned(self) -> None:
        sim = _make_simulation()
        result = sim.run()
        assert len(result.final_squad) == 14  # 11 + 3 subs
        assert result.final_manager.manager_name == "Original Manager"


class TestSimulationTrigger:
    def test_manager_change_trigger(self) -> None:
        sim = _make_simulation()
        trigger = ManagerChangeTrigger(
            outgoing_manager_name="Original Manager",
            incoming_manager_name="New Manager",
            transition_type="mid_season",
            applied_at=10,  # takes effect at week 11
        )
        sim.apply_trigger(trigger)
        result = sim.run()

        # Manager should be replaced
        assert result.final_manager.manager_name == "New Manager"

    def test_trigger_timing(self) -> None:
        """Trigger at applied_at=11 takes effect at week 12."""
        sim = _make_simulation()
        trigger = ManagerChangeTrigger(
            outgoing_manager_name="Original Manager",
            incoming_manager_name="Late Manager",
            transition_type="mid_season",
            applied_at=11,  # takes effect at week 12
        )
        sim.apply_trigger(trigger)
        result = sim.run()
        assert result.final_manager.manager_name == "Late Manager"

    def test_trigger_after_all_fixtures_not_applied(self) -> None:
        """Trigger at applied_at=20 never fires (fixtures end at week 13)."""
        sim = _make_simulation()
        trigger = ManagerChangeTrigger(
            outgoing_manager_name="Original Manager",
            incoming_manager_name="Never Manager",
            transition_type="mid_season",
            applied_at=20,
        )
        sim.apply_trigger(trigger)
        result = sim.run()
        assert result.final_manager.manager_name == "Original Manager"

    def test_transfer_trigger_raises(self) -> None:
        sim = _make_simulation()
        trigger = TransferInTrigger(
            player_name="New Player",
            expected_role="starter",
            applied_at=10,
        )
        with pytest.raises(NotImplementedError):
            sim.apply_trigger(trigger)

    def test_manager_change_resets_understanding(self) -> None:
        sim = _make_simulation()
        trigger = ManagerChangeTrigger(
            outgoing_manager_name="Original Manager",
            incoming_manager_name="New Manager",
            transition_type="mid_season",
            applied_at=10,
        )
        sim.apply_trigger(trigger)
        result = sim.run()
        # After trigger, understanding should have started from 0.25
        # and increased over remaining weeks
        for p in result.final_squad:
            assert p.tactical_understanding >= 0.25

    def test_manager_change_resets_tactical_attributes(self) -> None:
        sim = _make_simulation()
        trigger = ManagerChangeTrigger(
            outgoing_manager_name="Original Manager",
            incoming_manager_name="New Manager",
            transition_type="mid_season",
            applied_at=10,
        )
        sim.apply_trigger(trigger)
        result = sim.run()
        mgr = result.final_manager
        # Tactical attributes should be reset to neutral defaults
        assert mgr.pressing_intensity == 50.0
        assert mgr.possession_preference == 0.5
        assert mgr.preferred_formation == "4-4-2"

    def test_result_is_snapshot(self) -> None:
        """Mutating the result should not affect engine state."""
        sim = _make_simulation()
        result = sim.run()
        # Mutate the returned squad
        result.final_squad[0].current_form = 999.0
        # Engine's internal state should be unaffected
        assert sim._squad[0].current_form != 999.0


class TestBranchIndependence:
    def test_branches_do_not_share_state(self) -> None:
        """Two simulations from deepcopied inputs should diverge."""
        base_squad = _make_squad()
        base_manager = _make_manager()

        sim_a = Simulation(
            team=_make_team(),
            squad=copy.deepcopy(base_squad),
            manager=copy.deepcopy(base_manager),
            fixture_list=_make_fixture_list(),
            opponent_strengths=_make_opponent_strengths(),
            rules=_rules(),
            handler=RuleBasedHandler(_rules()),
            rng=np.random.default_rng(100),
        )

        sim_b = Simulation(
            team=_make_team(),
            squad=copy.deepcopy(base_squad),
            manager=copy.deepcopy(base_manager),
            fixture_list=_make_fixture_list(),
            opponent_strengths=_make_opponent_strengths(),
            rules=_rules(),
            handler=RuleBasedHandler(_rules()),
            rng=np.random.default_rng(100),
        )

        # Apply trigger only to branch B
        sim_b.apply_trigger(
            ManagerChangeTrigger(
                outgoing_manager_name="Original Manager",
                incoming_manager_name="New Manager",
                transition_type="mid_season",
                applied_at=10,
            )
        )

        result_a = sim_a.run()
        result_b = sim_b.run()

        # Branch A keeps original manager
        assert result_a.final_manager.manager_name == "Original Manager"
        # Branch B has new manager
        assert result_b.final_manager.manager_name == "New Manager"

        # Squad states should differ (different tactical understanding paths)
        understanding_a = [p.tactical_understanding for p in result_a.final_squad]
        understanding_b = [p.tactical_understanding for p in result_b.final_squad]
        assert understanding_a != understanding_b
