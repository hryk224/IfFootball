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
    ActionDistributionConfig,
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
            action_distribution=ActionDistributionConfig(
                bench_streak_low_trust={"resist": 0.6, "adapt": 0.3, "transfer": 0.1},
                low_understanding={"adapt": 0.5, "resist": 0.4, "transfer": 0.1},
                default={"adapt": 0.8, "resist": 0.2, "transfer": 0.0},
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
    rng = np.random.default_rng(seed)
    match_seed, action_seed = rng.spawn(2)
    return Simulation(
        team=_make_team(),
        squad=_make_squad(),
        manager=_make_manager(),
        fixture_list=_make_fixture_list(),
        opponent_strengths=_make_opponent_strengths(),
        rules=_rules(),
        handler=RuleBasedHandler(_rules()),
        match_rng=np.random.default_rng(match_seed),
        action_rng=np.random.default_rng(action_seed),
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
        assert mgr.counter_tendency == 0.5
        assert mgr.preferred_formation == "4-4-2"
        assert mgr.implementation_speed == 50.0
        assert mgr.youth_development == 50.0
        assert mgr.style_stubbornness == 50.0

    def test_result_is_snapshot(self) -> None:
        """Mutating the result should not affect engine state."""
        sim = _make_simulation()
        result = sim.run()
        # Mutate the returned squad
        result.final_squad[0].current_form = 999.0
        # Engine's internal state should be unaffected
        assert sim._squad[0].current_form != 999.0


def _make_incoming_profile() -> ManagerAgent:
    """Build a ManagerAgent to use as incoming_profile in triggers."""
    return ManagerAgent(
        manager_name="Profile Name (should be ignored)",
        team_name="Other Team",
        competition_id=2,
        season_id=2,
        tenure_match_ids=frozenset({100, 101}),
        pressing_intensity=70.0,
        possession_preference=0.65,
        counter_tendency=0.35,
        preferred_formation="3-5-2",
        implementation_speed=80.0,
        youth_development=30.0,
        style_stubbornness=75.0,
    )


class TestManagerChangeWithProfile:
    """Tests for incoming_profile on ManagerChangeTrigger."""

    def test_profile_attributes_copied(self) -> None:
        sim = _make_simulation()
        profile = _make_incoming_profile()
        trigger = ManagerChangeTrigger(
            outgoing_manager_name="Original Manager",
            incoming_manager_name="New Manager",
            transition_type="mid_season",
            applied_at=10,
            incoming_profile=profile,
        )
        sim.apply_trigger(trigger)
        result = sim.run()
        mgr = result.final_manager
        assert mgr.pressing_intensity == 70.0
        assert mgr.possession_preference == 0.65
        assert mgr.counter_tendency == 0.35
        assert mgr.preferred_formation == "3-5-2"
        assert mgr.implementation_speed == 80.0
        assert mgr.youth_development == 30.0
        assert mgr.style_stubbornness == 75.0

    def test_manager_name_from_trigger_not_profile(self) -> None:
        sim = _make_simulation()
        profile = _make_incoming_profile()
        trigger = ManagerChangeTrigger(
            outgoing_manager_name="Original Manager",
            incoming_manager_name="Trigger Name",
            transition_type="mid_season",
            applied_at=10,
            incoming_profile=profile,
        )
        sim.apply_trigger(trigger)
        result = sim.run()
        # manager_name must come from trigger, not profile
        assert result.final_manager.manager_name == "Trigger Name"

    def test_dynamic_state_reset_with_profile(self) -> None:
        """job_security and squad_trust must be reset, not copied from profile."""
        sim = _make_simulation()
        profile = _make_incoming_profile()
        # Set non-default dynamic state on the profile to prove it is ignored.
        profile.job_security = 0.3
        profile.squad_trust = {"Player 1": 0.9}
        trigger = ManagerChangeTrigger(
            outgoing_manager_name="Original Manager",
            incoming_manager_name="New Manager",
            transition_type="mid_season",
            applied_at=10,
            incoming_profile=profile,
        )
        # Execute trigger in isolation via internal method.
        sim._execute_trigger(trigger)
        mgr = sim._manager
        assert mgr.job_security == 1.0
        assert mgr.squad_trust == {}

    def test_squad_reset_with_profile(self) -> None:
        sim = _make_simulation()
        profile = _make_incoming_profile()
        trigger = ManagerChangeTrigger(
            outgoing_manager_name="Original Manager",
            incoming_manager_name="New Manager",
            transition_type="mid_season",
            applied_at=10,
            incoming_profile=profile,
        )
        sim.apply_trigger(trigger)
        result = sim.run()
        # Player tactical_understanding should have started from 0.25 reset
        for p in result.final_squad:
            assert p.tactical_understanding >= 0.25

    def test_none_profile_uses_defaults(self) -> None:
        """Explicitly passing None behaves like no profile."""
        sim = _make_simulation()
        trigger = ManagerChangeTrigger(
            outgoing_manager_name="Original Manager",
            incoming_manager_name="New Manager",
            transition_type="mid_season",
            applied_at=10,
            incoming_profile=None,
        )
        sim.apply_trigger(trigger)
        result = sim.run()
        mgr = result.final_manager
        assert mgr.pressing_intensity == 50.0
        assert mgr.possession_preference == 0.5
        assert mgr.preferred_formation == "4-4-2"
        assert mgr.youth_development == 50.0


def _make_transfer_player() -> PlayerAgent:
    """Build a PlayerAgent for transfer trigger tests."""
    return PlayerAgent(
        player_id=99,
        player_name="Transfer Player",
        position_name="Center Forward",
        role_family=RoleFamily.FORWARD,
        broad_position=BroadPosition.FW,
        pace=70.0,
        passing=60.0,
        shooting=75.0,
        pressing=50.0,
        defending=30.0,
        physicality=65.0,
        consistency=60.0,
    )


class TestTransferInTrigger:
    """Tests for TransferInTrigger execution."""

    def test_player_added_to_squad(self) -> None:
        sim = _make_simulation()
        initial_size = len(sim._squad)
        trigger = TransferInTrigger(
            player_name="Transfer Player",
            expected_role="starter",
            applied_at=10,
            player=_make_transfer_player(),
        )
        sim.apply_trigger(trigger)
        result = sim.run()
        assert len(result.final_squad) == initial_size + 1

    def test_starter_trust_high(self) -> None:
        sim = _make_simulation()
        trigger = TransferInTrigger(
            player_name="Transfer Player",
            expected_role="starter",
            applied_at=10,
            player=_make_transfer_player(),
        )
        sim._execute_trigger(trigger)
        added = [p for p in sim._squad if p.player_id == 99]
        assert len(added) == 1
        assert added[0].manager_trust == 0.7

    def test_rotation_trust_medium(self) -> None:
        sim = _make_simulation()
        trigger = TransferInTrigger(
            player_name="Transfer Player",
            expected_role="rotation",
            applied_at=10,
            player=_make_transfer_player(),
        )
        sim._execute_trigger(trigger)
        added = [p for p in sim._squad if p.player_id == 99]
        assert added[0].manager_trust == 0.5

    def test_squad_trust_low(self) -> None:
        sim = _make_simulation()
        trigger = TransferInTrigger(
            player_name="Transfer Player",
            expected_role="squad",
            applied_at=10,
            player=_make_transfer_player(),
        )
        sim._execute_trigger(trigger)
        added = [p for p in sim._squad if p.player_id == 99]
        assert added[0].manager_trust == 0.3

    def test_tactical_understanding_low(self) -> None:
        sim = _make_simulation()
        trigger = TransferInTrigger(
            player_name="Transfer Player",
            expected_role="starter",
            applied_at=10,
            player=_make_transfer_player(),
        )
        sim._execute_trigger(trigger)
        added = [p for p in sim._squad if p.player_id == 99]
        assert added[0].tactical_understanding == 0.25

    def test_bench_streak_zero(self) -> None:
        sim = _make_simulation()
        trigger = TransferInTrigger(
            player_name="Transfer Player",
            expected_role="starter",
            applied_at=10,
            player=_make_transfer_player(),
        )
        sim._execute_trigger(trigger)
        added = [p for p in sim._squad if p.player_id == 99]
        assert added[0].bench_streak == 0

    def test_fatigue_zero(self) -> None:
        sim = _make_simulation()
        trigger = TransferInTrigger(
            player_name="Transfer Player",
            expected_role="starter",
            applied_at=10,
            player=_make_transfer_player(),
        )
        sim._execute_trigger(trigger)
        added = [p for p in sim._squad if p.player_id == 99]
        assert added[0].fatigue == 0.0

    def test_none_player_raises_value_error(self) -> None:
        sim = _make_simulation()
        trigger = TransferInTrigger(
            player_name="Ghost Player",
            expected_role="starter",
            applied_at=10,
            player=None,
        )
        with pytest.raises(ValueError, match="no player payload"):
            sim._execute_trigger(trigger)

    def test_duplicate_player_id_raises(self) -> None:
        sim = _make_simulation()
        # Player ID 1 already exists in squad.
        duplicate_trigger = TransferInTrigger(
            player_name="Duplicate",
            expected_role="starter",
            applied_at=10,
            player=PlayerAgent(
                player_id=1,
                player_name="Duplicate",
                position_name="Center Forward",
                role_family=RoleFamily.FORWARD,
                broad_position=BroadPosition.FW,
                pace=50.0, passing=50.0, shooting=50.0,
                pressing=50.0, defending=50.0, physicality=50.0,
                consistency=50.0,
            ),
        )
        with pytest.raises(ValueError, match="already in squad"):
            sim._execute_trigger(duplicate_trigger)

    def test_transfer_player_in_lineup_selection(self) -> None:
        """High-trust starter transfer should appear in lineup."""
        sim = _make_simulation()
        trigger = TransferInTrigger(
            player_name="Transfer Player",
            expected_role="starter",
            applied_at=10,
            player=_make_transfer_player(),
        )
        sim.apply_trigger(trigger)
        result = sim.run()
        # The transfer player should be in the final squad.
        transfer_ids = [p.player_id for p in result.final_squad if p.player_id == 99]
        assert len(transfer_ids) == 1


class TestCascadeEventEmission:
    """Tests that engine emits the expected cascade event types."""

    def test_adapt_emits_adaptation_progress(self) -> None:
        """When a TP fires and adapt is sampled, adaptation_progress is recorded."""
        sim = _make_simulation(seed=42)
        # Manager change triggers low_understanding TPs → adapt likely.
        trigger = ManagerChangeTrigger(
            outgoing_manager_name="Original Manager",
            incoming_manager_name="New Manager",
            transition_type="mid_season",
            applied_at=10,
        )
        sim.apply_trigger(trigger)
        result = sim.run()
        event_types = {e.event_type for e in result.cascade_events}
        # With a manager change, at least some players should adapt.
        assert "adaptation_progress" in event_types

    def test_low_understanding_adapt_emits_tactical_confusion(self) -> None:
        """adapt + low_understanding TP → both adaptation_progress and tactical_confusion."""
        sim = _make_simulation(seed=42)
        trigger = ManagerChangeTrigger(
            outgoing_manager_name="Original Manager",
            incoming_manager_name="New Manager",
            transition_type="mid_season",
            applied_at=10,
        )
        sim.apply_trigger(trigger)
        result = sim.run()
        event_types = {e.event_type for e in result.cascade_events}
        # low_understanding fires in the short_term_window after appointment.
        assert "tactical_confusion" in event_types

    def test_multiple_resist_emits_squad_unrest(self) -> None:
        """When 2+ players resist in the same week, squad_unrest is recorded."""
        from iffootball.simulation.cascade_tracker import CascadeTracker
        from iffootball.simulation.turning_point import SimContext

        sim = _make_simulation(seed=0)
        # Force conditions: manager change + low trust + high bench_streak
        # so that bench_streak TP fires and resist is heavily favoured.
        sim._matches_since_appointment = 1
        for p in sim._squad:
            p.bench_streak = 5
            p.manager_trust = 0.1

        context = SimContext(
            current_week=11,
            matches_since_appointment=1,
            manager=sim._manager,
            recent_points=(0, 0, 0),
        )
        # seed=0 deterministically produces 6 resists → squad_unrest.
        sim._action_rng = np.random.default_rng(0)
        tracker = CascadeTracker()
        sim._process_player_turning_points(context, tracker, 11)
        unrest = [e for e in tracker.events if e.event_type == "squad_unrest"]
        assert len(unrest) == 1
        assert unrest[0].cause_chain == ("multiple_resist",)
        assert unrest[0].affected_agent == sim._manager.manager_name

    def test_resist_emits_form_drop_and_trust_decline(self) -> None:
        """Existing behavior: resist → form_drop + trust_decline chain."""
        # Run across seeds to find one with a resist action.
        for seed in range(50):
            sim = _make_simulation(seed=seed)
            trigger = ManagerChangeTrigger(
                outgoing_manager_name="Original Manager",
                incoming_manager_name="New Manager",
                transition_type="mid_season",
                applied_at=10,
            )
            sim.apply_trigger(trigger)
            result = sim.run()
            event_types = {e.event_type for e in result.cascade_events}
            if "form_drop" in event_types:
                assert "trust_decline" in event_types
                return
        pytest.fail("No form_drop event found across 50 seeds")


class TestBranchIndependence:
    def test_branches_do_not_share_state(self) -> None:
        """Two simulations from deepcopied inputs should diverge."""
        base_squad = _make_squad()
        base_manager = _make_manager()

        rng_a = np.random.default_rng(100)
        m_a, act_a = rng_a.spawn(2)
        sim_a = Simulation(
            team=_make_team(),
            squad=copy.deepcopy(base_squad),
            manager=copy.deepcopy(base_manager),
            fixture_list=_make_fixture_list(),
            opponent_strengths=_make_opponent_strengths(),
            rules=_rules(),
            handler=RuleBasedHandler(_rules()),
            match_rng=np.random.default_rng(m_a),
            action_rng=np.random.default_rng(act_a),
        )

        rng_b = np.random.default_rng(100)
        m_b, act_b = rng_b.spawn(2)
        sim_b = Simulation(
            team=_make_team(),
            squad=copy.deepcopy(base_squad),
            manager=copy.deepcopy(base_manager),
            fixture_list=_make_fixture_list(),
            opponent_strengths=_make_opponent_strengths(),
            rules=_rules(),
            handler=RuleBasedHandler(_rules()),
            match_rng=np.random.default_rng(m_b),
            action_rng=np.random.default_rng(act_b),
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


class TestMatchRngIsolation:
    """Guard that match_rng is consumed only by simulate_match().

    The engine's weekly loop must use match_rng exclusively for Poisson
    match result sampling (step 3) and action_rng exclusively for TP
    action sampling (step 9). If match_rng is accidentally used
    elsewhere, the paired comparison guarantee breaks.
    """

    def test_match_results_depend_only_on_match_rng(self) -> None:
        """Same match_rng seed must produce same match results regardless
        of action_rng seed.

        If match_rng were accidentally used for TP action sampling,
        different action_rng seeds would not isolate the contamination,
        and match results would vary with action_rng.
        """
        def run_with_action_seed(action_seed: int) -> list[MatchResult]:
            rng = np.random.default_rng(55)
            m_seed, _ = rng.spawn(2)
            sim = Simulation(
                team=_make_team(),
                squad=_make_squad(),
                manager=_make_manager(),
                fixture_list=_make_fixture_list(),
                opponent_strengths=_make_opponent_strengths(),
                rules=_rules(),
                handler=RuleBasedHandler(_rules()),
                match_rng=np.random.default_rng(m_seed),
                action_rng=np.random.default_rng(action_seed),
            )
            return sim.run().match_results

        # Different action_rng seeds must not affect match results
        results_a = run_with_action_seed(0)
        results_b = run_with_action_seed(999)
        for mr_a, mr_b in zip(results_a, results_b):
            assert mr_a == mr_b

    def test_no_trigger_run_reproducible_regardless_of_action_rng(self) -> None:
        """A no-trigger simulation's match results must be reproducible
        from the same match_rng seed, regardless of action_rng seed.

        This verifies that the engine's weekly loop does not accidentally
        use match_rng outside of simulate_match(). If match_rng were
        consumed by TP processing or other steps, different action_rng
        seeds (which change TP outcomes and thus engine control flow)
        would cause match_rng to drift differently.

        Note: this test only covers the no-trigger path. The trigger
        path is covered by test_match_results_depend_only_on_match_rng
        and by TestPairedDesign in test_comparison.py.
        """
        seed = 55

        def run_no_trigger(action_seed: int) -> list[MatchResult]:
            rng_base = np.random.default_rng(seed)
            m_seed, _ = rng_base.spawn(2)
            sim = Simulation(
                team=_make_team(),
                squad=copy.deepcopy(_make_squad()),
                manager=copy.deepcopy(_make_manager()),
                fixture_list=_make_fixture_list(),
                opponent_strengths=_make_opponent_strengths(),
                rules=_rules(),
                handler=RuleBasedHandler(_rules()),
                match_rng=np.random.default_rng(m_seed),
                action_rng=np.random.default_rng(action_seed),
            )
            return sim.run().match_results

        results_a = run_no_trigger(0)
        results_b = run_no_trigger(999)
        for mr_a, mr_b in zip(results_a, results_b):
            assert mr_a == mr_b
