"""Tests for weekly state update functions."""

from __future__ import annotations

import pytest

from iffootball.agents.manager import ManagerAgent
from iffootball.agents.player import BroadPosition, PlayerAgent, RoleFamily
from iffootball.config import (
    AdaptationConfig,
    ManagerTurningPointConfig,
    PlayerTurningPointConfig,
    SimulationRules,
    TurningPointConfig,
)
from iffootball.simulation.state_update import (
    calc_adaptation_rate,
    calc_tactical_familiarity,
    update_fatigue,
    update_job_security,
    update_manager_trust,
    update_tactical_understanding,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rules() -> SimulationRules:
    return SimulationRules(
        adaptation=AdaptationConfig(
            base_fatigue_increase=0.05,
            base_fatigue_recovery=0.03,
            tactical_understanding_gain=0.04,
            fatigue_penalty_weight=0.5,
            trust_increase_on_start=2.0,
            trust_decrease_on_bench=1.0,
            home_advantage_factor=1.1,
        ),
        turning_points=TurningPointConfig(
            player=PlayerTurningPointConfig(
                bench_streak_threshold=3,
                tactical_understanding_low=0.40,
                short_term_window=4,
                trust_low=40.0,
            ),
            manager=ManagerTurningPointConfig(
                job_security_warning=0.30,
                job_security_critical=0.10,
                style_stubbornness_threshold=80,
            ),
        ),
    )


def _make_player(
    player_id: int = 1,
    broad_position: BroadPosition = BroadPosition.FW,
    fatigue: float = 0.0,
    tactical_understanding: float = 50.0,
    tactical_adaptability: float = 50.0,
    manager_trust: float = 50.0,
) -> PlayerAgent:
    role = {
        BroadPosition.GK: RoleFamily.GOALKEEPER,
        BroadPosition.DF: RoleFamily.CENTER_BACK,
        BroadPosition.MF: RoleFamily.CENTRAL_MIDFIELDER,
        BroadPosition.FW: RoleFamily.FORWARD,
    }[broad_position]
    pos = {
        BroadPosition.GK: "Goalkeeper",
        BroadPosition.DF: "Center Back",
        BroadPosition.MF: "Right Center Midfield",
        BroadPosition.FW: "Center Forward",
    }[broad_position]
    return PlayerAgent(
        player_id=player_id,
        player_name=f"Player {player_id}",
        position_name=pos,
        role_family=role,
        broad_position=broad_position,
        pace=50.0,
        passing=50.0,
        shooting=50.0,
        pressing=50.0,
        defending=50.0,
        physicality=50.0,
        consistency=50.0,
        fatigue=fatigue,
        tactical_understanding=tactical_understanding,
        tactical_adaptability=tactical_adaptability,
        manager_trust=manager_trust,
    )


def _manager(
    preferred_formation: str | None = "4-3-3",
    implementation_speed: float = 50.0,
) -> ManagerAgent:
    return ManagerAgent(
        manager_name="Test Manager",
        team_name="Test Team",
        competition_id=1,
        season_id=1,
        tenure_match_ids=frozenset(),
        pressing_intensity=55.0,
        possession_preference=0.55,
        counter_tendency=0.45,
        preferred_formation=preferred_formation,
        implementation_speed=implementation_speed,
    )


# ---------------------------------------------------------------------------
# Step 4: update_fatigue
# ---------------------------------------------------------------------------


class TestUpdateFatigue:
    def test_starters_gain_fatigue(self) -> None:
        p = _make_player(fatigue=0.0)
        update_fatigue([p], {p.player_id}, _rules())
        assert p.fatigue == pytest.approx(0.05)

    def test_non_starters_recover(self) -> None:
        p = _make_player(fatigue=0.10)
        update_fatigue([p], set(), _rules())
        assert p.fatigue == pytest.approx(0.07)

    def test_fatigue_capped_at_1(self) -> None:
        p = _make_player(fatigue=0.98)
        update_fatigue([p], {p.player_id}, _rules())
        assert p.fatigue == 1.0

    def test_fatigue_floor_at_0(self) -> None:
        p = _make_player(fatigue=0.01)
        update_fatigue([p], set(), _rules())
        assert p.fatigue == 0.0


# ---------------------------------------------------------------------------
# Step 5: tactical understanding
# ---------------------------------------------------------------------------


class TestCalcTacticalFamiliarity:
    def test_gk_always_1(self) -> None:
        p = _make_player(broad_position=BroadPosition.GK)
        assert calc_tactical_familiarity(p, "4-3-3") == 1.0

    def test_fw_in_433_familiar(self) -> None:
        p = _make_player(broad_position=BroadPosition.FW)
        assert calc_tactical_familiarity(p, "4-3-3") == 1.0

    def test_fw_in_451_no_fw_slot(self) -> None:
        """4-5-1 has FW=1 slot, so FW is still familiar."""
        p = _make_player(broad_position=BroadPosition.FW)
        assert calc_tactical_familiarity(p, "4-5-1") == 1.0

    def test_fw_in_460_unfamiliar(self) -> None:
        """4-6-0 has FW=0, so FW player is unfamiliar."""
        p = _make_player(broad_position=BroadPosition.FW)
        assert calc_tactical_familiarity(p, "4-6-0") == 0.5

    def test_none_formation_uses_default(self) -> None:
        p = _make_player(broad_position=BroadPosition.DF)
        assert calc_tactical_familiarity(p, None) == 1.0


class TestCalcAdaptationRate:
    def test_neutral_values(self) -> None:
        p = _make_player(tactical_adaptability=50.0)
        mgr = _manager(implementation_speed=50.0)
        rate = calc_adaptation_rate(p, mgr)
        # 0.5 * 0.5 * 1.0 = 0.25
        assert rate == pytest.approx(0.25)

    def test_max_values(self) -> None:
        p = _make_player(tactical_adaptability=100.0)
        mgr = _manager(implementation_speed=100.0)
        rate = calc_adaptation_rate(p, mgr)
        # 1.0 * 1.0 * 1.0 = 1.0
        assert rate == pytest.approx(1.0)

    def test_unfamiliar_formation_reduces_rate(self) -> None:
        p = _make_player(
            broad_position=BroadPosition.FW, tactical_adaptability=50.0,
        )
        mgr = _manager(
            preferred_formation="4-6-0", implementation_speed=50.0,
        )
        rate = calc_adaptation_rate(p, mgr)
        # 0.5 * 0.5 * 0.5 = 0.125
        assert rate == pytest.approx(0.125)


class TestUpdateTacticalUnderstanding:
    def test_understanding_increases(self) -> None:
        p = _make_player(tactical_understanding=50.0, tactical_adaptability=50.0)
        mgr = _manager(implementation_speed=50.0)
        update_tactical_understanding([p], mgr, _rules())
        # rate=0.25, gain=0.04, delta=0.25*0.04*100=1.0
        assert p.tactical_understanding == pytest.approx(51.0)

    def test_capped_at_100(self) -> None:
        p = _make_player(tactical_understanding=99.5, tactical_adaptability=100.0)
        mgr = _manager(implementation_speed=100.0)
        update_tactical_understanding([p], mgr, _rules())
        assert p.tactical_understanding == 100.0


# ---------------------------------------------------------------------------
# Step 6: manager_trust
# ---------------------------------------------------------------------------


class TestUpdateManagerTrust:
    def test_starters_gain_trust(self) -> None:
        p = _make_player(manager_trust=50.0)
        update_manager_trust([p], {p.player_id}, _rules())
        assert p.manager_trust == pytest.approx(52.0)

    def test_non_starters_lose_trust(self) -> None:
        p = _make_player(manager_trust=50.0)
        update_manager_trust([p], set(), _rules())
        assert p.manager_trust == pytest.approx(49.0)

    def test_trust_capped_at_100(self) -> None:
        p = _make_player(manager_trust=99.5)
        update_manager_trust([p], {p.player_id}, _rules())
        assert p.manager_trust == 100.0

    def test_trust_floor_at_0(self) -> None:
        p = _make_player(manager_trust=0.5)
        update_manager_trust([p], set(), _rules())
        assert p.manager_trust == 0.0


# ---------------------------------------------------------------------------
# Step 7: job_security
# ---------------------------------------------------------------------------


class TestUpdateJobSecurity:
    def test_all_wins(self) -> None:
        mgr = _manager()
        update_job_security(mgr, [3, 3, 3, 3, 3])
        assert mgr.job_security == pytest.approx(1.0)

    def test_all_losses(self) -> None:
        mgr = _manager()
        update_job_security(mgr, [0, 0, 0, 0, 0])
        assert mgr.job_security == pytest.approx(0.0)

    def test_mixed_results(self) -> None:
        mgr = _manager()
        update_job_security(mgr, [3, 1, 0, 3, 1])
        # 8/15 ≈ 0.533
        assert mgr.job_security == pytest.approx(8.0 / 15.0)

    def test_fewer_than_5_matches(self) -> None:
        mgr = _manager()
        update_job_security(mgr, [3, 0])
        assert mgr.job_security == pytest.approx(3.0 / 15.0)

    def test_empty_no_change(self) -> None:
        mgr = _manager()
        original = mgr.job_security
        update_job_security(mgr, [])
        assert mgr.job_security == original
