"""Tests for turning point detection and rule-based handler."""

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
from iffootball.simulation.turning_point import (
    VALID_ACTIONS,
    ActionDistribution,
    RuleBasedHandler,
    SimContext,
    TurningPointHandler,
    detect_manager_turning_points,
    detect_player_turning_points,
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
            trust_increase_on_start=0.02,
            trust_decrease_on_bench=0.01,
            home_advantage_factor=1.1,
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
    )


def _make_player(
    bench_streak: int = 0,
    tactical_understanding: float = 50.0,
    manager_trust: float = 50.0,
) -> PlayerAgent:
    return PlayerAgent(
        player_id=1,
        player_name="Player 1",
        position_name="Center Forward",
        role_family=RoleFamily.FORWARD,
        broad_position=BroadPosition.FW,
        pace=50.0,
        passing=50.0,
        shooting=50.0,
        pressing=50.0,
        defending=50.0,
        physicality=50.0,
        consistency=50.0,
        bench_streak=bench_streak,
        tactical_understanding=tactical_understanding,
        manager_trust=manager_trust,
    )


def _manager(
    job_security: float = 1.0,
    style_stubbornness: float = 50.0,
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
        preferred_formation="4-3-3",
        job_security=job_security,
        style_stubbornness=style_stubbornness,
    )


def _context(
    matches_since_appointment: int | None = None,
    manager: ManagerAgent | None = None,
) -> SimContext:
    return SimContext(
        current_week=10,
        matches_since_appointment=matches_since_appointment,
        manager=manager or _manager(),
    )


# ---------------------------------------------------------------------------
# ActionDistribution
# ---------------------------------------------------------------------------


class TestActionDistribution:
    def test_normalises_to_1(self) -> None:
        ad = ActionDistribution({"adapt": 6, "resist": 3, "transfer": 1})
        assert sum(ad.choices.values()) == pytest.approx(1.0)
        assert ad.choices["adapt"] == pytest.approx(0.6)

    def test_already_normalised(self) -> None:
        ad = ActionDistribution({"adapt": 0.5, "resist": 0.3, "transfer": 0.2})
        assert sum(ad.choices.values()) == pytest.approx(1.0)

    def test_unknown_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown action keys"):
            ActionDistribution({"adapt": 0.5, "flee": 0.5})

    def test_negative_value_rejected(self) -> None:
        with pytest.raises(ValueError, match="Negative probability"):
            ActionDistribution({"adapt": 0.5, "resist": -0.1, "transfer": 0.6})

    def test_zero_total_rejected(self) -> None:
        with pytest.raises(ValueError, match="Total probability"):
            ActionDistribution({"adapt": 0.0, "resist": 0.0, "transfer": 0.0})

    def test_subset_of_valid_actions(self) -> None:
        """Only some actions present is valid."""
        ad = ActionDistribution({"adapt": 1.0})
        assert ad.choices["adapt"] == pytest.approx(1.0)

    def test_zero_probability_for_some(self) -> None:
        ad = ActionDistribution({"adapt": 0.8, "resist": 0.2, "transfer": 0.0})
        assert ad.choices["transfer"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# detect_player_turning_points
# ---------------------------------------------------------------------------


class TestDetectPlayerTurningPoints:
    def test_no_tp_by_default(self) -> None:
        player = _make_player()
        tps = detect_player_turning_points(player, _context(), _rules())
        assert tps == []

    def test_bench_streak_tp(self) -> None:
        player = _make_player(bench_streak=3)
        tps = detect_player_turning_points(player, _context(), _rules())
        assert "bench_streak" in tps

    def test_bench_streak_below_threshold(self) -> None:
        player = _make_player(bench_streak=2)
        tps = detect_player_turning_points(player, _context(), _rules())
        assert "bench_streak" not in tps

    def test_low_understanding_in_short_term(self) -> None:
        player = _make_player(tactical_understanding=0.30)
        ctx = _context(matches_since_appointment=2)
        tps = detect_player_turning_points(player, ctx, _rules())
        assert "low_understanding" in tps

    def test_low_understanding_outside_window(self) -> None:
        player = _make_player(tactical_understanding=0.30)
        ctx = _context(matches_since_appointment=10)
        tps = detect_player_turning_points(player, ctx, _rules())
        assert "low_understanding" not in tps

    def test_low_understanding_no_appointment(self) -> None:
        player = _make_player(tactical_understanding=0.30)
        ctx = _context(matches_since_appointment=None)
        tps = detect_player_turning_points(player, ctx, _rules())
        assert "low_understanding" not in tps

    def test_understanding_at_threshold_no_tp(self) -> None:
        # threshold is 0.40; at threshold → no TP
        player = _make_player(tactical_understanding=0.40)
        ctx = _context(matches_since_appointment=2)
        tps = detect_player_turning_points(player, ctx, _rules())
        assert "low_understanding" not in tps

    def test_both_tps_simultaneously(self) -> None:
        player = _make_player(bench_streak=5, tactical_understanding=0.20)
        ctx = _context(matches_since_appointment=1)
        tps = detect_player_turning_points(player, ctx, _rules())
        assert "bench_streak" in tps
        assert "low_understanding" in tps


# ---------------------------------------------------------------------------
# detect_manager_turning_points
# ---------------------------------------------------------------------------


class TestDetectManagerTurningPoints:
    def test_no_tp_secure(self) -> None:
        mgr = _manager(job_security=0.8)
        tps = detect_manager_turning_points(mgr, _rules())
        assert tps == []

    def test_warning_tp(self) -> None:
        mgr = _manager(job_security=0.20)
        tps = detect_manager_turning_points(mgr, _rules())
        assert "job_security_warning" in tps

    def test_critical_tp(self) -> None:
        mgr = _manager(job_security=0.05)
        tps = detect_manager_turning_points(mgr, _rules())
        assert "job_security_critical" in tps
        # Critical takes precedence; warning should not be in the list
        assert "job_security_warning" not in tps

    def test_warning_blocked_by_stubbornness(self) -> None:
        mgr = _manager(job_security=0.20, style_stubbornness=85.0)
        tps = detect_manager_turning_points(mgr, _rules())
        assert "job_security_warning" not in tps

    def test_critical_not_blocked_by_stubbornness(self) -> None:
        mgr = _manager(job_security=0.05, style_stubbornness=85.0)
        tps = detect_manager_turning_points(mgr, _rules())
        assert "job_security_critical" in tps

    def test_at_warning_threshold_no_tp(self) -> None:
        mgr = _manager(job_security=0.30)
        tps = detect_manager_turning_points(mgr, _rules())
        assert tps == []


# ---------------------------------------------------------------------------
# RuleBasedHandler
# ---------------------------------------------------------------------------


class TestRuleBasedHandler:
    def test_satisfies_protocol(self) -> None:
        handler: TurningPointHandler = RuleBasedHandler(_rules())
        assert hasattr(handler, "handle")

    def test_default_adapt_heavy(self) -> None:
        handler = RuleBasedHandler(_rules())
        player = _make_player()
        dist = handler.handle(player, _context())
        assert dist.choices["adapt"] > dist.choices["resist"]

    def test_bench_streak_low_trust_resist_heavy(self) -> None:
        handler = RuleBasedHandler(_rules())
        player = _make_player(bench_streak=5, manager_trust=0.20)
        dist = handler.handle(player, _context())
        assert dist.choices["resist"] > dist.choices["adapt"]

    def test_bench_streak_high_trust_not_resist(self) -> None:
        """High trust bench player should still adapt."""
        handler = RuleBasedHandler(_rules())
        player = _make_player(bench_streak=5, manager_trust=0.80)
        dist = handler.handle(player, _context())
        assert dist.choices["adapt"] > dist.choices["resist"]

    def test_low_understanding_adapt_slightly_higher(self) -> None:
        handler = RuleBasedHandler(_rules())
        player = _make_player(tactical_understanding=0.20)
        ctx = _context(matches_since_appointment=2)
        dist = handler.handle(player, ctx)
        assert dist.choices["adapt"] > dist.choices["resist"]
        # But resist is higher than default
        default_dist = handler.handle(_make_player(), _context())
        assert dist.choices["resist"] > default_dist.choices["resist"]

    def test_returns_valid_distribution(self) -> None:
        handler = RuleBasedHandler(_rules())
        dist = handler.handle(_make_player(), _context())
        assert all(k in VALID_ACTIONS for k in dist.choices)
        assert sum(dist.choices.values()) == pytest.approx(1.0)
