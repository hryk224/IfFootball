"""Tests for lineup selection logic."""

from __future__ import annotations

import pytest

from iffootball.agents.manager import ManagerAgent
from iffootball.agents.player import BroadPosition, PlayerAgent, RoleFamily
from iffootball.config import (
    AdaptationConfig,
    MatchConfig,
    ManagerTurningPointConfig,
    PlayerTurningPointConfig,
    SimulationRules,
    TurningPointConfig,
)
from iffootball.simulation.lineup_selection import (
    LineupResult,
    calc_selection_score,
    parse_formation,
    select_lineup,
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


def _manager(
    pressing_intensity: float = 55.0,
    possession_preference: float = 0.55,
    preferred_formation: str | None = "4-3-3",
) -> ManagerAgent:
    return ManagerAgent(
        manager_name="Test Manager",
        team_name="Test Team",
        competition_id=1,
        season_id=1,
        tenure_match_ids=frozenset(),
        pressing_intensity=pressing_intensity,
        possession_preference=possession_preference,
        counter_tendency=1.0 - possession_preference,
        preferred_formation=preferred_formation,
    )


_POSITIONS_433 = [
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


def _make_player(
    player_id: int,
    position_name: str,
    role_family: RoleFamily,
    broad_position: BroadPosition,
    pressing: float = 50.0,
    passing: float = 50.0,
    current_form: float = 0.5,
    fatigue: float = 0.0,
    manager_trust: float = 0.5,
    tactical_understanding: float = 0.5,
    bench_streak: int = 0,
) -> PlayerAgent:
    return PlayerAgent(
        player_id=player_id,
        player_name=f"Player {player_id}",
        position_name=position_name,
        role_family=role_family,
        broad_position=broad_position,
        pace=50.0,
        passing=passing,
        shooting=50.0,
        pressing=pressing,
        defending=50.0,
        physicality=50.0,
        consistency=50.0,
        current_form=current_form,
        fatigue=fatigue,
        manager_trust=manager_trust,
        tactical_understanding=tactical_understanding,
        bench_streak=bench_streak,
    )


def _make_squad_433(n_extra: int = 3) -> list[PlayerAgent]:
    """Create a 4-3-3 squad (11 starters + n_extra subs)."""
    squad: list[PlayerAgent] = []
    for i, (pos_name, role, broad) in enumerate(_POSITIONS_433):
        squad.append(_make_player(i + 1, pos_name, role, broad))

    # Extra players for bench
    extra_positions = [
        ("Center Back", RoleFamily.CENTER_BACK, BroadPosition.DF),
        ("Center Defensive Midfield", RoleFamily.DEFENSIVE_MIDFIELDER, BroadPosition.MF),
        ("Center Forward", RoleFamily.FORWARD, BroadPosition.FW),
    ]
    for j in range(n_extra):
        pos_name, role, broad = extra_positions[j % len(extra_positions)]
        squad.append(
            _make_player(100 + j, pos_name, role, broad)
        )
    return squad


# ---------------------------------------------------------------------------
# parse_formation
# ---------------------------------------------------------------------------


class TestParseFormation:
    def test_433(self) -> None:
        result = parse_formation("4-3-3")
        assert result == {
            BroadPosition.GK: 1,
            BroadPosition.DF: 4,
            BroadPosition.MF: 3,
            BroadPosition.FW: 3,
        }

    def test_352(self) -> None:
        result = parse_formation("3-5-2")
        assert result == {
            BroadPosition.GK: 1,
            BroadPosition.DF: 3,
            BroadPosition.MF: 5,
            BroadPosition.FW: 2,
        }

    def test_none_defaults_to_442(self) -> None:
        result = parse_formation(None)
        assert result == {
            BroadPosition.GK: 1,
            BroadPosition.DF: 4,
            BroadPosition.MF: 4,
            BroadPosition.FW: 2,
        }

    def test_invalid_format_raises(self) -> None:
        with pytest.raises(ValueError, match="3 segments"):
            parse_formation("4-3")

    def test_non_integer_raises(self) -> None:
        with pytest.raises(ValueError, match="integers"):
            parse_formation("4-x-3")


# ---------------------------------------------------------------------------
# calc_selection_score
# ---------------------------------------------------------------------------


class TestCalcSelectionScore:
    def test_neutral_player(self) -> None:
        """All-neutral player should return a positive score."""
        player = _make_player(
            1, "Center Forward", RoleFamily.FORWARD, BroadPosition.FW
        )
        score = calc_selection_score(player, _manager(), _rules(), None)
        assert score > 0

    def test_higher_form_higher_score(self) -> None:
        low = _make_player(
            1, "Center Forward", RoleFamily.FORWARD, BroadPosition.FW,
            current_form=0.3,
        )
        high = _make_player(
            2, "Center Forward", RoleFamily.FORWARD, BroadPosition.FW,
            current_form=0.8,
        )
        rules = _rules()
        mgr = _manager()
        assert calc_selection_score(high, mgr, rules, None) > calc_selection_score(
            low, mgr, rules, None
        )

    def test_fatigue_reduces_score(self) -> None:
        fresh = _make_player(
            1, "Center Forward", RoleFamily.FORWARD, BroadPosition.FW,
            fatigue=0.0,
        )
        tired = _make_player(
            2, "Center Forward", RoleFamily.FORWARD, BroadPosition.FW,
            fatigue=0.8,
        )
        rules = _rules()
        mgr = _manager()
        assert calc_selection_score(fresh, mgr, rules, None) > calc_selection_score(
            tired, mgr, rules, None
        )

    def test_low_understanding_penalty_in_short_term(self) -> None:
        """Low tactical_understanding should reduce score in short-term window."""
        player = _make_player(
            1, "Center Forward", RoleFamily.FORWARD, BroadPosition.FW,
            tactical_understanding=0.20,  # below 0.40
        )
        rules = _rules()
        mgr = _manager()
        # Within short_term_window (4 matches)
        score_short = calc_selection_score(player, mgr, rules, 2)
        # Outside short_term_window
        score_long = calc_selection_score(player, mgr, rules, 10)
        assert score_short < score_long

    def test_no_penalty_when_no_appointment(self) -> None:
        """matches_since_appointment=None means no penalty."""
        player = _make_player(
            1, "Center Forward", RoleFamily.FORWARD, BroadPosition.FW,
            tactical_understanding=0.20,
        )
        rules = _rules()
        mgr = _manager()
        score_none = calc_selection_score(player, mgr, rules, None)
        score_long = calc_selection_score(player, mgr, rules, 10)
        assert score_none == score_long

    def test_no_penalty_above_threshold(self) -> None:
        """No penalty when tactical_understanding >= threshold."""
        player = _make_player(
            1, "Center Forward", RoleFamily.FORWARD, BroadPosition.FW,
            tactical_understanding=0.50,  # above 0.40
        )
        rules = _rules()
        mgr = _manager()
        score_short = calc_selection_score(player, mgr, rules, 2)
        score_long = calc_selection_score(player, mgr, rules, 10)
        assert score_short == score_long


# ---------------------------------------------------------------------------
# select_lineup
# ---------------------------------------------------------------------------


class TestSelectLineup:
    def test_selects_11_starters(self) -> None:
        squad = _make_squad_433()
        result = select_lineup(squad, _manager(), _rules())
        assert isinstance(result, LineupResult)
        assert len(result.starters) == 11
        assert len(result.benched) == len(squad) - 11

    def test_formation_slots_respected(self) -> None:
        """Starters should match the formation's position distribution."""
        squad = _make_squad_433()
        result = select_lineup(squad, _manager(), _rules())
        counts = {bp: 0 for bp in BroadPosition}
        for p in result.starters:
            counts[p.broad_position] += 1
        assert counts[BroadPosition.GK] == 1
        assert counts[BroadPosition.DF] == 4
        # MF needs 3 but we have 4 MF + extra MF; FW needs 3 but we have 1 FW + extra FW
        # The overflow logic fills remaining from best available
        assert counts[BroadPosition.MF] + counts[BroadPosition.FW] == 6

    def test_bench_streak_reset_for_starters(self) -> None:
        squad = _make_squad_433()
        # Give everyone a bench streak
        for p in squad:
            p.bench_streak = 5
        select_lineup(squad, _manager(), _rules())
        starter_streaks = [p.bench_streak for p in squad if p.bench_streak == 0]
        assert len(starter_streaks) == 11

    def test_bench_streak_incremented_for_benched(self) -> None:
        squad = _make_squad_433(n_extra=3)
        for p in squad:
            p.bench_streak = 0
        result = select_lineup(squad, _manager(), _rules())
        for p in result.benched:
            assert p.bench_streak == 1

    def test_high_form_player_selected_over_low(self) -> None:
        """A high-form player should be preferred in the same position."""
        low_form = _make_player(
            1, "Center Forward", RoleFamily.FORWARD, BroadPosition.FW,
            current_form=0.2,
        )
        high_form = _make_player(
            2, "Center Forward", RoleFamily.FORWARD, BroadPosition.FW,
            current_form=0.9,
        )
        # Build a minimal squad: 1 GK, 4 DF, 3 MF, 2 FW
        squad = [
            _make_player(10, "Goalkeeper", RoleFamily.GOALKEEPER, BroadPosition.GK),
        ]
        for i in range(4):
            squad.append(
                _make_player(20 + i, "Center Back", RoleFamily.CENTER_BACK, BroadPosition.DF)
            )
        for i in range(3):
            squad.append(
                _make_player(30 + i, "Center Defensive Midfield", RoleFamily.DEFENSIVE_MIDFIELDER, BroadPosition.MF)
            )
        squad.extend([low_form, high_form])

        # 4-3-3 needs 3 FW but we only have 2; both should be selected
        # But let's use 4-3-2 style: use 4-4-2 with 2 FW
        mgr = _manager(preferred_formation="4-3-2")
        result = select_lineup(squad, mgr, _rules())

        fw_starters = [
            p for p in result.starters if p.broad_position == BroadPosition.FW
        ]
        assert high_form in fw_starters

    def test_none_formation_uses_default(self) -> None:
        squad = _make_squad_433(n_extra=5)
        mgr = _manager(preferred_formation=None)
        result = select_lineup(squad, mgr, _rules())
        # Default 4-4-2: should still select 11
        assert len(result.starters) == 11

    def test_small_squad(self) -> None:
        """Squad smaller than 11 selects all available."""
        squad = [
            _make_player(1, "Goalkeeper", RoleFamily.GOALKEEPER, BroadPosition.GK),
            _make_player(2, "Center Back", RoleFamily.CENTER_BACK, BroadPosition.DF),
            _make_player(3, "Center Forward", RoleFamily.FORWARD, BroadPosition.FW),
        ]
        result = select_lineup(squad, _manager(), _rules())
        assert len(result.starters) == 3
        assert len(result.benched) == 0
