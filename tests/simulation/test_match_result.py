"""Tests for match result simulation via Poisson model."""

from __future__ import annotations

import numpy as np
import pytest

from iffootball.agents.fixture import Fixture, OpponentStrength
from iffootball.agents.player import BroadPosition, PlayerAgent, RoleFamily
from iffootball.agents.team import TeamBaseline
from iffootball.config import AdaptationConfig, MatchConfig
from iffootball.simulation.match_result import (
    MatchResult,
    calc_agent_state_factor,
    simulate_match,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_ADAPTATION = AdaptationConfig(
    base_fatigue_increase=0.05,
    base_fatigue_recovery=0.03,
    tactical_understanding_gain=0.04,
    fatigue_penalty_weight=0.5,
    trust_increase_on_start=0.02,
    trust_decrease_on_bench=0.01,
)

_DEFAULT_MATCH = MatchConfig(home_advantage_factor=1.1)


def _make_player(
    player_id: int = 1,
    current_form: float = 0.5,
    fatigue: float = 0.0,
) -> PlayerAgent:
    """Create a minimal PlayerAgent with specified form and fatigue."""
    return PlayerAgent(
        player_id=player_id,
        player_name=f"Player {player_id}",
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
        current_form=current_form,
        fatigue=fatigue,
    )


def _make_starters(
    n: int = 11,
    current_form: float = 0.5,
    fatigue: float = 0.0,
) -> list[PlayerAgent]:
    return [_make_player(i, current_form, fatigue) for i in range(1, n + 1)]


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
        matches_remaining=10,
    )


def _make_opponent() -> OpponentStrength:
    return OpponentStrength(
        opponent_name="Team B",
        xg_for_per90=1.2,
        xg_against_per90=1.1,
        elo_rating=1500.0,
    )


def _make_fixture(is_home: bool = True) -> Fixture:
    return Fixture(match_week=5, opponent_name="Team B", is_home=is_home)


# ---------------------------------------------------------------------------
# calc_agent_state_factor
# ---------------------------------------------------------------------------


class TestCalcAgentStateFactor:
    def test_neutral_starters(self) -> None:
        """Neutral form (0.5) and zero fatigue -> factor 1.0."""
        starters = _make_starters(current_form=0.5, fatigue=0.0)
        factor = calc_agent_state_factor(starters, 0.5)
        assert factor == pytest.approx(1.0)

    def test_empty_starters_returns_neutral(self) -> None:
        factor = calc_agent_state_factor([], 0.5)
        assert factor == 1.0

    def test_high_form_increases_factor(self) -> None:
        starters = _make_starters(current_form=0.75, fatigue=0.0)
        factor = calc_agent_state_factor(starters, 0.5)
        assert factor == pytest.approx(0.75 / 0.5)
        assert factor == pytest.approx(1.5)

    def test_high_fatigue_decreases_factor(self) -> None:
        starters = _make_starters(current_form=0.5, fatigue=0.8)
        factor = calc_agent_state_factor(starters, 0.5)
        # 1.0 * (1.0 - 0.8 * 0.5) = 1.0 * 0.6 = 0.6
        assert factor == pytest.approx(0.6)

    def test_clamped_to_minimum(self) -> None:
        """Very high fatigue + low form should clamp to 0.5."""
        starters = _make_starters(current_form=0.1, fatigue=1.0)
        factor = calc_agent_state_factor(starters, 0.5)
        assert factor == 0.5

    def test_clamped_to_maximum(self) -> None:
        """Very high form should clamp to 1.5."""
        starters = _make_starters(current_form=1.0, fatigue=0.0)
        factor = calc_agent_state_factor(starters, 0.5)
        assert factor == pytest.approx(1.5)

    def test_fatigue_penalty_weight_zero(self) -> None:
        """When weight is 0, fatigue has no effect."""
        starters = _make_starters(current_form=0.5, fatigue=1.0)
        factor = calc_agent_state_factor(starters, 0.0)
        assert factor == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# simulate_match
# ---------------------------------------------------------------------------


class TestSimulateMatch:
    def test_returns_match_result(self) -> None:
        rng = np.random.default_rng(42)
        result = simulate_match(
            team=_make_team(),
            opponent=_make_opponent(),
            starters=_make_starters(),
            fixture=_make_fixture(is_home=True),
            adaptation=_DEFAULT_ADAPTATION,
            match_config=_DEFAULT_MATCH,
            rng=rng,
        )
        assert isinstance(result, MatchResult)
        assert result.home_goals >= 0
        assert result.away_goals >= 0
        assert result.is_home is True
        assert result.points_earned in (0, 1, 3)

    def test_reproducible_with_same_seed(self) -> None:
        """Same seed produces identical results."""
        r1 = simulate_match(
            team=_make_team(),
            opponent=_make_opponent(),
            starters=_make_starters(),
            fixture=_make_fixture(),
            adaptation=_DEFAULT_ADAPTATION,
            match_config=_DEFAULT_MATCH,
            rng=np.random.default_rng(123),
        )
        r2 = simulate_match(
            team=_make_team(),
            opponent=_make_opponent(),
            starters=_make_starters(),
            fixture=_make_fixture(),
            adaptation=_DEFAULT_ADAPTATION,
            match_config=_DEFAULT_MATCH,
            rng=np.random.default_rng(123),
        )
        assert r1 == r2

    def test_different_seeds_may_differ(self) -> None:
        """Different seeds should eventually produce different results."""
        results = {
            simulate_match(
                team=_make_team(),
                opponent=_make_opponent(),
                starters=_make_starters(),
                fixture=_make_fixture(),
                adaptation=_DEFAULT_ADAPTATION,
                match_config=_DEFAULT_MATCH,
                rng=np.random.default_rng(seed),
            )
            for seed in range(50)
        }
        # Over 50 seeds, we expect at least 2 different outcomes
        assert len(results) > 1

    def test_points_win(self) -> None:
        """Verify points=3 when simulated team scores more."""
        # Use a team with very high xG to make wins likely
        team = TeamBaseline(
            team_name="Strong",
            competition_id=1,
            season_id=1,
            played_match_ids=frozenset({1}),
            xg_for_per90=5.0,
            xg_against_per90=0.1,
            ppda=8.0,
            progressive_passes_per90=60.0,
            possession_pct=0.65,
            league_position=1,
            points_to_safety=20,
            points_to_title=0,
            matches_remaining=5,
        )
        opp = OpponentStrength(
            opponent_name="Weak",
            xg_for_per90=0.3,
            xg_against_per90=3.0,
            elo_rating=1300.0,
        )
        wins = 0
        for seed in range(100):
            result = simulate_match(
                team=team,
                opponent=opp,
                starters=_make_starters(),
                fixture=_make_fixture(),
                adaptation=_DEFAULT_ADAPTATION,
                match_config=_DEFAULT_MATCH,
                rng=np.random.default_rng(seed),
            )
            if result.points_earned == 3:
                wins += 1
        # Strong team should win most matches
        assert wins > 50

    def test_home_away_goals_mapping(self) -> None:
        """When is_home=True, simulated team's goals are home_goals."""
        rng = np.random.default_rng(42)
        home_result = simulate_match(
            team=_make_team(),
            opponent=_make_opponent(),
            starters=_make_starters(),
            fixture=_make_fixture(is_home=True),
            adaptation=_DEFAULT_ADAPTATION,
            match_config=_DEFAULT_MATCH,
            rng=rng,
        )
        assert home_result.is_home is True

        rng = np.random.default_rng(42)
        away_result = simulate_match(
            team=_make_team(),
            opponent=_make_opponent(),
            starters=_make_starters(),
            fixture=_make_fixture(is_home=False),
            adaptation=_DEFAULT_ADAPTATION,
            match_config=_DEFAULT_MATCH,
            rng=rng,
        )
        assert away_result.is_home is False
        # Same rng seed → same raw Poisson draws, but mapped to
        # opposite home/away slots.
        assert home_result.home_goals == away_result.away_goals
        assert home_result.away_goals == away_result.home_goals

    def test_empty_starters_uses_neutral_factor(self) -> None:
        """Empty starters should not crash; uses factor 1.0."""
        rng = np.random.default_rng(42)
        result = simulate_match(
            team=_make_team(),
            opponent=_make_opponent(),
            starters=[],
            fixture=_make_fixture(),
            adaptation=_DEFAULT_ADAPTATION,
            match_config=_DEFAULT_MATCH,
            rng=rng,
        )
        assert isinstance(result, MatchResult)
