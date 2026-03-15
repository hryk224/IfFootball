"""Tests for tactical estimate functions."""

from __future__ import annotations

import pytest

from iffootball.agents.league import LeagueContext
from iffootball.agents.manager import ManagerAgent
from iffootball.agents.team import TeamBaseline
from iffootball.visualization.tactical_estimate import (
    estimate_possession,
    estimate_ppda,
    estimate_progressive_passes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _baseline() -> TeamBaseline:
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


def _league() -> LeagueContext:
    return LeagueContext(
        competition_id=1,
        season_id=1,
        name="Test League",
        avg_ppda=11.0,
        avg_progressive_passes_per90=45.0,
        avg_xg_per90=1.3,
    )


def _manager(
    *,
    pressing: float = 50.0,
    possession: float = 0.5,
    formation: str | None = "4-3-3",
) -> ManagerAgent:
    return ManagerAgent(
        manager_name="Test Manager",
        team_name="Team A",
        competition_id=1,
        season_id=1,
        tenure_match_ids=frozenset(),
        pressing_intensity=pressing,
        possession_preference=possession,
        counter_tendency=1.0 - possession,
        preferred_formation=formation,
    )


# ---------------------------------------------------------------------------
# PPDA tests
# ---------------------------------------------------------------------------


class TestEstimatePPDA:
    def test_neutral_manager_no_change(self) -> None:
        result = estimate_ppda(
            _baseline(), _manager(), _league(), is_new_manager=False
        )
        # Neutral pressing (50.0) with no regression → baseline ppda.
        assert result == pytest.approx(10.0)

    def test_high_pressing_lowers_ppda(self) -> None:
        result = estimate_ppda(
            _baseline(),
            _manager(pressing=75.0),
            _league(),
            is_new_manager=False,
        )
        assert result < 10.0

    def test_low_pressing_raises_ppda(self) -> None:
        result = estimate_ppda(
            _baseline(),
            _manager(pressing=25.0),
            _league(),
            is_new_manager=False,
        )
        assert result > 10.0

    def test_new_manager_regresses_toward_league(self) -> None:
        neutral_new = estimate_ppda(
            _baseline(), _manager(), _league(), is_new_manager=True
        )
        neutral_old = estimate_ppda(
            _baseline(), _manager(), _league(), is_new_manager=False
        )
        # New manager blends toward league avg (11.0 > 10.0), so PPDA rises.
        assert neutral_new > neutral_old

    def test_zero_league_avg_falls_back(self) -> None:
        league = LeagueContext(1, 1, "Test", avg_ppda=0.0)
        result = estimate_ppda(
            _baseline(), _manager(), league, is_new_manager=True
        )
        assert result == _baseline().ppda

    def test_result_clamped_above_one(self) -> None:
        result = estimate_ppda(
            _baseline(),
            _manager(pressing=500.0),
            _league(),
            is_new_manager=False,
        )
        assert result >= 1.0


# ---------------------------------------------------------------------------
# Possession tests
# ---------------------------------------------------------------------------


class TestEstimatePossession:
    def test_neutral_manager_no_change(self) -> None:
        result = estimate_possession(
            _baseline(), _manager(), 0.5, is_new_manager=False
        )
        assert result == pytest.approx(0.55)

    def test_high_possession_pref_raises(self) -> None:
        result = estimate_possession(
            _baseline(),
            _manager(possession=0.7),
            0.5,
            is_new_manager=False,
        )
        assert result > 0.55

    def test_low_possession_pref_lowers(self) -> None:
        result = estimate_possession(
            _baseline(),
            _manager(possession=0.3),
            0.5,
            is_new_manager=False,
        )
        assert result < 0.55

    def test_clamped_to_zero_one(self) -> None:
        high = estimate_possession(
            _baseline(), _manager(possession=1.0), 0.5, is_new_manager=False
        )
        low = estimate_possession(
            _baseline(), _manager(possession=0.0), 0.5, is_new_manager=False
        )
        assert 0.0 <= high <= 1.0
        assert 0.0 <= low <= 1.0


# ---------------------------------------------------------------------------
# Progressive passes tests
# ---------------------------------------------------------------------------


class TestEstimateProgressivePasses:
    def test_neutral_manager_no_change(self) -> None:
        result = estimate_progressive_passes(
            _baseline(), _manager(), _league(), is_new_manager=False
        )
        assert result == pytest.approx(50.0)

    def test_high_possession_increases(self) -> None:
        result = estimate_progressive_passes(
            _baseline(),
            _manager(possession=0.7),
            _league(),
            is_new_manager=False,
        )
        assert result > 50.0

    def test_attacking_formation_increases(self) -> None:
        result = estimate_progressive_passes(
            _baseline(),
            _manager(formation="3-5-2"),
            _league(),
            is_new_manager=False,
        )
        # 3-5-2 has 7 midfield+forward (vs 6 neutral) → slight boost.
        assert result > 50.0

    def test_zero_league_avg_falls_back(self) -> None:
        league = LeagueContext(1, 1, "Test", avg_progressive_passes_per90=0.0)
        result = estimate_progressive_passes(
            _baseline(), _manager(), league, is_new_manager=True
        )
        assert result == _baseline().progressive_passes_per90

    def test_result_non_negative(self) -> None:
        result = estimate_progressive_passes(
            _baseline(),
            _manager(possession=0.0, formation="5-4-1"),
            _league(),
            is_new_manager=True,
        )
        assert result >= 0.0
