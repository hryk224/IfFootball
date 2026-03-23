"""Tests for fixture_stats converter module."""

from __future__ import annotations

import pandas as pd
import pytest

from iffootball.agents.fixture import FixtureList, OpponentStrength
from iffootball.converters.fixture_stats import (
    _ELO_INITIAL_RATING,
    _ELO_K,
    build_all_opponent_strengths,
    build_fixture_list,
    build_opponent_strength,
    calc_elo_ratings,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_matches(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _make_events(rows: list[dict[str, object]]) -> pd.DataFrame:
    if rows:
        return pd.DataFrame(rows)
    # Return an empty DataFrame with the columns required by calc_team_xg.
    return pd.DataFrame(columns=["match_id", "team", "type", "shot_statsbomb_xg"])


def _basic_matches() -> pd.DataFrame:
    """6-team, 5-week mini season for most tests."""
    return _make_matches([
        # week 1
        {"match_id": 1, "match_week": 1, "home_team": "Arsenal", "away_team": "Chelsea",
         "home_score": 2, "away_score": 1},
        {"match_id": 2, "match_week": 1, "home_team": "Liverpool", "away_team": "Everton",
         "home_score": 1, "away_score": 1},
        # week 2
        {"match_id": 3, "match_week": 2, "home_team": "Chelsea", "away_team": "Liverpool",
         "home_score": 0, "away_score": 2},
        {"match_id": 4, "match_week": 2, "home_team": "Everton", "away_team": "Arsenal",
         "home_score": 0, "away_score": 1},
        # week 3
        {"match_id": 5, "match_week": 3, "home_team": "Arsenal", "away_team": "Liverpool",
         "home_score": 1, "away_score": 0},
        # week 4
        {"match_id": 6, "match_week": 4, "home_team": "Chelsea", "away_team": "Arsenal",
         "home_score": 0, "away_score": 0},
        {"match_id": 7, "match_week": 4, "home_team": "Arsenal", "away_team": "Everton",
         "home_score": 3, "away_score": 0},
        # week 5
        {"match_id": 8, "match_week": 5, "home_team": "Liverpool", "away_team": "Arsenal",
         "home_score": 2, "away_score": 1},
    ])


# ---------------------------------------------------------------------------
# TestBuildFixtureList
# ---------------------------------------------------------------------------


class TestBuildFixtureList:
    def test_full_season_fixtures(self) -> None:
        matches = _basic_matches()
        result = build_fixture_list(matches, "Arsenal")
        # Arsenal plays in weeks 1,2,3,4(x2),5 = 6 fixtures
        assert len(result.fixtures) == 6

    def test_sorted_by_match_week(self) -> None:
        matches = _basic_matches()
        result = build_fixture_list(matches, "Arsenal")
        weeks = [f.match_week for f in result.fixtures]
        assert weeks == sorted(weeks)

    def test_same_week_sorted_by_match_id(self) -> None:
        matches = _basic_matches()
        result = build_fixture_list(matches, "Arsenal")
        week4 = [f for f in result.fixtures if f.match_week == 4]
        # match_id 6 (away at Chelsea) and 7 (home vs Everton) — match_id 6 first
        assert week4[0].opponent_name == "Chelsea"  # match_id=6, away
        assert week4[1].opponent_name == "Everton"   # match_id=7, home

    def test_is_home_flag(self) -> None:
        matches = _basic_matches()
        result = build_fixture_list(matches, "Arsenal")
        # match_id=1: Arsenal home vs Chelsea
        first = result.fixtures[0]
        assert first.opponent_name == "Chelsea"
        assert first.is_home is True
        # match_id=4: Everton home, Arsenal away
        week2 = [f for f in result.fixtures if f.match_week == 2][0]
        assert week2.opponent_name == "Everton"
        assert week2.is_home is False

    def test_team_name_stored(self) -> None:
        matches = _basic_matches()
        result = build_fixture_list(matches, "Arsenal")
        assert result.team_name == "Arsenal"

    def test_returns_fixture_list(self) -> None:
        matches = _basic_matches()
        result = build_fixture_list(matches, "Arsenal")
        assert isinstance(result, FixtureList)

    def test_fixtures_is_tuple(self) -> None:
        matches = _basic_matches()
        result = build_fixture_list(matches, "Arsenal")
        assert isinstance(result.fixtures, tuple)

    def test_team_with_no_matches(self) -> None:
        matches = _basic_matches()
        result = build_fixture_list(matches, "NonExistent")
        assert result.fixtures == ()

    def test_fixture_list_is_immutable(self) -> None:
        matches = _basic_matches()
        result = build_fixture_list(matches, "Arsenal")
        with pytest.raises((AttributeError, TypeError)):
            result.fixtures = ()  # type: ignore[misc]

    def test_after_week_filters_fixtures(self) -> None:
        matches = _basic_matches()
        result = build_fixture_list(matches, "Arsenal", after_week=3)
        # Only week 4 and 5 fixtures
        assert all(f.match_week > 3 for f in result.fixtures)
        assert len(result.fixtures) == 3  # week4 x2 + week5

    def test_after_week_none_returns_all(self) -> None:
        matches = _basic_matches()
        full = build_fixture_list(matches, "Arsenal")
        explicit_none = build_fixture_list(matches, "Arsenal", after_week=None)
        assert len(full.fixtures) == len(explicit_none.fixtures)

    def test_after_week_beyond_season_returns_empty(self) -> None:
        matches = _basic_matches()
        result = build_fixture_list(matches, "Arsenal", after_week=38)
        assert result.fixtures == ()

    def test_fixture_fields_are_immutable(self) -> None:
        matches = _basic_matches()
        result = build_fixture_list(matches, "Arsenal")
        fixture = result.fixtures[0]
        with pytest.raises((AttributeError, TypeError)):
            fixture.is_home = not fixture.is_home  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestCalcEloRatings
# ---------------------------------------------------------------------------


class TestCalcEloRatings:
    def test_all_teams_initialised(self) -> None:
        matches = _basic_matches()
        # At week 0, no matches played → all at initial rating
        result = calc_elo_ratings(matches, up_to_match_week=0)
        for team in ["Arsenal", "Chelsea", "Liverpool", "Everton"]:
            assert result[team] == pytest.approx(_ELO_INITIAL_RATING)

    def test_winner_gains_points(self) -> None:
        matches = _basic_matches()
        result = calc_elo_ratings(matches, up_to_match_week=1)
        # Arsenal beat Chelsea in week 1 → Arsenal rating increases
        assert result["Arsenal"] > _ELO_INITIAL_RATING
        assert result["Chelsea"] < _ELO_INITIAL_RATING

    def test_draw_near_equal_ratings_no_change(self) -> None:
        matches = _basic_matches()
        # Liverpool vs Everton draw in week 1; both start at 1500
        # expected = 0.5 for both; actual = 0.5; delta = K * (0.5 - 0.5) = 0
        result = calc_elo_ratings(matches, up_to_match_week=1)
        assert result["Liverpool"] == pytest.approx(_ELO_INITIAL_RATING)
        assert result["Everton"] == pytest.approx(_ELO_INITIAL_RATING)

    def test_elo_sum_preserved(self) -> None:
        # Total Elo across all teams should be conserved (zero-sum).
        matches = _basic_matches()
        result_before = calc_elo_ratings(matches, up_to_match_week=0)
        result_after = calc_elo_ratings(matches, up_to_match_week=3)
        assert sum(result_before.values()) == pytest.approx(sum(result_after.values()))

    def test_up_to_week_cutoff(self) -> None:
        matches = _basic_matches()
        result_w1 = calc_elo_ratings(matches, up_to_match_week=1)
        result_w2 = calc_elo_ratings(matches, up_to_match_week=2)
        # Chelsea lost week1, won(via Liverpool beat Chelsea) — ratings differ
        assert result_w1["Chelsea"] != result_w2["Chelsea"]

    def test_deterministic_same_week_order(self) -> None:
        # Same result regardless of DataFrame row order.
        matches = _basic_matches()
        shuffled = matches.sample(frac=1, random_state=42).reset_index(drop=True)
        result_orig = calc_elo_ratings(matches, up_to_match_week=3)
        result_shuf = calc_elo_ratings(shuffled, up_to_match_week=3)
        for team in ["Arsenal", "Chelsea", "Liverpool", "Everton"]:
            assert result_orig[team] == pytest.approx(result_shuf[team])

    def test_k_factor_applied(self) -> None:
        # Single match: Arsenal beats Chelsea from equal starting ratings.
        # Expected = 0.5 (equal ratings), actual home = 1.0
        # Arsenal gain = K * (1.0 - 0.5) = K * 0.5 = 10.0
        matches = _make_matches([
            {"match_id": 1, "match_week": 1, "home_team": "Arsenal",
             "away_team": "Chelsea", "home_score": 1, "away_score": 0},
        ])
        result = calc_elo_ratings(matches, up_to_match_week=1)
        assert result["Arsenal"] == pytest.approx(_ELO_INITIAL_RATING + _ELO_K * 0.5)
        assert result["Chelsea"] == pytest.approx(_ELO_INITIAL_RATING - _ELO_K * 0.5)


# ---------------------------------------------------------------------------
# TestBuildOpponentStrength
# ---------------------------------------------------------------------------


class TestBuildOpponentStrength:
    def _events(self) -> pd.DataFrame:
        # Chelsea: 1 shot (xg=0.3) in week1 match; Arsenal shoots at Chelsea
        return _make_events([
            {"match_id": 1, "team": "Arsenal", "type": "Shot", "shot_statsbomb_xg": 0.4},
            {"match_id": 1, "team": "Chelsea", "type": "Shot", "shot_statsbomb_xg": 0.2},
            {"match_id": 3, "team": "Chelsea", "type": "Shot", "shot_statsbomb_xg": 0.3},
            {"match_id": 3, "team": "Liverpool", "type": "Shot", "shot_statsbomb_xg": 0.5},
        ])

    def test_returns_opponent_strength(self) -> None:
        matches = _basic_matches()
        events = self._events()
        elo = calc_elo_ratings(matches, up_to_match_week=3)
        result = build_opponent_strength(events, matches, "Chelsea", 3, elo)
        assert isinstance(result, OpponentStrength)

    def test_opponent_name(self) -> None:
        matches = _basic_matches()
        events = self._events()
        elo = calc_elo_ratings(matches, up_to_match_week=3)
        result = build_opponent_strength(events, matches, "Chelsea", 3, elo)
        assert result.opponent_name == "Chelsea"

    def test_elo_from_precomputed(self) -> None:
        matches = _basic_matches()
        events = self._events()
        elo = calc_elo_ratings(matches, up_to_match_week=3)
        result = build_opponent_strength(events, matches, "Chelsea", 3, elo)
        assert result.elo_rating == pytest.approx(elo["Chelsea"])

    def test_opponent_strength_is_immutable(self) -> None:
        matches = _basic_matches()
        events = _make_events([])
        elo = calc_elo_ratings(matches, up_to_match_week=3)
        result = build_opponent_strength(events, matches, "Chelsea", 3, elo)
        with pytest.raises((AttributeError, TypeError)):
            result.elo_rating = 9999.0  # type: ignore[misc]

    def test_unknown_opponent_uses_initial_elo(self) -> None:
        matches = _basic_matches()
        events = self._events()
        elo: dict[str, float] = {}  # empty dict
        result = build_opponent_strength(events, matches, "Chelsea", 3, elo)
        assert result.elo_rating == pytest.approx(_ELO_INITIAL_RATING)

    def test_xg_uses_pre_trigger_matches_only(self) -> None:
        # trigger_week=1: only match_id=1 for Chelsea
        matches = _basic_matches()
        events = self._events()
        elo = calc_elo_ratings(matches, up_to_match_week=1)
        result_w1 = build_opponent_strength(events, matches, "Chelsea", 1, elo)
        elo3 = calc_elo_ratings(matches, up_to_match_week=3)
        result_w3 = build_opponent_strength(events, matches, "Chelsea", 3, elo3)
        # xG at week1 uses only match_id=1; week3 adds match_id=3
        assert result_w1.xg_for_per90 != result_w3.xg_for_per90


# ---------------------------------------------------------------------------
# TestBuildAllOpponentStrengths
# ---------------------------------------------------------------------------


class TestBuildAllOpponentStrengths:
    def test_returns_dict(self) -> None:
        matches = _basic_matches()
        events = _make_events([])
        fl = build_fixture_list(matches, "Arsenal")
        result = build_all_opponent_strengths(events, matches, fl, trigger_week=3)
        assert isinstance(result, dict)

    def test_keys_are_unique_opponents(self) -> None:
        matches = _basic_matches()
        events = _make_events([])
        fl = build_fixture_list(matches, "Arsenal")
        expected_opponents = {f.opponent_name for f in fl.fixtures}
        result = build_all_opponent_strengths(events, matches, fl, trigger_week=3)
        assert set(result) == expected_opponents

    def test_values_are_opponent_strength(self) -> None:
        matches = _basic_matches()
        events = _make_events([])
        fl = build_fixture_list(matches, "Arsenal")
        result = build_all_opponent_strengths(events, matches, fl, trigger_week=3)
        for v in result.values():
            assert isinstance(v, OpponentStrength)

    def test_duplicate_opponent_computed_once(self) -> None:
        # Create fixture list where same opponent appears twice
        matches = _make_matches([
            {"match_id": 10, "match_week": 4, "home_team": "Arsenal",
             "away_team": "Chelsea", "home_score": 1, "away_score": 0},
            {"match_id": 11, "match_week": 5, "home_team": "Chelsea",
             "away_team": "Arsenal", "home_score": 0, "away_score": 1},
        ])
        events = _make_events([])
        fl = build_fixture_list(matches, "Arsenal")
        result = build_all_opponent_strengths(events, matches, fl, trigger_week=3)
        # Only one entry for Chelsea despite two fixtures
        assert list(result.keys()).count("Chelsea") == 1

    def test_empty_fixture_list(self) -> None:
        matches = _basic_matches()
        events = _make_events([])
        fl = FixtureList(team_name="Arsenal", fixtures=())
        result = build_all_opponent_strengths(events, matches, fl, trigger_week=38)
        assert result == {}
