"""Tests for team_stats conversion layer.

Aggregation formula tests are kept thick, as metric definitions are M1-specific
and subtle bugs (PPDA numerator/denominator swap, progressive pass threshold, etc.)
would be hard to catch at integration level.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from iffootball.agents.team import TeamBaseline
from iffootball.converters.team_stats import (
    _team_match_ids,
    build_team_baseline,
    calc_league_standing,
    calc_possession_pct,
    calc_ppda,
    calc_progressive_passes_per90,
    calc_team_xg,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

TEAM = "Manchester United"
OPP = "Arsenal"


def _matches(rows: list[dict]) -> pd.DataFrame:  # type: ignore[type-arg]
    return pd.DataFrame(rows)


def _events(rows: list[dict]) -> pd.DataFrame:  # type: ignore[type-arg]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _team_match_ids
# ---------------------------------------------------------------------------


class TestTeamMatchIds:
    def test_returns_only_matches_with_team_events(self) -> None:
        events = _events(
            [
                {"match_id": 1, "team": TEAM},
                {"match_id": 2, "team": OPP},   # TEAM has no events here
            ]
        )
        result = _team_match_ids(events, TEAM, frozenset([1, 2]))
        assert result == frozenset([1])

    def test_unrelated_match_excluded(self) -> None:
        # match 99 has events but not for TEAM
        events = _events([{"match_id": 99, "team": "Chelsea"}])
        result = _team_match_ids(events, TEAM, frozenset([99]))
        assert result == frozenset()

    def test_subset_of_played_ids_returned(self) -> None:
        events = _events(
            [
                {"match_id": 1, "team": TEAM},
                {"match_id": 2, "team": TEAM},
            ]
        )
        result = _team_match_ids(events, TEAM, frozenset([1, 2, 3]))
        assert result == frozenset([1, 2])


def _base_match(
    match_id: int,
    match_week: int,
    home_team: str = TEAM,
    away_team: str = OPP,
    home_score: int = 1,
    away_score: int = 0,
) -> dict:  # type: ignore[type-arg]
    return {
        "match_id": match_id,
        "match_week": match_week,
        "home_team": home_team,
        "away_team": away_team,
        "home_score": home_score,
        "away_score": away_score,
    }


# ---------------------------------------------------------------------------
# calc_team_xg
# ---------------------------------------------------------------------------


class TestCalcTeamXg:
    def test_xg_for_sums_own_shots(self) -> None:
        events = _events(
            [
                {"match_id": 1, "type": "Shot", "team": TEAM, "shot_statsbomb_xg": 0.3},
                {"match_id": 1, "type": "Shot", "team": TEAM, "shot_statsbomb_xg": 0.2},
            ]
        )
        xg_for, _ = calc_team_xg(events, TEAM, frozenset([1]))
        assert xg_for == pytest.approx(0.5)

    def test_xg_against_sums_opponent_shots(self) -> None:
        # TEAM must have at least one event for the match to be included.
        events = _events(
            [
                {"match_id": 1, "type": "Pass", "team": TEAM},  # TEAM presence
                {"match_id": 1, "type": "Shot", "team": OPP, "shot_statsbomb_xg": 0.4},
            ]
        )
        _, xg_against = calc_team_xg(events, TEAM, frozenset([1]))
        assert xg_against == pytest.approx(0.4)

    def test_xg_divided_by_match_count(self) -> None:
        # 2 matches, 0.6 total xg_for → 0.3 per match (per 90)
        events = _events(
            [
                {"match_id": 1, "type": "Shot", "team": TEAM, "shot_statsbomb_xg": 0.3},
                {"match_id": 2, "type": "Shot", "team": TEAM, "shot_statsbomb_xg": 0.3},
            ]
        )
        xg_for, _ = calc_team_xg(events, TEAM, frozenset([1, 2]))
        assert xg_for == pytest.approx(0.3)

    def test_empty_match_ids_returns_zero(self) -> None:
        events = _events([{"match_id": 1, "type": "Shot", "team": TEAM, "shot_statsbomb_xg": 0.5}])
        xg_for, xg_against = calc_team_xg(events, TEAM, frozenset())
        assert xg_for == 0.0
        assert xg_against == 0.0

    def test_matches_outside_played_ids_excluded(self) -> None:
        events = _events(
            [
                {"match_id": 1, "type": "Shot", "team": TEAM, "shot_statsbomb_xg": 0.5},
                {"match_id": 99, "type": "Shot", "team": TEAM, "shot_statsbomb_xg": 9.9},
            ]
        )
        xg_for, _ = calc_team_xg(events, TEAM, frozenset([1]))
        assert xg_for == pytest.approx(0.5)

    def test_nan_xg_values_excluded(self) -> None:
        events = _events(
            [
                {"match_id": 1, "type": "Shot", "team": TEAM, "shot_statsbomb_xg": None},
                {"match_id": 1, "type": "Shot", "team": TEAM, "shot_statsbomb_xg": 0.3},
            ]
        )
        xg_for, _ = calc_team_xg(events, TEAM, frozenset([1]))
        assert xg_for == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# calc_ppda
# ---------------------------------------------------------------------------


class TestCalcPpda:
    def test_basic_ppda_ratio(self) -> None:
        # 10 opponent passes / 2 own defensive actions = 5.0
        events = _events(
            [{"match_id": 1, "type": "Pass", "team": OPP}] * 10
            + [{"match_id": 1, "type": "Pressure", "team": TEAM}] * 2
        )
        result = calc_ppda(events, TEAM, frozenset([1]))
        assert result == pytest.approx(5.0)

    def test_ppda_uses_pressure_tackle_interception_as_denominator(self) -> None:
        events = _events(
            [
                {"match_id": 1, "type": "Pass", "team": OPP},
                {"match_id": 1, "type": "Pass", "team": OPP},
                {"match_id": 1, "type": "Pass", "team": OPP},
                {"match_id": 1, "type": "Pressure", "team": TEAM},
                {"match_id": 1, "type": "Tackle", "team": TEAM},
                {"match_id": 1, "type": "Interception", "team": TEAM},
            ]
        )
        result = calc_ppda(events, TEAM, frozenset([1]))
        assert result == pytest.approx(3.0 / 3.0)

    def test_ppda_own_passes_excluded_from_numerator(self) -> None:
        # Own passes must NOT count as "opponent passes"
        events = _events(
            [
                {"match_id": 1, "type": "Pass", "team": TEAM},  # own pass — excluded
                {"match_id": 1, "type": "Pass", "team": OPP},   # opponent pass — included
                {"match_id": 1, "type": "Pressure", "team": TEAM},
            ]
        )
        result = calc_ppda(events, TEAM, frozenset([1]))
        assert result == pytest.approx(1.0)

    def test_ppda_zero_defensive_actions_returns_nan(self) -> None:
        events = _events([{"match_id": 1, "type": "Pass", "team": OPP}])
        result = calc_ppda(events, TEAM, frozenset([1]))
        assert math.isnan(result)

    def test_ppda_nan_preserved_when_no_defensive_actions(self) -> None:
        # PPDA NaN must NOT be silently converted to 0.0
        # (0.0 would mean "maximum pressing", which is misleading for missing data)
        events = _events([{"match_id": 1, "type": "Pass", "team": OPP}])
        result = calc_ppda(events, TEAM, frozenset([1]))
        assert math.isnan(result)

    def test_ppda_excludes_unrelated_match(self) -> None:
        # match 2 has no TEAM events → excluded. Ratio is based on match 1 only.
        events = _events(
            [
                {"match_id": 1, "type": "Pass", "team": OPP},
                {"match_id": 1, "type": "Pressure", "team": TEAM},
                # match 2: third team only — should not affect TEAM's PPDA
                {"match_id": 2, "type": "Pass", "team": "Chelsea"},
                {"match_id": 2, "type": "Pass", "team": "Liverpool"},
            ]
        )
        result = calc_ppda(events, TEAM, frozenset([1, 2]))
        assert result == pytest.approx(1.0)

    def test_lower_ppda_means_more_pressing(self) -> None:
        # High pressing: few passes allowed per defensive action
        high_press = _events(
            [{"match_id": 1, "type": "Pass", "team": OPP}] * 5
            + [{"match_id": 1, "type": "Pressure", "team": TEAM}] * 5
        )
        low_press = _events(
            [{"match_id": 1, "type": "Pass", "team": OPP}] * 20
            + [{"match_id": 1, "type": "Pressure", "team": TEAM}] * 5
        )
        assert calc_ppda(high_press, TEAM, frozenset([1])) < calc_ppda(
            low_press, TEAM, frozenset([1])
        )


# ---------------------------------------------------------------------------
# calc_progressive_passes_per90
# ---------------------------------------------------------------------------


class TestCalcProgressivePasses:
    def _make_pass(
        self,
        match_id: int,
        team: str,
        start_x: float,
        end_x: float,
        outcome: object = None,
    ) -> dict:  # type: ignore[type-arg]
        return {
            "match_id": match_id,
            "type": "Pass",
            "team": team,
            "location": [start_x, 40.0],
            "pass_end_location": [end_x, 40.0],
            "pass_outcome": outcome,
        }

    def test_progressive_pass_counted(self) -> None:
        # advance = 15 >= 10: progressive
        events = _events([self._make_pass(1, TEAM, 40.0, 55.0)])
        result = calc_progressive_passes_per90(events, TEAM, frozenset([1]))
        assert result == pytest.approx(1.0)

    def test_advance_below_threshold_not_counted(self) -> None:
        # advance = 9 < 10: not progressive
        events = _events([self._make_pass(1, TEAM, 40.0, 49.0)])
        result = calc_progressive_passes_per90(events, TEAM, frozenset([1]))
        assert result == pytest.approx(0.0)

    def test_exact_threshold_is_counted(self) -> None:
        # advance == 10: boundary, should be counted
        events = _events([self._make_pass(1, TEAM, 40.0, 50.0)])
        result = calc_progressive_passes_per90(events, TEAM, frozenset([1]))
        assert result == pytest.approx(1.0)

    def test_incomplete_pass_not_counted(self) -> None:
        events = _events([self._make_pass(1, TEAM, 40.0, 55.0, outcome="Incomplete")])
        result = calc_progressive_passes_per90(events, TEAM, frozenset([1]))
        assert result == pytest.approx(0.0)

    def test_opponent_progressive_pass_not_counted(self) -> None:
        events = _events([self._make_pass(1, OPP, 40.0, 55.0)])
        result = calc_progressive_passes_per90(events, TEAM, frozenset([1]))
        assert result == pytest.approx(0.0)

    def test_backward_pass_not_counted(self) -> None:
        # advance = -10 (backward): not progressive
        events = _events([self._make_pass(1, TEAM, 55.0, 40.0)])
        result = calc_progressive_passes_per90(events, TEAM, frozenset([1]))
        assert result == pytest.approx(0.0)

    def test_divided_by_match_count(self) -> None:
        # 2 progressive passes across 2 matches → 1.0 per match
        events = _events(
            [
                self._make_pass(1, TEAM, 40.0, 55.0),
                self._make_pass(2, TEAM, 40.0, 55.0),
            ]
        )
        result = calc_progressive_passes_per90(events, TEAM, frozenset([1, 2]))
        assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# calc_possession_pct
# ---------------------------------------------------------------------------


class TestCalcPossessionPct:
    def test_equal_events_returns_half(self) -> None:
        events = _events(
            [
                {"match_id": 1, "type": "Pass", "team": TEAM},
                {"match_id": 1, "type": "Pass", "team": OPP},
            ]
        )
        result = calc_possession_pct(events, TEAM, frozenset([1]))
        assert result == pytest.approx(0.5)

    def test_possession_counts_pass_and_carry(self) -> None:
        events = _events(
            [
                {"match_id": 1, "type": "Pass", "team": TEAM},
                {"match_id": 1, "type": "Carry", "team": TEAM},
                {"match_id": 1, "type": "Pass", "team": OPP},
                {"match_id": 1, "type": "Carry", "team": OPP},
            ]
        )
        result = calc_possession_pct(events, TEAM, frozenset([1]))
        assert result == pytest.approx(0.5)

    def test_non_pass_carry_events_excluded(self) -> None:
        # Shot and Pressure do not count toward possession
        events = _events(
            [
                {"match_id": 1, "type": "Pass", "team": TEAM},
                {"match_id": 1, "type": "Shot", "team": OPP},  # excluded
                {"match_id": 1, "type": "Pressure", "team": OPP},  # excluded
            ]
        )
        result = calc_possession_pct(events, TEAM, frozenset([1]))
        assert result == pytest.approx(1.0)

    def test_no_relevant_events_returns_zero(self) -> None:
        events = _events([{"match_id": 1, "type": "Shot", "team": TEAM}])
        result = calc_possession_pct(events, TEAM, frozenset([1]))
        assert result == 0.0

    def test_range_is_zero_to_one(self) -> None:
        events = _events(
            [{"match_id": 1, "type": "Pass", "team": TEAM}] * 7
            + [{"match_id": 1, "type": "Pass", "team": OPP}] * 3
        )
        result = calc_possession_pct(events, TEAM, frozenset([1]))
        assert 0.0 <= result <= 1.0
        assert result == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# calc_league_standing
# ---------------------------------------------------------------------------


def _make_season_matches() -> pd.DataFrame:
    """10-team mini-season, 2 match_weeks, each team plays once per week."""
    teams = [TEAM, OPP, "Chelsea", "Liverpool", "Man City",
             "Tottenham", "Leicester", "Everton", "Wolves", "Burnley"]
    rows = []
    mid = 1
    # week 1: first 5 teams play the second 5
    for i in range(5):
        rows.append(_base_match(mid, 1, teams[i], teams[i + 5], 1, 0))
        mid += 1
    # week 2: same pairings, reverse scores
    for i in range(5):
        rows.append(_base_match(mid, 2, teams[i + 5], teams[i], 1, 0))
        mid += 1
    return _matches(rows)


class TestCalcLeagueStanding:
    def test_returns_correct_position_after_one_win(self) -> None:
        # TEAM wins week 1, OPP loses → TEAM above OPP
        matches = _make_season_matches()
        played = frozenset(matches[matches["match_week"] == 1]["match_id"].tolist())
        pos, _, _, _ = calc_league_standing(matches, TEAM, played)
        # TEAM should be in top half after winning
        assert pos <= 5

    def test_points_to_title_is_nonpositive(self) -> None:
        matches = _make_season_matches()
        played = frozenset(matches[matches["match_week"] == 1]["match_id"].tolist())
        _, _, to_title, _ = calc_league_standing(matches, TEAM, played)
        assert to_title <= 0

    def test_matches_remaining_decreases_after_more_matches(self) -> None:
        matches = _make_season_matches()
        played_w1 = frozenset(matches[matches["match_week"] == 1]["match_id"].tolist())
        played_w2 = frozenset(matches["match_id"].tolist())
        _, _, _, rem_w1 = calc_league_standing(matches, TEAM, played_w1)
        _, _, _, rem_w2 = calc_league_standing(matches, TEAM, played_w2)
        assert rem_w1 > rem_w2

    def test_team_in_relegation_zone_has_negative_points_to_safety(self) -> None:
        # Build a scenario where TEAM has 0 points after 1 match (lost)
        matches = _matches(
            [_base_match(1, 1, TEAM, OPP, 0, 3)]
            + [_base_match(mid, 1, f"T{mid}", f"T{mid + 10}", 1, 0) for mid in range(2, 10)]
        )
        played = frozenset([1])
        _, to_safety, _, _ = calc_league_standing(matches, TEAM, played)
        # TEAM has 0 pts; others have 3 pts → TEAM is in relegation zone
        assert to_safety <= 0

    def test_points_to_safety_definition(self) -> None:
        # All teams equal points → to_safety == 0 (or could be at exact boundary)
        matches = _matches(
            [_base_match(i, 1, f"T{i}", f"T{i + 10}", 1, 1) for i in range(1, 11)]
        )
        # TEAM is not in this league — ensure we handle gracefully
        played = frozenset(matches["match_id"].tolist())
        pos, to_safety, to_title, _ = calc_league_standing(matches, "T1", played)
        # All draws → all equal points → to_title == 0
        assert to_title == 0

    def test_empty_played_ids_returns_zeros(self) -> None:
        matches = _make_season_matches()
        result = calc_league_standing(matches, TEAM, frozenset())
        assert result == (0, 0, 0, 0)

    def test_tiebreak_is_deterministic_by_team_name(self) -> None:
        # Two teams with identical points must always produce the same ranking order.
        # "Arsenal" sorts before "Manchester United" alphabetically.
        matches = _matches(
            [
                _base_match(1, 1, TEAM, OPP, 1, 1),   # draw: both get 1 pt
                _base_match(2, 1, "Chelsea", "Liverpool", 0, 0),   # filler
                _base_match(3, 1, "Everton", "Wolves", 0, 0),      # filler
                _base_match(4, 1, "Burnley", "Fulham", 0, 0),      # filler
            ]
        )
        played = frozenset([1])
        pos_team, _, _, _ = calc_league_standing(matches, TEAM, played)
        pos_opp, _, _, _ = calc_league_standing(matches, OPP, played)
        # OPP == "Arsenal" < "Manchester United" → Arsenal ranks first
        assert pos_opp < pos_team

    def test_matches_remaining_uses_cutoff_week(self) -> None:
        # Even if played_match_ids contains only week-2 match, matches_remaining
        # should be based on cutoff_week=2 for all teams, not just played count.
        matches = _make_season_matches()
        # Only pass week-2 matches for TEAM (skip week-1)
        week2_for_team = frozenset(
            matches[(matches["match_week"] == 2)
                    & ((matches["home_team"] == TEAM) | (matches["away_team"] == TEAM))
                    ]["match_id"].tolist()
        )
        _, _, _, remaining = calc_league_standing(matches, TEAM, week2_for_team)
        # After cutoff_week=2, 0 matches remain in the 2-week season
        assert remaining == 0


# ---------------------------------------------------------------------------
# build_team_baseline (integration)
# ---------------------------------------------------------------------------


class TestBuildTeamBaseline:
    def _make_events(self) -> pd.DataFrame:
        return _events(
            [
                {"match_id": 1, "type": "Shot", "team": TEAM, "shot_statsbomb_xg": 0.3},
                {"match_id": 1, "type": "Pass", "team": TEAM,
                 "location": [40.0, 40.0], "pass_end_location": [55.0, 40.0], "pass_outcome": None},
                {"match_id": 1, "type": "Pass", "team": TEAM},
                {"match_id": 1, "type": "Pass", "team": OPP},
                {"match_id": 1, "type": "Pressure", "team": TEAM},
            ]
        )

    def _make_matches(self) -> pd.DataFrame:
        return _matches(
            [_base_match(1, 1, TEAM, OPP, 1, 0)]
            + [_base_match(mid, 1, f"T{mid}", f"T{mid + 10}", 0, 0) for mid in range(2, 10)]
        )

    def test_returns_team_baseline_instance(self) -> None:
        baseline = build_team_baseline(
            self._make_events(), self._make_matches(), TEAM, frozenset([1]), 2, 27
        )
        assert isinstance(baseline, TeamBaseline)

    def test_team_name_and_ids_stored(self) -> None:
        baseline = build_team_baseline(
            self._make_events(), self._make_matches(), TEAM, frozenset([1]), 2, 27
        )
        assert baseline.team_name == TEAM
        assert baseline.competition_id == 2
        assert baseline.season_id == 27
        assert baseline.played_match_ids == frozenset([1])

    def test_xg_for_positive(self) -> None:
        baseline = build_team_baseline(
            self._make_events(), self._make_matches(), TEAM, frozenset([1]), 2, 27
        )
        assert baseline.xg_for_per90 > 0.0

    def test_possession_pct_in_range(self) -> None:
        baseline = build_team_baseline(
            self._make_events(), self._make_matches(), TEAM, frozenset([1]), 2, 27
        )
        assert 0.0 <= baseline.possession_pct <= 1.0

    def test_cultural_inertia_is_placeholder(self) -> None:
        baseline = build_team_baseline(
            self._make_events(), self._make_matches(), TEAM, frozenset([1]), 2, 27
        )
        assert baseline.cultural_inertia == pytest.approx(0.5)

    def test_ppda_nan_propagated_to_baseline(self) -> None:
        # When no defensive actions exist, ppda must be NaN in the TeamBaseline.
        # It must NOT be silently replaced with 0.0.
        events_no_defense = _events(
            [
                {"match_id": 1, "type": "Shot", "team": TEAM, "shot_statsbomb_xg": 0.3},
                {"match_id": 1, "type": "Pass", "team": TEAM,
                 "location": [40.0, 40.0], "pass_end_location": [55.0, 40.0], "pass_outcome": None},
                {"match_id": 1, "type": "Pass", "team": OPP},
                # No Pressure / Tackle / Interception for TEAM
            ]
        )
        baseline = build_team_baseline(
            events_no_defense, self._make_matches(), TEAM, frozenset([1]), 2, 27
        )
        assert math.isnan(baseline.ppda)
