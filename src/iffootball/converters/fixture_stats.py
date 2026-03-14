"""Convert StatsBomb matches/events data into FixtureList and OpponentStrength.

Pipeline:
  1. build_fixture_list()           — remaining fixtures after trigger_week
  2. calc_elo_ratings()             — Elo ratings for all teams up to trigger_week
  3. build_opponent_strength()      — xG + Elo → OpponentStrength per opponent
  4. build_all_opponent_strengths() — batch-build for all opponents in FixtureList

Elo specification (M1):
  Initial rating : _ELO_INITIAL_RATING = 1500.0 (all teams)
  K-factor       : _ELO_K = 20.0 (Club Elo standard for club-level matches)
  Update formula : new = old + K * (actual - expected)
                   expected = 1 / (1 + 10 ** ((rating_opp - rating_team) / 400))
                   actual   : win=1.0, draw=0.5, loss=0.0
  Processing order: match_week ascending, then match_id ascending within the
                    same week (deterministic).
  Snapshot scope : only matches with match_week <= trigger_week are used.
"""

from __future__ import annotations

import pandas as pd

from iffootball.agents.fixture import Fixture, FixtureList, OpponentStrength
from iffootball.converters.team_stats import calc_team_xg

# Elo constants — adjust here if recalibration is needed in future milestones.
_ELO_INITIAL_RATING = 1500.0
_ELO_K = 20.0


# ---------------------------------------------------------------------------
# FixtureList
# ---------------------------------------------------------------------------


def build_fixture_list(
    matches: pd.DataFrame,
    team_name: str,
    trigger_week: int,
) -> FixtureList:
    """Return remaining fixtures for team_name after trigger_week.

    Only matches with match_week > trigger_week are included.
    Fixtures are sorted by match_week ascending, then match_id ascending
    within the same week for deterministic ordering.

    Args:
        matches:      Matches DataFrame for the full competition/season.
                      Must include match_id, match_week, home_team, away_team.
        team_name:    StatsBomb team name of the simulated team.
        trigger_week: Trigger injection point. Fixtures at this week and
                      earlier are excluded (the trigger takes effect from
                      trigger_week + 1).

    Returns:
        FixtureList with fixtures sorted by (match_week, match_id).
    """
    team_mask = (matches["home_team"] == team_name) | (matches["away_team"] == team_name)
    remaining = matches[team_mask & (matches["match_week"] > trigger_week)].copy()
    remaining = remaining.sort_values(["match_week", "match_id"])

    fixtures: list[Fixture] = []
    for _, row in remaining.iterrows():
        is_home = bool(row["home_team"] == team_name)
        opponent_name = str(row["away_team"] if is_home else row["home_team"])
        fixtures.append(
            Fixture(
                match_week=int(row["match_week"]),
                opponent_name=opponent_name,
                is_home=is_home,
            )
        )

    return FixtureList(
        team_name=team_name,
        trigger_week=trigger_week,
        fixtures=tuple(fixtures),
    )


# ---------------------------------------------------------------------------
# Elo ratings
# ---------------------------------------------------------------------------


def calc_elo_ratings(
    matches: pd.DataFrame,
    up_to_match_week: int,
) -> dict[str, float]:
    """Return Elo ratings for all teams after processing matches up to the given week.

    Ratings are initialised at _ELO_INITIAL_RATING (1500.0) for all teams
    that appear in the dataset. Matches are processed in match_week ascending,
    then match_id ascending order to ensure deterministic results.

    Only matches with match_week <= up_to_match_week and non-null scores are
    processed.

    Args:
        matches:          Matches DataFrame for the full competition/season.
                          Must include match_id, match_week, home_team,
                          away_team, home_score, away_score.
        up_to_match_week: Latest match_week to include (inclusive).

    Returns:
        Dict mapping team_name → Elo rating (float).
    """
    played = matches[
        (matches["match_week"] <= up_to_match_week)
    ].dropna(subset=["home_score", "away_score"]).sort_values(["match_week", "match_id"])

    # Initialise all teams that appear anywhere in the full matches DataFrame.
    all_teams = set(matches["home_team"]) | set(matches["away_team"])
    ratings: dict[str, float] = {team: _ELO_INITIAL_RATING for team in all_teams}

    for _, row in played.iterrows():
        home = str(row["home_team"])
        away = str(row["away_team"])
        hs = int(row["home_score"])
        as_ = int(row["away_score"])

        r_home = ratings[home]
        r_away = ratings[away]

        expected_home = 1.0 / (1.0 + 10.0 ** ((r_away - r_home) / 400.0))
        expected_away = 1.0 - expected_home

        if hs > as_:
            actual_home, actual_away = 1.0, 0.0
        elif hs == as_:
            actual_home, actual_away = 0.5, 0.5
        else:
            actual_home, actual_away = 0.0, 1.0

        ratings[home] = r_home + _ELO_K * (actual_home - expected_home)
        ratings[away] = r_away + _ELO_K * (actual_away - expected_away)

    return ratings


# ---------------------------------------------------------------------------
# OpponentStrength
# ---------------------------------------------------------------------------


def build_opponent_strength(
    events: pd.DataFrame,
    matches: pd.DataFrame,
    opponent_name: str,
    trigger_week: int,
    elo_ratings: dict[str, float],
) -> OpponentStrength:
    """Build OpponentStrength for a single opponent at trigger_week.

    xG metrics are computed over all matches played by opponent_name up to
    and including trigger_week. Elo rating is taken from the pre-computed
    elo_ratings dict.

    Args:
        events:        Combined events DataFrame. Must cover all matches up
                       to trigger_week for accurate xG aggregation.
        matches:       Matches DataFrame for the full competition/season.
        opponent_name: StatsBomb team name of the opponent.
        trigger_week:  Snapshot point; only matches up to this week are used.
        elo_ratings:   Pre-computed Elo ratings dict (from calc_elo_ratings).

    Returns:
        OpponentStrength with xG and Elo metrics at trigger_week.
    """
    opponent_mask = (
        (matches["home_team"] == opponent_name) | (matches["away_team"] == opponent_name)
    )
    pre_trigger = matches[
        opponent_mask & (matches["match_week"] <= trigger_week)
    ].dropna(subset=["home_score", "away_score"])

    played_match_ids = frozenset(int(mid) for mid in pre_trigger["match_id"])

    xg_for, xg_against = calc_team_xg(events, opponent_name, played_match_ids)
    elo = elo_ratings.get(opponent_name, _ELO_INITIAL_RATING)

    return OpponentStrength(
        opponent_name=opponent_name,
        xg_for_per90=xg_for,
        xg_against_per90=xg_against,
        elo_rating=elo,
    )


def build_all_opponent_strengths(
    events: pd.DataFrame,
    matches: pd.DataFrame,
    fixture_list: FixtureList,
) -> dict[str, OpponentStrength]:
    """Build OpponentStrength for every unique opponent in fixture_list.

    Opponents are deduplicated from FixtureList.fixtures so that teams
    appearing multiple times (e.g., cup replay) are computed only once.
    Elo ratings are computed once for all teams and shared across opponents.

    Args:
        events:       Combined events DataFrame for all matches up to
                      fixture_list.trigger_week.
        matches:      Matches DataFrame for the full competition/season.
        fixture_list: FixtureList whose opponents will be built.

    Returns:
        Dict mapping opponent_name → OpponentStrength.
    """
    trigger_week = fixture_list.trigger_week
    elo_ratings = calc_elo_ratings(matches, trigger_week)

    unique_opponents = {f.opponent_name for f in fixture_list.fixtures}

    return {
        opp: build_opponent_strength(events, matches, opp, trigger_week, elo_ratings)
        for opp in unique_opponents
    }
