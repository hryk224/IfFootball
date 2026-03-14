"""Convert StatsBomb matches/events data into TeamBaseline instances.

All aggregation functions accept a frozenset[int] of match IDs as the slice
condition. Callers must ensure all match IDs belong to a single
competition + season. match_week is never exposed in the public API.

M1 metric definitions:
  PPDA                  = opponent Pass events / own (Pressure + Tackle + Interception) events
  progressive_passes    = completed Pass where pass_end_location[0] - location[0] >= 10
                          (StatsBomb pitch x-coordinate advance; pitch range 0–120)
  possession_pct        = team (Pass + Carry) events / all (Pass + Carry) events
  cultural_inertia      = 0.5 placeholder (updated by manager-agent-initialization)
"""

from __future__ import annotations

import pandas as pd

from iffootball.agents.team import TeamBaseline

# Relegation zone size assumed for all supported competitions in M1.
_RELEGATION_SPOTS = 3

# Minimum x-coordinate advance for a progressive pass (StatsBomb pitch coords 0–120).
_PROGRESSIVE_PASS_THRESHOLD = 10.0


def _team_match_ids(
    events: pd.DataFrame,
    team_name: str,
    played_match_ids: frozenset[int],
) -> frozenset[int]:
    """Return the subset of played_match_ids where team_name has at least one event.

    Guards against unrelated matches being silently included in metric aggregations.
    If played_match_ids contains a match where team_name never appears in events,
    that match is excluded rather than silently polluting opponent-side metrics
    (e.g., xg_against, PPDA numerator).
    """
    df = events[events["match_id"].isin(played_match_ids) & (events["team"] == team_name)]
    return frozenset(int(mid) for mid in df["match_id"].dropna().unique())


# ---------------------------------------------------------------------------
# xG
# ---------------------------------------------------------------------------


def calc_team_xg(
    events: pd.DataFrame,
    team_name: str,
    played_match_ids: frozenset[int],
) -> tuple[float, float]:
    """Return (xg_for_per90, xg_against_per90) for team_name.

    Both values are computed as total xG / number of matches where team_name
    has at least one event. Assumes each match is 90 minutes.
    Matches in played_match_ids where team_name has no events are excluded.
    """
    team_ids = _team_match_ids(events, team_name, played_match_ids)
    n = len(team_ids)
    if n == 0:
        return 0.0, 0.0

    df = events[events["match_id"].isin(team_ids)]
    shots = df[(df["type"] == "Shot") & df["shot_statsbomb_xg"].notna()]

    xg_for = float(shots[shots["team"] == team_name]["shot_statsbomb_xg"].sum())
    xg_against = float(shots[shots["team"] != team_name]["shot_statsbomb_xg"].sum())

    return xg_for / n, xg_against / n


# ---------------------------------------------------------------------------
# PPDA
# ---------------------------------------------------------------------------


def calc_ppda(
    events: pd.DataFrame,
    team_name: str,
    played_match_ids: frozenset[int],
) -> float:
    """Return PPDA (Passes Allowed Per Defensive Action) for team_name.

    M1 definition:
      numerator   = opponent Pass events (type == "Pass", team != team_name)
      denominator = own defensive actions (type in Pressure / Tackle / Interception,
                    team == team_name)

    Lower PPDA means higher pressing intensity.
    Returns float("nan") if denominator is zero.
    Matches in played_match_ids where team_name has no events are excluded.
    """
    team_ids = _team_match_ids(events, team_name, played_match_ids)
    df = events[events["match_id"].isin(team_ids)].dropna(subset=["team"])

    opponent_passes = int(
        ((df["type"] == "Pass") & (df["team"] != team_name)).sum()
    ) if not df.empty else 0
    own_def_actions = int(
        df[(df["team"] == team_name) & (df["type"].isin(["Pressure", "Tackle", "Interception"]))].shape[0]
    )

    if own_def_actions == 0:
        return float("nan")
    return float(opponent_passes) / float(own_def_actions)


# ---------------------------------------------------------------------------
# Progressive passes
# ---------------------------------------------------------------------------


def _is_progressive_pass(row: pd.Series) -> bool:
    """Return True if the pass is a completed progressive pass.

    Criteria:
      - type == "Pass"
      - pass_outcome is NaN (completed pass)
      - pass_end_location[0] - location[0] >= _PROGRESSIVE_PASS_THRESHOLD
        (advances >= 10 pitch x-coordinate units toward opponent goal)
    """
    if row.get("type") != "Pass":
        return False
    if pd.notna(row.get("pass_outcome")):
        return False
    loc = row.get("location")
    end_loc = row.get("pass_end_location")
    if not (isinstance(loc, list) and isinstance(end_loc, list)):
        return False
    return bool((end_loc[0] - loc[0]) >= _PROGRESSIVE_PASS_THRESHOLD)


def calc_progressive_passes_per90(
    events: pd.DataFrame,
    team_name: str,
    played_match_ids: frozenset[int],
) -> float:
    """Return progressive passes per 90 min for team_name.

    A progressive pass is a completed pass that advances the ball >= 10
    pitch x-coordinate units (StatsBomb range 0–120) toward the opponent's goal.

    Total minutes assumed = (matches where team_name has events) * 90.
    Matches in played_match_ids where team_name has no events are excluded.
    """
    team_ids = _team_match_ids(events, team_name, played_match_ids)
    n = len(team_ids)
    if n == 0:
        return 0.0

    df = events[
        (events["match_id"].isin(team_ids)) & (events["team"] == team_name)
    ]
    count = int(df.apply(_is_progressive_pass, axis=1).sum())
    return count / n


# ---------------------------------------------------------------------------
# Possession
# ---------------------------------------------------------------------------


def calc_possession_pct(
    events: pd.DataFrame,
    team_name: str,
    played_match_ids: frozenset[int],
) -> float:
    """Return possession percentage for team_name as a ratio (0.0–1.0).

    Possession is approximated as:
      team (Pass + Carry) events / all (Pass + Carry) events

    Returns 0.0 if no pass or carry events are found.
    Matches in played_match_ids where team_name has no events are excluded.
    """
    team_ids = _team_match_ids(events, team_name, played_match_ids)
    df = events[
        events["match_id"].isin(team_ids) & events["type"].isin(["Pass", "Carry"])
    ].dropna(subset=["team"])

    total = len(df)
    if total == 0:
        return 0.0

    own = int((df["team"] == team_name).sum())
    return own / total


# ---------------------------------------------------------------------------
# League standing
# ---------------------------------------------------------------------------


def _compute_points_table(
    matches: pd.DataFrame,
    up_to_match_week: int,
) -> dict[str, int]:
    """Return a dict mapping team_name -> points for all matches up to match_week."""
    played = matches[matches["match_week"] <= up_to_match_week].dropna(
        subset=["home_score", "away_score"]
    )
    points: dict[str, int] = {}
    for _, row in played.iterrows():
        home = str(row["home_team"])
        away = str(row["away_team"])
        hs = int(row["home_score"])
        as_ = int(row["away_score"])
        points.setdefault(home, 0)
        points.setdefault(away, 0)
        if hs > as_:
            points[home] += 3
        elif hs == as_:
            points[home] += 1
            points[away] += 1
        else:
            points[away] += 3
    return points


def calc_league_standing(
    matches: pd.DataFrame,
    team_name: str,
    played_match_ids: frozenset[int],
) -> tuple[int, int, int, int]:
    """Return (league_position, points_to_safety, points_to_title, matches_remaining).

    played_match_ids is expected to be the complete set of all matches played
    by team_name up to a given point in the season (not an arbitrary subset).
    All league-wide calculations use the latest match_week derived from
    played_match_ids as a consistent cutoff for the whole table.

    League position uses ALL teams' results up to cutoff_week, ensuring a
    consistent snapshot across all teams. match_week is derived internally
    and is not part of the public API.

    Tie-breaking for equal points uses team_name alphabetically so that
    results are fully deterministic and reproducible.

    points_to_safety:
        team_points − max(bottom-3 teams' points).
        Positive = above relegation zone, negative = in relegation zone.

    points_to_title:
        team_points − leader_points. Always <= 0.

    matches_remaining:
        Total season matches for team_name − matches played by team_name
        up to cutoff_week (consistent with the league table cutoff).
    """
    played_rows = matches[matches["match_id"].isin(played_match_ids)]
    if played_rows.empty:
        return 0, 0, 0, 0

    cutoff_week = int(played_rows["match_week"].max())
    points_table = _compute_points_table(matches, cutoff_week)

    if team_name not in points_table:
        return 0, 0, 0, 0

    # Sort by (-points, team_name) for deterministic tie-breaking.
    all_teams = sorted(points_table, key=lambda t: (-points_table[t], t))
    n_teams = len(all_teams)

    team_pts = points_table[team_name]
    leader_pts = points_table[all_teams[0]]

    # Relegation zone = bottom _RELEGATION_SPOTS teams.
    relegation_max_pts = points_table[all_teams[n_teams - _RELEGATION_SPOTS]]
    points_to_safety = team_pts - relegation_max_pts
    points_to_title = team_pts - leader_pts

    league_position = all_teams.index(team_name) + 1

    # matches_remaining uses cutoff_week as the reference point, consistent
    # with how the league table is computed.
    total_in_season = int(
        ((matches["home_team"] == team_name) | (matches["away_team"] == team_name)).sum()
    )
    matches_played_by_cutoff = int(
        (
            (matches["match_week"] <= cutoff_week)
            & ((matches["home_team"] == team_name) | (matches["away_team"] == team_name))
        ).sum()
    )
    matches_remaining = total_in_season - matches_played_by_cutoff

    return league_position, int(points_to_safety), int(points_to_title), matches_remaining


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_team_baseline(
    events: pd.DataFrame,
    matches: pd.DataFrame,
    team_name: str,
    played_match_ids: frozenset[int],
    competition_id: int,
    season_id: int,
) -> TeamBaseline:
    """Build a TeamBaseline for team_name over the given set of played matches.

    Args:
        events:           Combined match events DataFrame. Must include all events
                          for matches in played_match_ids.
        matches:          Matches DataFrame for the full competition/season.
                          Used for league standing and matches_remaining.
        team_name:        StatsBomb team name (exact spelling required).
        played_match_ids: Complete set of all matches played by team_name up to
                          the desired snapshot point. Caller must ensure all IDs
                          belong to a single competition + season, and that this
                          represents the full history of the team (not an
                          arbitrary subset), because league standing calculations
                          derive a cutoff_week from this set.
        competition_id:   StatsBomb competition ID. Not derived from matches to
                          avoid ambiguity when matches from multiple competitions
                          are present.
        season_id:        StatsBomb season ID.

    Returns:
        TeamBaseline with cultural_inertia set to 0.5 placeholder.
    """
    xg_for, xg_against = calc_team_xg(events, team_name, played_match_ids)
    ppda = calc_ppda(events, team_name, played_match_ids)
    prog_passes = calc_progressive_passes_per90(events, team_name, played_match_ids)
    possession = calc_possession_pct(events, team_name, played_match_ids)
    position, to_safety, to_title, remaining = calc_league_standing(
        matches, team_name, played_match_ids
    )

    return TeamBaseline(
        team_name=team_name,
        competition_id=competition_id,
        season_id=season_id,
        played_match_ids=played_match_ids,
        xg_for_per90=xg_for,
        xg_against_per90=xg_against,
        ppda=ppda,
        progressive_passes_per90=prog_passes,
        possession_pct=possession,
        league_position=position,
        points_to_safety=to_safety,
        points_to_title=to_title,
        matches_remaining=remaining,
    )
