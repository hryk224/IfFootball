"""Convert StatsBomb matches/events/lineups data into ManagerAgent instances.

Pipeline:
  1. extract_manager_tenure()        — match IDs for this manager's tenure
  2. calc_manager_pressing_intensity()
     calc_manager_possession_preference()
     calc_preferred_formation()      — tactical attribute derivation
  3. calc_cultural_inertia()         — tenure length → TeamBaseline.cultural_inertia
  4. build_manager_agent()           — assemble ManagerAgent

Manager tenure is derived from matches.home_managers / away_managers (string
fields), not from lineups. This was confirmed against StatsBomb Open Data.

M1 limitations:
  - implementation_speed: provisional 50.0 (tactical stabilisation requires
    time-series analysis, out of M1 scope)
  - youth_development: provisional 50.0 (age data unavailable in Open Data)
  - style_stubbornness: provisional 50.0 (LLM-derived, filled by
    llm-knowledge-query task)
"""

from __future__ import annotations

import pandas as pd

from iffootball.agents.manager import ManagerAgent
from iffootball.converters.stats_to_attributes import _POSITION_TO_ROLE, _ROLE_TO_BROAD
from iffootball.agents.player import BroadPosition

# One full league season (Premier League / La Liga both use 38 match rounds).
# Used to scale cultural_inertia: a manager who led every match gets 1.0.
_FULL_SEASON_MATCHES = 38

# Minimum starting XI size to attempt formation derivation.
_MIN_STARTING_XI = 11


# ---------------------------------------------------------------------------
# Manager name parsing
# ---------------------------------------------------------------------------


def _parse_managers(raw: object) -> list[str]:
    """Parse a raw manager field value into a list of manager name strings.

    Handles:
      - NaN / None              → returns []
      - empty or whitespace-only string → returns []
      - comma-separated entry   → splits and strips each name, skips blanks
        (e.g., "Tiago Cardoso Mendes, Diego Pablo Simeone")
      - single name             → returns a one-element list

    Name-level normalisation (e.g., abbreviations, accent variants) is
    intentionally out of M1 scope; within a single season StatsBomb records
    are consistent enough for tenure extraction.
    """
    if pd.isna(raw):  # type: ignore[call-overload]
        return []
    names = [n.strip() for n in str(raw).split(",")]
    return [n for n in names if n]


# ---------------------------------------------------------------------------
# Tenure extraction
# ---------------------------------------------------------------------------


def extract_manager_tenure(
    matches: pd.DataFrame,
    team_name: str,
    manager_name: str,
) -> frozenset[int]:
    """Return match IDs where team_name was managed by manager_name.

    Scans home_managers and away_managers fields for the team's matches.
    A match is included in the tenure if manager_name appears (after
    comma-splitting and stripping) in the field for team_name's side.

    Transition matches (comma-separated manager entries) are included in
    both managers' tenure sets. This reflects that both managers are
    associated with the match record.

    Args:
        matches:      Matches DataFrame for a single competition/season.
                      Must include match_id, home_team, away_team,
                      home_managers, away_managers.
        team_name:    StatsBomb team name (exact spelling).
        manager_name: Manager name as it appears in StatsBomb matches
                      (after splitting comma-separated entries).

    Returns:
        frozenset of match_id integers for this manager's tenure.
    """
    match_ids: set[int] = set()

    home_rows = matches[matches["home_team"] == team_name]
    for _, row in home_rows.iterrows():
        if manager_name in _parse_managers(row["home_managers"]):
            match_ids.add(int(row["match_id"]))

    away_rows = matches[matches["away_team"] == team_name]
    for _, row in away_rows.iterrows():
        if manager_name in _parse_managers(row["away_managers"]):
            match_ids.add(int(row["match_id"]))

    return frozenset(match_ids)


# ---------------------------------------------------------------------------
# Pressing intensity
# ---------------------------------------------------------------------------


def calc_manager_pressing_intensity(
    events: pd.DataFrame,
    team_name: str,
    tenure_match_ids: frozenset[int],
) -> float:
    """Return average Pressure events per 90 min for team_name over tenure.

    Computed as total Pressure count / number of matches (each treated as
    90 min).  Returns 0.0 if tenure_match_ids is empty.

    Args:
        events:           Combined events DataFrame for tenure matches.
        team_name:        StatsBomb team name.
        tenure_match_ids: Match IDs to include (from extract_manager_tenure).
    """
    n = len(tenure_match_ids)
    if n == 0:
        return 0.0

    df = events[
        events["match_id"].isin(tenure_match_ids) & (events["team"] == team_name)
    ]
    pressures = int((df["type"] == "Pressure").sum())
    return pressures / n


# ---------------------------------------------------------------------------
# Possession preference
# ---------------------------------------------------------------------------


def calc_manager_possession_preference(
    events: pd.DataFrame,
    team_name: str,
    tenure_match_ids: frozenset[int],
) -> float:
    """Return team possession preference as a ratio (0.0–1.0) over tenure.

    Possession is approximated as:
      team (Pass + Carry) events / all (Pass + Carry) events

    Returns 0.0 if no Pass or Carry events are found.

    Args:
        events:           Combined events DataFrame for tenure matches.
        team_name:        StatsBomb team name.
        tenure_match_ids: Match IDs to include.
    """
    if not tenure_match_ids:
        return 0.0

    df = events[
        events["match_id"].isin(tenure_match_ids)
        & events["type"].isin(["Pass", "Carry"])
    ].dropna(subset=["team"])

    total = len(df)
    if total == 0:
        return 0.0

    own = int((df["team"] == team_name).sum())
    return own / total


# ---------------------------------------------------------------------------
# Preferred formation
# ---------------------------------------------------------------------------


def _infer_formation(lineup_df: pd.DataFrame) -> str | None:
    """Infer formation string from a single match's lineup DataFrame.

    Formation is derived from the starting XI (players who have any position
    entry with start_reason == "Starting XI"). BroadPosition counts are used
    to build a "{DF}-{MF}-{FW}" string (GK is always 1 and excluded from
    the output).

    Returns None if:
      - Fewer than _MIN_STARTING_XI (11) starting players are found.
      - Any starting player has a position name not in the supported mapping.
      - The positions list is empty or malformed for a starting player.

    Args:
        lineup_df: Lineup DataFrame for one team in one match.
                   Must have a 'positions' column with lists of position
                   dicts (each dict has 'position' and 'start_reason').
    """
    broad_counts: dict[BroadPosition, int] = {
        BroadPosition.GK: 0,
        BroadPosition.DF: 0,
        BroadPosition.MF: 0,
        BroadPosition.FW: 0,
    }
    starters_found = 0

    for _, row in lineup_df.iterrows():
        positions = row.get("positions")
        if not isinstance(positions, list) or not positions:
            continue
        # Find any position entry with start_reason == "Starting XI".
        starting_entry = next(
            (p for p in positions if p.get("start_reason") == "Starting XI"),
            None,
        )
        if starting_entry is None:
            continue

        position_name = starting_entry.get("position")
        if not isinstance(position_name, str):
            return None
        if position_name not in _POSITION_TO_ROLE:
            return None

        role = _POSITION_TO_ROLE[position_name]
        broad = _ROLE_TO_BROAD[role]
        broad_counts[broad] += 1
        starters_found += 1

    if starters_found < _MIN_STARTING_XI:
        return None

    df_count = broad_counts[BroadPosition.DF]
    mf_count = broad_counts[BroadPosition.MF]
    fw_count = broad_counts[BroadPosition.FW]
    return f"{df_count}-{mf_count}-{fw_count}"


def calc_preferred_formation(
    lineups_by_match: dict[int, dict[str, pd.DataFrame]],
    team_name: str,
    tenure_match_ids: frozenset[int],
) -> str | None:
    """Return the most frequent formation for team_name over tenure matches.

    Formation is derived per match via _infer_formation(); matches where
    formation cannot be determined are skipped (not counted). Returns None
    if no valid formation is found across all tenure matches.

    Tie-breaking: when multiple formations share the highest frequency,
    the lexicographically smallest string is returned (e.g., "3-5-2" before
    "4-3-3") to ensure deterministic results.

    Args:
        lineups_by_match: Dict mapping match_id → (team_name → lineup_df).
        team_name:        StatsBomb team name.
        tenure_match_ids: Match IDs to consider.
    """
    formation_counts: dict[str, int] = {}

    for match_id in tenure_match_ids:
        match_lineups = lineups_by_match.get(match_id)
        if match_lineups is None:
            continue
        lineup_df = match_lineups.get(team_name)
        if lineup_df is None:
            continue
        formation = _infer_formation(lineup_df)
        if formation is not None:
            formation_counts[formation] = formation_counts.get(formation, 0) + 1

    if not formation_counts:
        return None

    return sorted(formation_counts, key=lambda f: (-formation_counts[f], f))[0]


# ---------------------------------------------------------------------------
# Cultural inertia
# ---------------------------------------------------------------------------


def calc_cultural_inertia(tenure_match_count: int) -> float:
    """Return cultural_inertia for TeamBaseline from manager tenure length.

    Maps tenure (number of matches managed) to a 0.0–1.0 scale using a
    linear scale anchored to a full season (_FULL_SEASON_MATCHES = 38).

    A manager who led every match in the season returns 1.0.
    Tenure beyond one full season is capped at 1.0.

    This value should be used to update TeamBaseline.cultural_inertia via
    dataclasses.replace(baseline, cultural_inertia=calc_cultural_inertia(n)).

    Args:
        tenure_match_count: Number of matches in the manager's tenure
                            within the target competition/season.
    """
    return min(tenure_match_count / _FULL_SEASON_MATCHES, 1.0)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_manager_agent(
    events: pd.DataFrame,
    matches: pd.DataFrame,
    lineups_by_match: dict[int, dict[str, pd.DataFrame]],
    team_name: str,
    manager_name: str,
    competition_id: int,
    season_id: int,
) -> ManagerAgent:
    """Build a ManagerAgent for manager_name at team_name.

    Args:
        events:           Combined match events DataFrame. Should cover all
                          matches in the manager's tenure.
        matches:          Matches DataFrame for the full competition/season.
                          Used for tenure extraction via manager fields.
        lineups_by_match: Dict mapping match_id → (team_name → lineup_df).
                          Used for formation derivation.
        team_name:        StatsBomb team name (exact spelling).
        manager_name:     Manager name as it appears in StatsBomb matches
                          (single name after comma-splitting).
        competition_id:   StatsBomb competition ID.
        season_id:        StatsBomb season ID.

    Returns:
        ManagerAgent with M1 provisional values for implementation_speed,
        youth_development, style_stubbornness, and squad_trust (empty dict).
    """
    tenure_match_ids = extract_manager_tenure(matches, team_name, manager_name)

    pressing = calc_manager_pressing_intensity(events, team_name, tenure_match_ids)
    possession = calc_manager_possession_preference(events, team_name, tenure_match_ids)
    counter = 1.0 - possession
    formation = calc_preferred_formation(lineups_by_match, team_name, tenure_match_ids)

    return ManagerAgent(
        manager_name=manager_name,
        team_name=team_name,
        competition_id=competition_id,
        season_id=season_id,
        tenure_match_ids=tenure_match_ids,
        pressing_intensity=pressing,
        possession_preference=possession,
        counter_tendency=counter,
        preferred_formation=formation,
    )
