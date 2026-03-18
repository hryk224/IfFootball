"""Convert StatsBomb event data into PlayerAgent instances.

Pipeline:
  1. calc_minutes_played()      — minutes per player from match events
  2. aggregate_player_stats()   — per-90 metrics + xG variance per player
  3. normalize_percentile()     — rank within >=900-min cohort (0–100 scale)
                                  + consistency from position-group percentiles
  4. build_player_agents()      — assemble PlayerAgent list

Consistency derivation (M3):
  Position-group specific: each group uses a primary metric and a common
  pass-count std as secondary. Inverted percentile within the group so
  low variance = high consistency. Composite weight: 0.5 primary + 0.5 pass.
  Players with fewer than _MIN_MATCHES_FOR_CONSISTENCY matches for a
  metric fall back to the other metric alone, then to 50.0 if both fail.

  Position groups and primary metrics:
    GK: pass_std only (no xG/defensive relevance)
    DF: def_std (Tackle + Interception per match)
    MF: pass_std (involvement consistency)
    FW: xg_std (attacking output consistency)

  Composite weight 0.5/0.5 is provisional (M3). Subject to tuning.

Position mapping:
  StatsBomb position_name -> RoleFamily -> BroadPosition
  Mapping is defined as dict constants (_POSITION_TO_ROLE, _ROLE_TO_BROAD) so
  changes require editing exactly one place.
"""

from __future__ import annotations

import pandas as pd

from iffootball.agents.player import BroadPosition, PlayerAgent, RoleFamily, SampleTier

# ---------------------------------------------------------------------------
# Position mapping constants
# ---------------------------------------------------------------------------

# Maps every supported StatsBomb position_name to a RoleFamily.
# Unknown names raise ValueError in to_role_family() — intentional for early
# failure during M1 development.
_POSITION_TO_ROLE: dict[str, RoleFamily] = {
    "Goalkeeper": RoleFamily.GOALKEEPER,
    # --- Defenders ---
    "Right Back": RoleFamily.FULL_BACK,
    "Left Back": RoleFamily.FULL_BACK,
    "Right Center Back": RoleFamily.CENTER_BACK,
    "Left Center Back": RoleFamily.CENTER_BACK,
    "Center Back": RoleFamily.CENTER_BACK,
    # Wing backs have distinct wide/forward responsibilities; kept separate from full backs.
    "Right Wing Back": RoleFamily.WING_BACK,
    "Left Wing Back": RoleFamily.WING_BACK,
    # --- Midfielders ---
    "Right Defensive Midfield": RoleFamily.DEFENSIVE_MIDFIELDER,
    "Center Defensive Midfield": RoleFamily.DEFENSIVE_MIDFIELDER,
    "Left Defensive Midfield": RoleFamily.DEFENSIVE_MIDFIELDER,
    # M1: side midfielders (Right/Left Midfield) treated as central MF.
    # Revisit if winger-vs-box-to-box distinction becomes rule-relevant.
    "Right Midfield": RoleFamily.CENTRAL_MIDFIELDER,
    "Left Midfield": RoleFamily.CENTRAL_MIDFIELDER,
    "Right Center Midfield": RoleFamily.CENTRAL_MIDFIELDER,
    "Left Center Midfield": RoleFamily.CENTRAL_MIDFIELDER,
    "Center Attacking Midfield": RoleFamily.ATTACKING_MIDFIELDER,
    # Wingers are classified as MF in BroadPosition for M1 (side attackers lean MF).
    "Right Wing": RoleFamily.WINGER,
    "Left Wing": RoleFamily.WINGER,
    # --- Forwards ---
    "Right Center Forward": RoleFamily.FORWARD,
    "Left Center Forward": RoleFamily.FORWARD,
    "Center Forward": RoleFamily.FORWARD,
}

_ROLE_TO_BROAD: dict[RoleFamily, BroadPosition] = {
    RoleFamily.GOALKEEPER: BroadPosition.GK,
    RoleFamily.CENTER_BACK: BroadPosition.DF,
    RoleFamily.FULL_BACK: BroadPosition.DF,
    RoleFamily.WING_BACK: BroadPosition.DF,
    RoleFamily.DEFENSIVE_MIDFIELDER: BroadPosition.MF,
    RoleFamily.CENTRAL_MIDFIELDER: BroadPosition.MF,
    RoleFamily.ATTACKING_MIDFIELDER: BroadPosition.MF,
    RoleFamily.WINGER: BroadPosition.MF,
    RoleFamily.FORWARD: BroadPosition.FW,
}

# The set of position names this module explicitly supports.
# Tests should validate against this constant rather than hardcoding a count,
# so that newly added mappings are automatically covered.
SUPPORTED_POSITION_NAMES: frozenset[str] = frozenset(_POSITION_TO_ROLE)


# ---------------------------------------------------------------------------
# Position conversion
# ---------------------------------------------------------------------------


def to_role_family(position_name: str) -> RoleFamily:
    """Return the RoleFamily for a StatsBomb position name.

    Raises:
        ValueError: if position_name is not in the supported mapping.
            Fail-fast behaviour is intentional for M1 to surface unexpected data early.
    """
    try:
        return _POSITION_TO_ROLE[position_name]
    except KeyError:
        raise ValueError(
            f"Unsupported position_name: {position_name!r}. "
            f"Add it to _POSITION_TO_ROLE in stats_to_attributes.py."
        ) from None


def to_broad_position(role_family: RoleFamily) -> BroadPosition:
    """Return the BroadPosition for a RoleFamily."""
    return _ROLE_TO_BROAD[role_family]


# ---------------------------------------------------------------------------
# Stats aggregation
# ---------------------------------------------------------------------------

_MINUTES_PER_MATCH = 90.0
_MIN_MINUTES_THRESHOLD = 900.0
_PARTIAL_MINUTES_THRESHOLD = 270.0  # 3 full matches; minimum for position data

# Minimum matches with xG data to compute consistency from variance.
# Players below this threshold keep the neutral 50.0 placeholder.
_MIN_MATCHES_FOR_CONSISTENCY = 5


def calc_minutes_played(events: pd.DataFrame) -> pd.Series:
    """Calculate total minutes played per player from match events.

    Uses 'player_id' and 'minute' columns. Minutes per match are capped at 90
    before summing, so stoppage time (minute > 90) does not inflate totals.

    Returns:
        Series indexed by player_id with total minutes played.
    """
    played = (
        events.dropna(subset=["player_id", "minute"])
        .groupby(["player_id", "match_id"])["minute"]
        .max()
        .clip(upper=90)
        .reset_index()
        .groupby("player_id")["minute"]
        .sum()
    )
    return played


def aggregate_player_stats(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-90 metrics per player from match events.

    Metrics computed:
      - xg_per90:        sum of shot.statsbomb_xg / (minutes / 90)
      - pressures_per90: count of Pressure events / (minutes / 90)
      - tackles_per90:   count of Tackle events / (minutes / 90)
      - passes_per90:    count of Pass events / (minutes / 90)
      - xg_std:          std of per-match xG involvement (for consistency)
      - xg_match_count:  number of matches with xG data (for consistency filter)

    Only players with at least 1 minute recorded are included.

    Returns:
        DataFrame indexed by player_id with columns for each metric.
    """
    minutes = calc_minutes_played(events)
    minutes = minutes[minutes > 0]

    def _per90(values: pd.Series, mins: pd.Series) -> pd.Series:
        return values.reindex(mins.index, fill_value=0.0) / (mins / _MINUTES_PER_MATCH)

    shots = events[events["type"] == "Shot"].copy()
    xg_total = (
        shots.dropna(subset=["player_id"])
        .groupby("player_id")["shot_statsbomb_xg"]
        .sum()
    )

    # Per-match xG for consistency calculation.
    xg_per_match = (
        shots.dropna(subset=["player_id", "shot_statsbomb_xg"])
        .groupby(["player_id", "match_id"])["shot_statsbomb_xg"]
        .sum()
    )
    xg_std = xg_per_match.groupby("player_id").std(ddof=0).fillna(0.0)
    xg_match_count = xg_per_match.groupby("player_id").size()

    pressures = (
        events[events["type"] == "Pressure"]
        .dropna(subset=["player_id"])
        .groupby("player_id")
        .size()
        .astype(float)
    )

    tackles = (
        events[events["type"] == "Tackle"]
        .dropna(subset=["player_id"])
        .groupby("player_id")
        .size()
        .astype(float)
    )

    passes = (
        events[events["type"] == "Pass"]
        .dropna(subset=["player_id"])
        .groupby("player_id")
        .size()
        .astype(float)
    )

    # Per-player played-match index: all (player_id, match_id) pairs where
    # the player has any event. Used to fill zero-event matches when computing
    # per-match std for consistency.
    played_matches = (
        events.dropna(subset=["player_id"])
        .groupby(["player_id", "match_id"])
        .size()
    )
    played_match_index = played_matches.index

    # Per-match counts for consistency calculation (pass, defensive).
    # Reindex against played matches so zero-event matches count as 0.
    pass_per_match_raw = (
        events[events["type"] == "Pass"]
        .dropna(subset=["player_id"])
        .groupby(["player_id", "match_id"])
        .size()
        .astype(float)
    )
    pass_per_match = pass_per_match_raw.reindex(played_match_index, fill_value=0.0)
    pass_std = pass_per_match.groupby("player_id").std(ddof=0).fillna(0.0)
    pass_match_count = pass_per_match.groupby("player_id").size()

    defensive_events = events[events["type"].isin(("Tackle", "Interception"))].copy()
    def_per_match_raw = (
        defensive_events.dropna(subset=["player_id"])
        .groupby(["player_id", "match_id"])
        .size()
        .astype(float)
    )
    def_per_match = def_per_match_raw.reindex(played_match_index, fill_value=0.0)
    def_std = def_per_match.groupby("player_id").std(ddof=0).fillna(0.0)
    def_match_count = def_per_match.groupby("player_id").size()

    return pd.DataFrame(
        {
            "minutes": minutes,
            "xg_per90": _per90(xg_total, minutes),
            "pressures_per90": _per90(pressures, minutes),
            "tackles_per90": _per90(tackles, minutes),
            "passes_per90": _per90(passes, minutes),
            "xg_std": xg_std.reindex(minutes.index, fill_value=0.0),
            "xg_match_count": xg_match_count.reindex(minutes.index, fill_value=0),
            "pass_std": pass_std.reindex(minutes.index, fill_value=0.0),
            "pass_match_count": pass_match_count.reindex(minutes.index, fill_value=0),
            "def_std": def_std.reindex(minutes.index, fill_value=0.0),
            "def_match_count": def_match_count.reindex(minutes.index, fill_value=0),
        }
    )


# Helper columns excluded from percentile normalization.
_HELPER_COLS = frozenset({
    "minutes",
    "xg_std", "xg_match_count",
    "pass_std", "pass_match_count",
    "def_std", "def_match_count",
    "consistency",
})

# Primary consistency metric per BroadPosition.
# Each maps to (std_column, match_count_column).
_PRIMARY_METRIC: dict[BroadPosition, tuple[str, str]] = {
    BroadPosition.GK: ("pass_std", "pass_match_count"),
    BroadPosition.DF: ("def_std", "def_match_count"),
    BroadPosition.MF: ("pass_std", "pass_match_count"),
    BroadPosition.FW: ("xg_std", "xg_match_count"),
}

# Secondary metric (common for all positions).
_SECONDARY_METRIC = ("pass_std", "pass_match_count")

# Composite weight: primary vs secondary. Provisional (M3).
_PRIMARY_WEIGHT = 0.5
_SECONDARY_WEIGHT = 0.5


def normalize_percentile(
    stats: pd.DataFrame,
    min_minutes: float = _MIN_MINUTES_THRESHOLD,
    position_map: dict[int, BroadPosition] | None = None,
) -> pd.DataFrame:
    """Percentile-normalise technical metrics within the qualified cohort.

    Only players with minutes >= min_minutes contribute to the percentile
    distribution. Players below the threshold are excluded from the output.

    Percentile values are scaled to 0–100.

    Consistency is derived from position-group specific metrics:
      GK: pass_std, DF: def_std, MF: pass_std, FW: xg_std
    with pass_std as common secondary. Percentile ranking is within
    the position group. Low std = high consistency (inverted percentile).
    Composite: 0.5 * primary + 0.5 * secondary (provisional M3 weight).

    Fallback: if primary is unavailable, use secondary alone.
    If both unavailable, use 50.0 neutral.

    Args:
        stats:        DataFrame from aggregate_player_stats().
        min_minutes:  Minimum minutes for cohort inclusion (default 900).
        position_map: player_id -> BroadPosition mapping. When None,
                      falls back to xg_std-only consistency (M2 compat).

    Returns:
        DataFrame with percentile scores (0–100) plus a 'consistency'
        column, limited to qualified players.
    """
    qualified = stats[stats["minutes"] >= min_minutes].copy()

    # Standard percentile columns (exclude helper columns).
    metric_cols = [c for c in qualified.columns if c not in _HELPER_COLS]
    qualified[metric_cols] = qualified[metric_cols].rank(pct=True) * 100

    # Consistency calculation.
    if position_map is not None:
        qualified["consistency"] = _calc_group_consistency(qualified, position_map)
    else:
        # M2 fallback: xg_std only (no position info).
        qualified["consistency"] = _calc_xg_only_consistency(qualified)

    return qualified


def _calc_xg_only_consistency(qualified: pd.DataFrame) -> pd.Series:
    """M2-compatible consistency from xg_std only."""
    if "xg_std" in qualified.columns and "xg_match_count" in qualified.columns:
        std_pct = qualified["xg_std"].rank(pct=True) * 100
        consistency = 100.0 - std_pct
        has_enough = qualified["xg_match_count"] >= _MIN_MATCHES_FOR_CONSISTENCY
        return consistency.where(has_enough, 50.0)
    return pd.Series(50.0, index=qualified.index)


def _calc_group_consistency(
    qualified: pd.DataFrame,
    position_map: dict[int, BroadPosition],
) -> pd.Series:
    """Position-group consistency with primary + secondary composite."""
    result = pd.Series(50.0, index=qualified.index)

    # Assign broad_position to each player.
    positions = pd.Series(
        {pid: position_map.get(pid) for pid in qualified.index},
        dtype=object,
    )

    for bp in BroadPosition:
        group_mask = positions == bp
        group = qualified[group_mask]
        if group.empty:
            continue

        primary_col, primary_count_col = _PRIMARY_METRIC[bp]
        secondary_col, secondary_count_col = _SECONDARY_METRIC

        primary_scores = _inverted_group_percentile(
            group, primary_col, primary_count_col
        )
        secondary_scores = _inverted_group_percentile(
            group, secondary_col, secondary_count_col
        )

        # Composite with fallback.
        for pid in group.index:
            p_val = primary_scores.get(pid)
            s_val = secondary_scores.get(pid)

            if p_val is not None and s_val is not None:
                result[pid] = _PRIMARY_WEIGHT * p_val + _SECONDARY_WEIGHT * s_val
            elif p_val is not None:
                result[pid] = p_val
            elif s_val is not None:
                result[pid] = s_val
            # else: stays 50.0

    return result


def _inverted_group_percentile(
    group: pd.DataFrame,
    std_col: str,
    count_col: str,
) -> dict[int, float | None]:
    """Compute inverted percentile within a position group.

    Returns dict mapping player_id to score (0-100) or None if
    the player doesn't have enough matches for the metric.
    """
    scores: dict[int, float | None] = {}
    if std_col not in group.columns or count_col not in group.columns:
        return {pid: None for pid in group.index}

    has_enough = group[count_col] >= _MIN_MATCHES_FOR_CONSISTENCY
    eligible = group[has_enough]

    if eligible.empty:
        return {pid: None for pid in group.index}

    # Rank within the eligible subset.
    pct = eligible[std_col].rank(pct=True) * 100
    inverted = 100.0 - pct

    for pid in group.index:
        if pid in inverted.index:
            scores[pid] = float(inverted[pid])
        else:
            scores[pid] = None

    return scores


# ---------------------------------------------------------------------------
# Stat -> attribute mapping
# ---------------------------------------------------------------------------

def _map_technical_attributes(row: pd.Series) -> dict[str, float]:
    """Map normalised per-90 metrics to PlayerAgent technical attribute names.

    This mapping reflects M2 design:
      - pace:        not derivable from event data; set to neutral 50.0
      - passing:     passes_per90 percentile
      - shooting:    xg_per90 percentile
      - pressing:    pressures_per90 percentile
      - defending:   tackles_per90 percentile
      - physicality: not derivable from event data; set to neutral 50.0
      - consistency: derived from xG variance (100 - std_percentile).
                     Players with too few matches get 50.0 placeholder.
    """
    return {
        "pace": 50.0,
        "passing": float(row.get("passes_per90", 50.0)),
        "shooting": float(row.get("xg_per90", 50.0)),
        "pressing": float(row.get("pressures_per90", 50.0)),
        "defending": float(row.get("tackles_per90", 50.0)),
        "physicality": 50.0,
        "consistency": float(row.get("consistency", 50.0)),
    }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _derive_representative_positions(events: pd.DataFrame) -> dict[int, str]:
    """Return the most frequently occurring position per player from event data.

    Frequency is measured by event count (not time-weighted).
    Ties are broken by taking the lexicographically first value (mode().iloc[0]).

    Returns:
        Dict mapping player_id (int) to position_name (str).
        Players whose events have no position data are excluded.
    """
    pos_data = events[["player_id", "position"]].dropna(subset=["player_id", "position"])
    if pos_data.empty:
        return {}
    result: dict[int, str] = {}
    for player_id, group in pos_data.groupby("player_id"):
        modes = group["position"].mode()
        if not modes.empty:
            result[int(player_id)] = str(modes.iloc[0])  # type: ignore[arg-type]
    return result


def build_player_agents(
    events: pd.DataFrame,
    lineups: dict[str, pd.DataFrame],
) -> list[PlayerAgent]:
    """Build PlayerAgent instances for qualified and partial players.

    Two tiers of players are included:
      - FULL (>= 900 min): percentile-normalised attributes from stats.
      - PARTIAL (270-899 min): neutral fallback attributes (50.0 for all
        technical metrics). Present for scenario naturalness but with
        limited data confidence.

    Players below 270 minutes or without a lineup entry are excluded.
    This is a structural limitation of StatsBomb Open Data: players with
    no event-level appearance data cannot be included.

    Args:
        events: Combined match events DataFrame with columns including
                player_id, position, type, minute, match_id.
                Representative position is derived from the most frequent
                position value per player (event count, not time-weighted).
        lineups: Dict mapping team name -> lineup DataFrame with columns
                 player_id, player_name. Used for player names only;
                 players with no lineup entry are skipped.

    Returns:
        List of PlayerAgent instances (FULL + PARTIAL tier).

    Raises:
        ValueError: if a FULL tier player has no position data in events,
                    or if their representative position is not in
                    SUPPORTED_POSITION_NAMES.
    """
    raw_stats = aggregate_player_stats(events)
    representative_positions = _derive_representative_positions(events)
    minutes = calc_minutes_played(events)

    # Build position map for group-level consistency.
    position_map: dict[int, BroadPosition] = {}
    for pid, pos_name in representative_positions.items():
        try:
            role = to_role_family(pos_name)
            position_map[pid] = to_broad_position(role)
        except ValueError:
            pass  # Unknown position; will be caught later in agent construction.

    normalised = normalize_percentile(raw_stats, position_map=position_map)
    normalised_pids: set[int] = set(int(pid) for pid in normalised.index)

    # Build player_id -> player_name from lineups (identity source).
    player_names: dict[int, str] = {}
    for df in lineups.values():
        for _, row in df.iterrows():
            player_names[int(row["player_id"])] = str(row["player_name"])

    agents: list[PlayerAgent] = []

    # --- FULL tier: >= 900 min, percentile-normalised ---
    for player_id, stats_row in normalised.iterrows():
        pid = int(player_id)  # type: ignore[call-overload]
        if pid not in player_names:
            # Player qualifies by minutes but has no lineup entry; skip.
            continue
        position_name = representative_positions.get(pid)
        if position_name is None:
            raise ValueError(
                f"Player {pid} has no position data in events. "
                "Cannot derive representative position."
            )
        role = to_role_family(position_name)
        broad = to_broad_position(role)
        attrs = _map_technical_attributes(stats_row)
        agents.append(
            PlayerAgent(
                player_id=pid,
                player_name=player_names[pid],
                position_name=position_name,
                role_family=role,
                broad_position=broad,
                pace=attrs["pace"],
                passing=attrs["passing"],
                shooting=attrs["shooting"],
                pressing=attrs["pressing"],
                defending=attrs["defending"],
                physicality=attrs["physicality"],
                consistency=attrs["consistency"],
                sample_tier=SampleTier.FULL,
            )
        )

    # --- PARTIAL tier: 270-899 min, neutral fallback ---
    _NEUTRAL = 50.0
    for pid_raw, player_minutes in minutes.items():
        pid = int(pid_raw)  # type: ignore[call-overload]
        if pid in normalised_pids:
            continue  # Already included as FULL tier.
        if player_minutes < _PARTIAL_MINUTES_THRESHOLD:
            continue  # Below partial threshold.
        if pid not in player_names:
            continue  # No lineup entry.
        position_name = representative_positions.get(pid)
        if position_name is None:
            continue  # No position data; skip silently for partial tier.
        try:
            role = to_role_family(position_name)
            broad = to_broad_position(role)
        except ValueError:
            continue  # Unsupported position; skip silently for partial tier.
        agents.append(
            PlayerAgent(
                player_id=pid,
                player_name=player_names[pid],
                position_name=position_name,
                role_family=role,
                broad_position=broad,
                pace=_NEUTRAL,
                passing=_NEUTRAL,
                shooting=_NEUTRAL,
                pressing=_NEUTRAL,
                defending=_NEUTRAL,
                physicality=_NEUTRAL,
                consistency=_NEUTRAL,
                sample_tier=SampleTier.PARTIAL,
            )
        )

    return agents
