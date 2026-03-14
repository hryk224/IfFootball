"""Initialization pipeline connecting M1 components.

Orchestrates: StatsBomb collection -> converters -> LLM queries -> storage.

Data is split into three scopes to prevent future leakage:
  - target-team pre-trigger: PlayerAgent, TeamBaseline, ManagerAgent
  - league-wide pre-trigger: OpponentStrength, LeagueContext
  - full-season matches: FixtureList (metadata only, no events)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import pandas as pd

from iffootball.agents.fixture import FixtureList, OpponentStrength
from iffootball.agents.league import LeagueContext
from iffootball.agents.manager import ManagerAgent
from iffootball.agents.player import PlayerAgent
from iffootball.agents.team import TeamBaseline
from iffootball.collectors.statsbomb import StatsBombDataSource
from iffootball.converters.fixture_stats import (
    build_all_opponent_strengths,
    build_fixture_list,
)
from iffootball.converters.manager_stats import (
    build_manager_agent,
    calc_cultural_inertia,
)
from iffootball.converters.stats_to_attributes import build_player_agents
from iffootball.converters.team_stats import (
    build_team_baseline,
    calc_ppda,
    calc_progressive_passes_per90,
    calc_team_xg,
)
from iffootball.llm.client import LLMClient
from iffootball.llm.knowledge_query import (
    query_league_characteristics,
    query_manager_style,
)
from iffootball.storage.db import Database


@dataclass(frozen=True)
class InitializationResult:
    """Result of the initialization pipeline."""

    player_agents: list[PlayerAgent]
    team_baseline: TeamBaseline
    manager_agent: ManagerAgent
    fixture_list: FixtureList
    opponent_strengths: dict[str, OpponentStrength]
    league_context: LeagueContext


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Minimum columns expected by downstream converter functions.
# When no events are collected, an empty DataFrame with these columns is
# returned so that column-access (e.g. events["match_id"]) does not raise
# KeyError on an empty, column-less DataFrame.
_EMPTY_EVENTS_COLUMNS: list[str] = [
    "match_id",
    "team",
    "player_id",
    "type",
    "minute",
    "position",
    "pass_outcome",
    "location",
    "pass_end_location",
    "shot_statsbomb_xg",
]


def _collect_events(
    collector: StatsBombDataSource,
    match_ids: set[int],
) -> pd.DataFrame:
    """Collect and concatenate events for a set of match IDs."""
    if not match_ids:
        return pd.DataFrame(columns=_EMPTY_EVENTS_COLUMNS)
    frames = [collector.get_events(mid) for mid in sorted(match_ids)]
    return pd.concat(frames, ignore_index=True)


def _collect_lineups(
    collector: StatsBombDataSource,
    match_ids: set[int],
) -> dict[int, dict[str, pd.DataFrame]]:
    """Collect lineups for a set of match IDs.

    Returns:
        Dict mapping match_id -> (team_name -> lineup DataFrame).
    """
    return {mid: collector.get_lineups(mid) for mid in sorted(match_ids)}


def _merge_lineups(
    lineups_by_match: dict[int, dict[str, pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    """Merge per-match lineups into team_name -> combined DataFrame.

    Used as input to build_player_agents which needs a flat team->lineup
    mapping for player name resolution.
    """
    by_team: dict[str, list[pd.DataFrame]] = {}
    for match_lineups in lineups_by_match.values():
        for team_name, df in match_lineups.items():
            by_team.setdefault(team_name, []).append(df)
    return {
        team: pd.concat(dfs, ignore_index=True)
        for team, dfs in by_team.items()
    }


def _filter_team_matches(
    matches: pd.DataFrame,
    team_name: str,
) -> pd.DataFrame:
    """Filter matches to those involving team_name."""
    mask = (matches["home_team"] == team_name) | (
        matches["away_team"] == team_name
    )
    return matches[mask].copy()


def build_league_context(
    events: pd.DataFrame,
    matches: pd.DataFrame,
    competition_id: int,
    season_id: int,
    league_name: str,
) -> LeagueContext:
    """Build LeagueContext with fact attributes from league-wide pre-trigger data.

    Computes league averages by calculating per-team metrics and averaging
    across all teams that have played at least one match.

    Args:
        events:         League-wide events up to trigger_week.
        matches:        League-wide matches up to trigger_week.
        competition_id: StatsBomb competition ID.
        season_id:      StatsBomb season ID.
        league_name:    Human-readable league name.
    """
    all_teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))

    if not all_teams or events.empty:
        return LeagueContext(
            competition_id=competition_id,
            season_id=season_id,
            name=league_name,
        )

    played_matches = matches.dropna(subset=["home_score", "away_score"])

    ppda_values: list[float] = []
    prog_pass_values: list[float] = []
    xg_values: list[float] = []

    for team in all_teams:
        team_mask = (played_matches["home_team"] == team) | (
            played_matches["away_team"] == team
        )
        team_rows = played_matches[team_mask]
        if team_rows.empty:
            continue

        team_mids = frozenset(int(mid) for mid in team_rows["match_id"])

        xg_for, _ = calc_team_xg(events, team, team_mids)
        xg_values.append(xg_for)

        ppda = calc_ppda(events, team, team_mids)
        if not pd.isna(ppda):
            ppda_values.append(ppda)

        prog = calc_progressive_passes_per90(events, team, team_mids)
        prog_pass_values.append(prog)

    avg_ppda = sum(ppda_values) / len(ppda_values) if ppda_values else 0.0
    avg_prog = (
        sum(prog_pass_values) / len(prog_pass_values)
        if prog_pass_values
        else 0.0
    )
    avg_xg = sum(xg_values) / len(xg_values) if xg_values else 0.0

    return LeagueContext(
        competition_id=competition_id,
        season_id=season_id,
        name=league_name,
        avg_ppda=avg_ppda,
        avg_progressive_passes_per90=avg_prog,
        avg_xg_per90=avg_xg,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def initialize(
    collector: StatsBombDataSource,
    competition_id: int,
    season_id: int,
    team_name: str,
    manager_name: str,
    trigger_week: int,
    league_name: str,
    llm_client: LLMClient | None = None,
    db: Database | None = None,
) -> InitializationResult:
    """Run the full initialization pipeline.

    Connects M1 components into a single execution unit:
    StatsBomb data -> agent construction -> optional LLM enrichment
    -> optional DB persistence.

    Data is scoped into three systems to prevent future data leakage:
      - Full-season matches (metadata only): FixtureList
      - League-wide pre-trigger (match_week <= trigger_week):
        OpponentStrength, LeagueContext
      - Target-team pre-trigger (subset of league-wide):
        PlayerAgent, TeamBaseline, ManagerAgent

    Args:
        collector:      StatsBomb data source implementation.
        competition_id: StatsBomb competition ID.
        season_id:      StatsBomb season ID.
        team_name:      Target team (StatsBomb exact spelling).
        manager_name:   Manager name (StatsBomb exact spelling).
        trigger_week:   Trigger injection point. Agents are built from
                        data up to and including this week.
        league_name:    Human-readable league name for LeagueContext.
        llm_client:     Optional LLM client for hypothesis enrichment.
        db:             Optional database for persistence.

    Returns:
        InitializationResult with all constructed agents and context.
    """
    # -- 1. Full-season matches (metadata only) --
    all_matches = collector.get_matches(competition_id, season_id)

    fixture_list = build_fixture_list(all_matches, team_name, trigger_week)

    # -- 2. League-wide pre-trigger scope --
    league_pre_matches = all_matches[
        all_matches["match_week"] <= trigger_week
    ].copy()
    league_pre_match_ids = set(
        int(mid) for mid in league_pre_matches["match_id"]
    )
    league_pre_events = _collect_events(collector, league_pre_match_ids)

    # -- 3. Target-team pre-trigger scope (subset of league-wide) --
    team_pre_matches = _filter_team_matches(league_pre_matches, team_name)
    team_pre_match_ids = set(
        int(mid) for mid in team_pre_matches["match_id"]
    )
    team_pre_events = league_pre_events[
        league_pre_events["match_id"].isin(team_pre_match_ids)
    ].copy()

    lineups_by_match = _collect_lineups(collector, team_pre_match_ids)
    merged_lineups = _merge_lineups(lineups_by_match)

    # -- 4. Build target-team agents --
    player_agents = build_player_agents(team_pre_events, merged_lineups)

    team_baseline = build_team_baseline(
        team_pre_events,
        all_matches,  # full season for correct standings + matches_remaining
        team_name,
        frozenset(team_pre_match_ids),
        competition_id,
        season_id,
    )

    manager_agent = build_manager_agent(
        team_pre_events,
        team_pre_matches,  # team matches only for tenure extraction
        lineups_by_match,
        team_name,
        manager_name,
        competition_id,
        season_id,
    )

    # -- 5. Update cultural_inertia from manager tenure --
    tenure_len = len(manager_agent.tenure_match_ids)
    team_baseline = dataclasses.replace(
        team_baseline,
        cultural_inertia=calc_cultural_inertia(tenure_len),
    )

    # -- 6. Build league-wide objects --
    opponent_strengths = build_all_opponent_strengths(
        league_pre_events, all_matches, fixture_list,
    )

    league_context = build_league_context(
        league_pre_events,
        league_pre_matches,
        competition_id,
        season_id,
        league_name,
    )

    # -- 7. LLM enrichment (optional) --
    if llm_client is not None:
        style = query_manager_style(
            llm_client,
            manager_name,
            formation_options=[],
        )
        manager_agent = dataclasses.replace(
            manager_agent,
            style_stubbornness=style.style_stubbornness,
        )

        chars = query_league_characteristics(llm_client, league_name)
        league_context = dataclasses.replace(
            league_context,
            pressing_level=chars.pressing_level,
            physicality_level=chars.physicality_level,
            tactical_complexity=chars.tactical_complexity,
        )

    # -- 8. DB persistence (optional; LeagueContext not persisted) --
    if db is not None:
        db.save_player_agents(player_agents, competition_id, season_id)
        db.save_team_baseline(team_baseline)
        db.save_manager_agent(manager_agent)
        db.save_fixture_list(fixture_list, competition_id, season_id)
        db.save_opponent_strengths(
            opponent_strengths, competition_id, season_id, trigger_week,
        )

    return InitializationResult(
        player_agents=player_agents,
        team_baseline=team_baseline,
        manager_agent=manager_agent,
        fixture_list=fixture_list,
        opponent_strengths=opponent_strengths,
        league_context=league_context,
    )
