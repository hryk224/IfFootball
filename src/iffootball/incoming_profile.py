"""Incoming manager profile resolution.

Resolves an incoming manager's tactical profile from cache or StatsBomb
data. Used by the app to differentiate simulation results per manager.

Resolution order:
    1. Demo cache DBs (fast, offline)
    2. StatsBomb runtime build (slow, requires API)
    3. Neutral fallback (always available)
"""

from __future__ import annotations

import logging
from pathlib import Path

from iffootball.agents.manager import ManagerAgent
from iffootball.storage.db import Database

_logger = logging.getLogger(__name__)


def resolve_incoming_profile(
    name: str,
    competition_id: int,
    season_id: int,
    cache_dir: Path | None = None,
) -> ManagerAgent:
    """Resolve an incoming manager's tactical profile.

    Attempts to find the manager's real profile from demo cache DBs,
    then from StatsBomb data at runtime. Falls back to neutral defaults
    if not found.

    Args:
        name:           Incoming manager name (exact StatsBomb spelling).
        competition_id: StatsBomb competition ID.
        season_id:      StatsBomb season ID.
        cache_dir:      Directory containing demo cache .db files.
                        None disables cache lookup.

    Returns:
        ManagerAgent with tactical attributes. May be from cache,
        runtime build, or neutral fallback.
    """
    # 1. Cache lookup.
    if cache_dir is not None:
        cached = _resolve_from_cache(name, competition_id, season_id, cache_dir)
        if cached is not None:
            return cached

    # 2. StatsBomb runtime build.
    runtime = _build_from_statsbomb(name, competition_id, season_id)
    if runtime is not None:
        return runtime

    # 3. Neutral fallback.
    _logger.info(
        "Incoming manager profile not found for %s; using neutral defaults.",
        name,
    )
    return neutral_manager_profile(name)


def _resolve_from_cache(
    name: str,
    competition_id: int,
    season_id: int,
    cache_dir: Path,
) -> ManagerAgent | None:
    """Search demo cache DBs for the incoming manager."""
    if not cache_dir.exists():
        return None

    for db_path in cache_dir.glob("*.db"):
        db: Database | None = None
        try:
            db = Database(db_path)
            rows = db._conn.execute(
                "SELECT DISTINCT manager_name, team_name "
                "FROM manager_agents "
                "WHERE manager_name = ? AND competition_id = ? AND season_id = ?",
                (name, competition_id, season_id),
            ).fetchall()
            for row in rows:
                agent = db.load_manager_agent(
                    row[0], row[1], competition_id, season_id,
                )
                if agent is not None:
                    return agent
        except Exception as exc:
            _logger.debug(
                "Cache lookup failed for %s in %s: %s", name, db_path, exc,
            )
        finally:
            if db is not None:
                db.close()

    return None


def _build_from_statsbomb(
    name: str,
    competition_id: int,
    season_id: int,
) -> ManagerAgent | None:
    """Try to build ManagerAgent from StatsBomb data at runtime."""
    try:
        from iffootball.candidates import CandidateResolver
        from iffootball.collectors.statsbomb import StatsBombOpenDataCollector
        from iffootball.converters.manager_stats import build_manager_agent

        collector = StatsBombOpenDataCollector()
        resolver = CandidateResolver(collector)
        candidates = resolver.managers(competition_id, season_id)
        match = next(
            (c for c in candidates if c.manager_name == name), None,
        )
        if match is None:
            return None

        import pandas as pd

        matches = collector.get_matches(competition_id, season_id)
        team_matches = matches[
            (matches["home_team"] == match.team_name)
            | (matches["away_team"] == match.team_name)
        ]
        events_by_match = {}
        lineups_by_match = {}
        for match_id in team_matches["match_id"]:
            events_by_match[match_id] = collector.get_events(match_id)
            lineups_by_match[match_id] = collector.get_lineups(match_id)

        all_events = [events_by_match[mid] for mid in events_by_match]
        if not all_events:
            return None

        merged_events = pd.concat(all_events, ignore_index=True)
        return build_manager_agent(
            merged_events,
            team_matches,
            lineups_by_match,
            match.team_name,
            name,
            competition_id,
            season_id,
        )
    except Exception as exc:
        _logger.warning(
            "Failed to build incoming profile for %s from StatsBomb: %s",
            name,
            exc,
        )
        return None


def neutral_manager_profile(name: str) -> ManagerAgent:
    """Build a neutral ManagerAgent stub when no data is available."""
    return ManagerAgent(
        manager_name=name,
        team_name="",
        competition_id=0,
        season_id=0,
        tenure_match_ids=frozenset(),
        pressing_intensity=50.0,
        possession_preference=0.5,
        counter_tendency=0.5,
        preferred_formation="4-4-2",
    )
