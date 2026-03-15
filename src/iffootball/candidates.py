"""Candidate resolution for scenario building.

Resolves team, manager, and incoming-manager candidates from StatsBomb
data. The resolver collects candidates across all available competitions
but provides filtering for UI display (e.g., same-league-only).

Usage:
    resolver = CandidateResolver(collector)
    teams = resolver.teams(competition_id=2, season_id=27)
    managers = resolver.managers(competition_id=2, season_id=27, team_name="Chelsea")
    incoming = resolver.incoming_candidates(
        competition_id=2, season_id=27, exclude_team="Chelsea"
    )
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from iffootball.collectors.statsbomb import StatsBombDataSource


@dataclass(frozen=True)
class ManagerCandidate:
    """A manager available as an incoming candidate.

    Attributes:
        manager_name: StatsBomb canonical name.
        team_name:    Team the manager is associated with.
    """

    manager_name: str
    team_name: str


class CandidateResolver:
    """Resolves candidates from StatsBomb data.

    Caches match data per (competition_id, season_id) to avoid
    repeated API calls.
    """

    def __init__(self, collector: StatsBombDataSource) -> None:
        self._collector = collector
        self._matches_cache: dict[tuple[int, int], pd.DataFrame] = {}

    def _get_matches(self, competition_id: int, season_id: int) -> pd.DataFrame:
        """Get matches with caching."""
        key = (competition_id, season_id)
        if key not in self._matches_cache:
            self._matches_cache[key] = self._collector.get_matches(
                competition_id, season_id
            )
        return self._matches_cache[key]

    def teams(self, competition_id: int, season_id: int) -> list[str]:
        """Return all team names in the competition/season."""
        matches = self._get_matches(competition_id, season_id)
        home = set(matches["home_team"].unique())
        away = set(matches["away_team"].unique())
        return sorted(home | away)

    def managers(
        self,
        competition_id: int,
        season_id: int,
        team_name: str | None = None,
        at_week: int | None = None,
    ) -> list[ManagerCandidate]:
        """Return managers for a team (or all teams if team_name is None).

        Args:
            competition_id: StatsBomb competition ID.
            season_id:      StatsBomb season ID.
            team_name:      Filter to a specific team. None = all teams.
            at_week:        Filter to managers active at this match week.
                            None = all managers across the full season.
        """
        matches = self._get_matches(competition_id, season_id)
        result: dict[tuple[str, str], ManagerCandidate] = {}

        for _, row in matches.iterrows():
            # Filter by match_week if at_week is specified.
            if at_week is not None and "match_week" in row.index:
                if int(row["match_week"]) > at_week:
                    continue

            for side, mgr_col in [
                ("home_team", "home_managers"),
                ("away_team", "away_managers"),
            ]:
                t = str(row[side])
                if team_name is not None and t != team_name:
                    continue
                for m in str(row.get(mgr_col, "")).split(", "):
                    m = m.strip()
                    if m:
                        key = (m, t)
                        if key not in result:
                            result[key] = ManagerCandidate(
                                manager_name=m, team_name=t
                            )

        return sorted(result.values(), key=lambda c: (c.team_name, c.manager_name))

    def manager_at_week(
        self,
        competition_id: int,
        season_id: int,
        team_name: str,
        week: int,
    ) -> ManagerCandidate | None:
        """Return the manager most recently active at or before the given week.

        Uses the last match at or before `week` to determine the current
        manager. Returns None if no matches found.
        """
        matches = self._get_matches(competition_id, season_id)
        team_matches = matches[
            ((matches["home_team"] == team_name) | (matches["away_team"] == team_name))
        ]
        if "match_week" in team_matches.columns:
            team_matches = team_matches[team_matches["match_week"] <= week]

        if team_matches.empty:
            return None

        # Get the latest match.
        latest = team_matches.iloc[-1]
        if latest["home_team"] == team_name:
            mgr_col = "home_managers"
        else:
            mgr_col = "away_managers"

        mgr_str = str(latest.get(mgr_col, ""))
        names = [m.strip() for m in mgr_str.split(", ") if m.strip()]
        if not names:
            return None

        return ManagerCandidate(manager_name=names[0], team_name=team_name)

    def incoming_candidates(
        self,
        competition_id: int,
        season_id: int,
        exclude_team: str | None = None,
    ) -> list[ManagerCandidate]:
        """Return managers available as incoming candidates from one league.

        Set exclude_team to filter out the current team's managers.
        """
        all_managers = self.managers(competition_id, season_id)
        if exclude_team is not None:
            all_managers = [
                m for m in all_managers if m.team_name != exclude_team
            ]
        return all_managers

    def incoming_candidates_cross_league(
        self,
        targets: list[tuple[int, int]],
        exclude_team: str | None = None,
    ) -> list[ManagerCandidate]:
        """Return incoming candidates from multiple competitions.

        Args:
            targets:      List of (competition_id, season_id) pairs.
            exclude_team: Team name to exclude (applies across all leagues).

        Returns:
            Merged and deduplicated list of ManagerCandidate, sorted by
            team_name then manager_name.
        """
        seen: dict[tuple[str, str], ManagerCandidate] = {}
        for comp_id, szn_id in targets:
            for c in self.incoming_candidates(comp_id, szn_id, exclude_team):
                key = (c.manager_name, c.team_name)
                if key not in seen:
                    seen[key] = c
        return sorted(seen.values(), key=lambda c: (c.team_name, c.manager_name))
