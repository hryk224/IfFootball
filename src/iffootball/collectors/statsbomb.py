"""StatsBomb data collection layer.

Provides an abstract interface (StatsBombDataSource) and a concrete
implementation for StatsBomb Open Data (StatsBombOpenDataCollector).

All statsbombpy imports are confined to StatsBombOpenDataCollector to allow
future replacement with StatsBomb IQ or other data sources without changing
calling code.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd


@dataclass
class CompetitionTarget:
    """A competition/season pair with a list of target clubs."""

    competition_id: int
    season_id: int
    clubs: list[str]


def load_targets(config_path: Path) -> list[CompetitionTarget]:
    """Load target competitions and clubs from a TOML config file.

    The config file must use array-of-tables format:

        [[competitions]]
        competition_id = 2
        season_id = 27
        clubs = ["Manchester United", ...]
    """
    with config_path.open("rb") as f:
        data = tomllib.load(f)
    return [
        CompetitionTarget(
            competition_id=entry["competition_id"],
            season_id=entry["season_id"],
            clubs=entry["clubs"],
        )
        for entry in data.get("competitions", [])
    ]


class StatsBombDataSource(Protocol):
    """Interface for StatsBomb data retrieval.

    Implementations must not expose raw statsbombpy return values to callers.
    All methods return normalized pandas DataFrames.
    """

    def get_competitions(self) -> pd.DataFrame:
        """Return available competitions as a DataFrame."""
        ...

    def get_matches(self, competition_id: int, season_id: int) -> pd.DataFrame:
        """Return all matches for the given competition and season."""
        ...

    def get_events(self, match_id: int) -> pd.DataFrame:
        """Return all events for the given match."""
        ...

    def get_lineups(self, match_id: int) -> dict[str, pd.DataFrame]:
        """Return lineups for the given match, keyed by team name.

        Returns a dict mapping team name -> DataFrame with player lineup data.
        The schema is normalized independently of the underlying data source.
        """
        ...


class StatsBombOpenDataCollector:
    """StatsBomb Open Data implementation of StatsBombDataSource.

    All statsbombpy usage is confined to this class.
    """

    def get_competitions(self) -> pd.DataFrame:
        from statsbombpy import sb as _sb  # type: ignore[import-untyped]

        return pd.DataFrame(_sb.competitions())

    def get_matches(self, competition_id: int, season_id: int) -> pd.DataFrame:
        from statsbombpy import sb as _sb

        return pd.DataFrame(
            _sb.matches(competition_id=competition_id, season_id=season_id)
        )

    def get_events(self, match_id: int) -> pd.DataFrame:
        from statsbombpy import sb as _sb

        return pd.DataFrame(_sb.events(match_id=match_id))

    def get_lineups(self, match_id: int) -> dict[str, pd.DataFrame]:
        from statsbombpy import sb as _sb

        raw: dict[str, pd.DataFrame] = _sb.lineups(match_id=match_id)
        return {team: pd.DataFrame(df) for team, df in raw.items()}


def get_target_matches(
    collector: StatsBombDataSource,
    targets: list[CompetitionTarget],
) -> pd.DataFrame:
    """Return all matches involving target clubs across all target competitions.

    A match is included if home_team or away_team is present in the target club list.
    """
    frames: list[pd.DataFrame] = []
    for target in targets:
        matches = collector.get_matches(target.competition_id, target.season_id)
        mask = (
            matches["home_team"].isin(target.clubs)
            | matches["away_team"].isin(target.clubs)
        )
        frames.append(matches[mask])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
