"""Tests for StatsBomb data collection layer."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd

from iffootball.collectors.statsbomb import (
    CompetitionTarget,
    get_target_matches,
    load_targets,
)


class _MockCollector:
    """Minimal mock of StatsBombDataSource for unit tests."""

    def get_competitions(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"competition_id": [2], "competition_name": ["Premier League"]}
        )

    def get_matches(self, competition_id: int, season_id: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "match_id": [1, 2, 3],
                "home_team": ["Manchester United", "Arsenal", "Liverpool"],
                "away_team": ["Chelsea", "Manchester United", "Tottenham Hotspur"],
            }
        )

    def get_events(self, match_id: int) -> pd.DataFrame:
        return pd.DataFrame({"event_id": [1], "type": ["Pass"]})

    def get_lineups(self, match_id: int) -> dict[str, pd.DataFrame]:
        return {
            "Manchester United": pd.DataFrame({"player_name": ["Player A"]}),
            "Chelsea": pd.DataFrame({"player_name": ["Player B"]}),
        }


def test_load_targets_single_competition(tmp_path: Path) -> None:
    config_file = tmp_path / "targets.toml"
    config_file.write_text(
        textwrap.dedent("""\
            [[competitions]]
            competition_id = 2
            season_id = 27
            clubs = ["Manchester United", "Arsenal"]
        """)
    )

    targets = load_targets(config_file)

    assert len(targets) == 1
    assert targets[0].competition_id == 2
    assert targets[0].season_id == 27
    assert targets[0].clubs == ["Manchester United", "Arsenal"]


def test_load_targets_multiple_competitions(tmp_path: Path) -> None:
    config_file = tmp_path / "targets.toml"
    config_file.write_text(
        textwrap.dedent("""\
            [[competitions]]
            competition_id = 2
            season_id = 27
            clubs = ["Manchester United"]

            [[competitions]]
            competition_id = 11
            season_id = 27
            clubs = ["Real Madrid"]
        """)
    )

    targets = load_targets(config_file)

    assert len(targets) == 2
    assert targets[1].competition_id == 11


def test_get_target_matches_filters_by_club() -> None:
    collector = _MockCollector()
    targets = [
        CompetitionTarget(competition_id=2, season_id=27, clubs=["Manchester United"])
    ]

    result = get_target_matches(collector, targets)

    # match_id 1: MU vs Chelsea (home = MU) -> included
    # match_id 2: Arsenal vs MU (away = MU) -> included
    # match_id 3: Liverpool vs Tottenham -> excluded
    assert len(result) == 2
    assert set(result["match_id"]) == {1, 2}


def test_get_target_matches_empty_targets() -> None:
    collector = _MockCollector()
    result = get_target_matches(collector, [])

    assert result.empty


def test_get_target_matches_name_must_match_exactly() -> None:
    # Club names are compared exactly. A mismatch (e.g. missing accent) returns 0 rows.
    collector = _MockCollector()
    targets = [
        CompetitionTarget(
            competition_id=2, season_id=27, clubs=["manchester united"]  # wrong case
        )
    ]

    result = get_target_matches(collector, targets)

    assert result.empty
