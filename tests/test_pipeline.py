"""Tests for the initialization pipeline."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from iffootball.pipeline import (
    InitializationResult,
    build_league_context,
    initialize,
)
from iffootball.storage.db import Database


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEAM = "Team A"
OPP_1 = "Team B"
OPP_2 = "Team C"
MANAGER = "Manager X"
COMPETITION_ID = 1
SEASON_ID = 1
TRIGGER_WEEK = 2
LEAGUE_NAME = "Test League"

# Standard 4-3-3 starting XI positions
_433_POSITIONS = [
    "Goalkeeper",
    "Right Back",
    "Right Center Back",
    "Left Center Back",
    "Left Back",
    "Center Defensive Midfield",
    "Right Center Midfield",
    "Left Center Midfield",
    "Right Wing",
    "Left Wing",
    "Center Forward",
]


# ---------------------------------------------------------------------------
# Mock StatsBombDataSource
# ---------------------------------------------------------------------------


def _make_lineup_row(
    player_id: int,
    player_name: str,
    position_name: str,
) -> dict[str, object]:
    """Build a lineup row with a starting XI position."""
    return {
        "player_id": player_id,
        "player_name": player_name,
        "positions": [
            {"position": position_name, "start_reason": "Starting XI"},
        ],
    }


def _make_player_events(
    match_id: int,
    team: str,
    player_id: int,
    position: str,
    minute_max: int = 90,
) -> list[dict[str, object]]:
    """Generate a basic set of events for one player in one match.

    Produces Pass, Pressure, Tackle, Carry, and Shot events so that all
    converter functions can compute meaningful values.
    """
    base = {
        "match_id": match_id,
        "team": team,
        "player_id": player_id,
        "position": position,
    }
    events: list[dict[str, object]] = []
    # Pass events (10 per match) — some progressive
    for i in range(10):
        events.append(
            {
                **base,
                "type": "Pass",
                "minute": min(i * 9, minute_max),
                "pass_outcome": None,
                "location": [30.0, 40.0],
                "pass_end_location": [50.0, 40.0],  # +20 → progressive
                "shot_statsbomb_xg": None,
            }
        )
    # Pressure events (5)
    for i in range(5):
        events.append(
            {
                **base,
                "type": "Pressure",
                "minute": i * 18,
                "pass_outcome": None,
                "location": None,
                "pass_end_location": None,
                "shot_statsbomb_xg": None,
            }
        )
    # Tackle events (3)
    for i in range(3):
        events.append(
            {
                **base,
                "type": "Tackle",
                "minute": i * 30,
                "pass_outcome": None,
                "location": None,
                "pass_end_location": None,
                "shot_statsbomb_xg": None,
            }
        )
    # Carry events (5)
    for i in range(5):
        events.append(
            {
                **base,
                "type": "Carry",
                "minute": i * 18,
                "pass_outcome": None,
                "location": None,
                "pass_end_location": None,
                "shot_statsbomb_xg": None,
            }
        )
    # Shot event (1)
    events.append(
        {
            **base,
            "type": "Shot",
            "minute": 45,
            "pass_outcome": None,
            "location": [100.0, 40.0],
            "pass_end_location": None,
            "shot_statsbomb_xg": 0.3,
        }
    )
    return events


class MockCollector:
    """Mock StatsBombDataSource that produces deterministic test data.

    Simulates a 3-team, 3-week league:
      Week 1: Team A vs Team B (match 101), Team C bye
      Week 2: Team A vs Team C (match 102), Team B bye
      Week 3: Team B vs Team A (match 103), Team C bye

    trigger_week=2 means:
      - Pre-trigger matches: 101, 102 (weeks 1-2)
      - Post-trigger fixture: 103 (week 3)
    """

    def get_competitions(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"competition_id": COMPETITION_ID, "season_id": SEASON_ID}]
        )

    def get_matches(
        self, competition_id: int, season_id: int
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "match_id": 101,
                    "match_week": 1,
                    "home_team": TEAM,
                    "away_team": OPP_1,
                    "home_score": 2,
                    "away_score": 1,
                    "home_managers": MANAGER,
                    "away_managers": "Opp Manager 1",
                },
                {
                    "match_id": 102,
                    "match_week": 2,
                    "home_team": TEAM,
                    "away_team": OPP_2,
                    "home_score": 1,
                    "away_score": 1,
                    "home_managers": MANAGER,
                    "away_managers": "Opp Manager 2",
                },
                {
                    "match_id": 103,
                    "match_week": 3,
                    "home_team": OPP_1,
                    "away_team": TEAM,
                    "home_score": 0,
                    "away_score": 1,
                    "home_managers": "Opp Manager 1",
                    "away_managers": MANAGER,
                },
            ]
        )

    def get_events(self, match_id: int) -> pd.DataFrame:
        events_map = {
            101: self._match_101_events(),
            102: self._match_102_events(),
            103: self._match_103_events(),
        }
        return events_map[match_id]

    def get_lineups(self, match_id: int) -> dict[str, pd.DataFrame]:
        lineups_map = {
            101: self._match_101_lineups(),
            102: self._match_102_lineups(),
            103: self._match_103_lineups(),
        }
        return lineups_map[match_id]

    def _make_team_events(
        self,
        match_id: int,
        team: str,
        player_ids: list[int],
    ) -> list[dict[str, object]]:
        """Generate events for all 11 players of a team in a match."""
        all_events: list[dict[str, object]] = []
        for i, pid in enumerate(player_ids):
            pos = _433_POSITIONS[i % len(_433_POSITIONS)]
            all_events.extend(
                _make_player_events(match_id, team, pid, pos)
            )
        return all_events

    def _match_101_events(self) -> pd.DataFrame:
        team_a = self._make_team_events(101, TEAM, list(range(1, 12)))
        team_b = self._make_team_events(101, OPP_1, list(range(101, 112)))
        return pd.DataFrame(team_a + team_b)

    def _match_102_events(self) -> pd.DataFrame:
        team_a = self._make_team_events(102, TEAM, list(range(1, 12)))
        team_c = self._make_team_events(102, OPP_2, list(range(201, 212)))
        return pd.DataFrame(team_a + team_c)

    def _match_103_events(self) -> pd.DataFrame:
        team_b = self._make_team_events(103, OPP_1, list(range(101, 112)))
        team_a = self._make_team_events(103, TEAM, list(range(1, 12)))
        return pd.DataFrame(team_a + team_b)

    def _make_team_lineup(
        self,
        player_ids: list[int],
        team: str,
    ) -> pd.DataFrame:
        rows = []
        for i, pid in enumerate(player_ids):
            pos = _433_POSITIONS[i % len(_433_POSITIONS)]
            rows.append(_make_lineup_row(pid, f"{team} Player {pid}", pos))
        return pd.DataFrame(rows)

    def _match_101_lineups(self) -> dict[str, pd.DataFrame]:
        return {
            TEAM: self._make_team_lineup(list(range(1, 12)), TEAM),
            OPP_1: self._make_team_lineup(list(range(101, 112)), OPP_1),
        }

    def _match_102_lineups(self) -> dict[str, pd.DataFrame]:
        return {
            TEAM: self._make_team_lineup(list(range(1, 12)), TEAM),
            OPP_2: self._make_team_lineup(list(range(201, 212)), OPP_2),
        }

    def _match_103_lineups(self) -> dict[str, pd.DataFrame]:
        return {
            OPP_1: self._make_team_lineup(list(range(101, 112)), OPP_1),
            TEAM: self._make_team_lineup(list(range(1, 12)), TEAM),
        }


# ---------------------------------------------------------------------------
# Mock LLMClient
# ---------------------------------------------------------------------------


class MockLLMClient:
    """Mock LLM that returns predictable JSON responses."""

    def complete(self, messages: list[dict[str, str]]) -> str:
        user_msg = messages[-1]["content"]
        data = json.loads(user_msg)
        if data.get("query_type") == "manager_style":
            return json.dumps(
                {
                    "style_stubbornness": "high",
                    "preferred_formation": None,
                }
            )
        if data.get("query_type") == "league_characteristics":
            return json.dumps(
                {
                    "pressing_level": "high",
                    "physicality_level": "mid",
                    "tactical_complexity": "low",
                }
            )
        return "{}"


# ---------------------------------------------------------------------------
# Tests: build_league_context
# ---------------------------------------------------------------------------


class TestBuildLeagueContext:
    def test_computes_league_averages(self) -> None:
        collector = MockCollector()
        all_matches = collector.get_matches(COMPETITION_ID, SEASON_ID)
        pre_matches = all_matches[
            all_matches["match_week"] <= TRIGGER_WEEK
        ].copy()
        pre_match_ids = set(int(mid) for mid in pre_matches["match_id"])
        pre_events = pd.concat(
            [collector.get_events(mid) for mid in sorted(pre_match_ids)],
            ignore_index=True,
        )

        ctx = build_league_context(
            pre_events, pre_matches, COMPETITION_ID, SEASON_ID, LEAGUE_NAME,
        )

        assert ctx.competition_id == COMPETITION_ID
        assert ctx.season_id == SEASON_ID
        assert ctx.name == LEAGUE_NAME
        assert ctx.avg_ppda > 0
        assert ctx.avg_xg_per90 > 0
        assert ctx.avg_progressive_passes_per90 > 0
        # Hypothesis fields remain None (no LLM)
        assert ctx.pressing_level is None

    def test_empty_events_returns_defaults(self) -> None:
        matches = pd.DataFrame(
            [
                {
                    "match_id": 1,
                    "match_week": 1,
                    "home_team": "A",
                    "away_team": "B",
                    "home_score": 1,
                    "away_score": 0,
                }
            ]
        )
        ctx = build_league_context(
            pd.DataFrame(), matches, 1, 1, "Empty League",
        )
        assert ctx.avg_ppda == 0.0
        assert ctx.avg_xg_per90 == 0.0


# ---------------------------------------------------------------------------
# Tests: initialize
# ---------------------------------------------------------------------------


class TestInitialize:
    """Integration tests for the full pipeline."""

    def test_basic_pipeline_no_llm_no_db(self) -> None:
        result = initialize(
            collector=MockCollector(),
            competition_id=COMPETITION_ID,
            season_id=SEASON_ID,
            team_name=TEAM,
            manager_name=MANAGER,
            trigger_week=TRIGGER_WEEK,
            league_name=LEAGUE_NAME,
        )

        assert isinstance(result, InitializationResult)

        # PlayerAgents: empty because 2 matches (180 min) < 900 min threshold
        assert result.player_agents == []

        # TeamBaseline
        tb = result.team_baseline
        assert tb.team_name == TEAM
        assert tb.competition_id == COMPETITION_ID
        # cultural_inertia updated from manager tenure (not default 0.5)
        assert tb.cultural_inertia != 0.5

        # ManagerAgent
        ma = result.manager_agent
        assert ma.manager_name == MANAGER
        assert ma.team_name == TEAM
        assert len(ma.tenure_match_ids) > 0
        # style_stubbornness remains default (no LLM)
        assert ma.style_stubbornness == 50.0

        # FixtureList has only post-trigger fixtures
        fl = result.fixture_list
        assert fl.team_name == TEAM
        for f in fl.fixtures:
            assert f.match_week > TRIGGER_WEEK

        # OpponentStrength for each fixture opponent
        for f in fl.fixtures:
            assert f.opponent_name in result.opponent_strengths

        # LeagueContext
        lc = result.league_context
        assert lc.name == LEAGUE_NAME
        assert lc.avg_xg_per90 > 0

    def test_trigger_week_boundary(self) -> None:
        """Verify no future data leaks into pre-trigger agents."""
        collector = MockCollector()

        result = initialize(
            collector=collector,
            competition_id=COMPETITION_ID,
            season_id=SEASON_ID,
            team_name=TEAM,
            manager_name=MANAGER,
            trigger_week=TRIGGER_WEEK,
            league_name=LEAGUE_NAME,
        )

        # TeamBaseline.played_match_ids should not include match 103 (week 3)
        assert 103 not in result.team_baseline.played_match_ids
        assert all(
            mid in {101, 102}
            for mid in result.team_baseline.played_match_ids
        )

        # ManagerAgent tenure should not include match 103
        assert 103 not in result.manager_agent.tenure_match_ids

    def test_with_llm_enrichment(self) -> None:
        result = initialize(
            collector=MockCollector(),
            competition_id=COMPETITION_ID,
            season_id=SEASON_ID,
            team_name=TEAM,
            manager_name=MANAGER,
            trigger_week=TRIGGER_WEEK,
            league_name=LEAGUE_NAME,
            llm_client=MockLLMClient(),
        )

        # style_stubbornness updated by LLM (high → 80.0)
        assert result.manager_agent.style_stubbornness == 80.0

        # LeagueContext hypothesis fields populated
        lc = result.league_context
        assert lc.pressing_level == "high"
        assert lc.physicality_level == "mid"
        assert lc.tactical_complexity == "low"

    def test_with_db_persistence(self) -> None:
        with Database(":memory:") as db:
            result = initialize(
                collector=MockCollector(),
                competition_id=COMPETITION_ID,
                season_id=SEASON_ID,
                team_name=TEAM,
                manager_name=MANAGER,
                trigger_week=TRIGGER_WEEK,
                league_name=LEAGUE_NAME,
                db=db,
            )

            # Verify data was persisted and can be loaded back
            loaded_players = db.load_player_agents(COMPETITION_ID, SEASON_ID)
            assert len(loaded_players) == len(result.player_agents)

            loaded_tb = db.load_team_baseline(
                TEAM, COMPETITION_ID, SEASON_ID
            )
            assert loaded_tb is not None
            assert loaded_tb.team_name == TEAM

            loaded_ma = db.load_manager_agent(
                MANAGER, TEAM, COMPETITION_ID, SEASON_ID
            )
            assert loaded_ma is not None
            assert loaded_ma.manager_name == MANAGER

            loaded_fl = db.load_fixture_list(
                TEAM, COMPETITION_ID, SEASON_ID
            )
            assert loaded_fl is not None
            assert len(loaded_fl.fixtures) == len(result.fixture_list.fixtures)

            loaded_os = db.load_opponent_strengths(
                COMPETITION_ID, SEASON_ID
            )
            assert len(loaded_os) == len(result.opponent_strengths)

    def test_cultural_inertia_updated_from_tenure(self) -> None:
        result = initialize(
            collector=MockCollector(),
            competition_id=COMPETITION_ID,
            season_id=SEASON_ID,
            team_name=TEAM,
            manager_name=MANAGER,
            trigger_week=TRIGGER_WEEK,
            league_name=LEAGUE_NAME,
        )

        # Manager has 2 tenure matches (week 1 and 2)
        tenure_len = len(result.manager_agent.tenure_match_ids)
        expected_ci = min(tenure_len / 38, 1.0)
        assert result.team_baseline.cultural_inertia == pytest.approx(
            expected_ci
        )

    def test_trigger_week_zero_no_pre_trigger_data(self) -> None:
        """Pipeline must not crash when trigger_week is before any match."""
        result = initialize(
            collector=MockCollector(),
            competition_id=COMPETITION_ID,
            season_id=SEASON_ID,
            team_name=TEAM,
            manager_name=MANAGER,
            trigger_week=0,  # all matches are week >= 1
            league_name=LEAGUE_NAME,
        )

        assert isinstance(result, InitializationResult)
        # No pre-trigger data → empty / default objects
        assert result.player_agents == []
        assert result.team_baseline.played_match_ids == frozenset()
        assert result.team_baseline.xg_for_per90 == 0.0
        assert result.manager_agent.tenure_match_ids == frozenset()
        assert result.team_baseline.cultural_inertia == 0.0

        # FixtureList should contain ALL matches (all are post-trigger)
        assert len(result.fixture_list.fixtures) == 3  # matches 101, 102, 103

        # LeagueContext defaults (no events to compute from)
        assert result.league_context.avg_ppda == 0.0
        assert result.league_context.avg_xg_per90 == 0.0
