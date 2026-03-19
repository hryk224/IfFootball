"""Tests for SQLite storage layer."""

from __future__ import annotations

import math

import pytest

from iffootball.agents.fixture import Fixture, FixtureList, OpponentStrength
from iffootball.agents.league import LeagueContext
from iffootball.agents.manager import ManagerAgent
from iffootball.agents.player import BroadPosition, PlayerAgent, RoleFamily
from iffootball.agents.team import TeamBaseline
from iffootball.simulation.cascade_tracker import CascadeEvent
from iffootball.simulation.comparison import (
    AggregatedResult,
    ComparisonResult,
    DeltaMetrics,
)
from iffootball.storage.db import Database, SchemaVersionError, _SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Fixtures (pytest)
# ---------------------------------------------------------------------------


@pytest.fixture()
def db() -> Database:
    return Database(":memory:")


def _make_player(player_id: int = 1) -> PlayerAgent:
    return PlayerAgent(
        player_id=player_id,
        player_name="Test Player",
        team_name="Test Team",
        position_name="Center Forward",
        role_family=RoleFamily.FORWARD,
        broad_position=BroadPosition.FW,
        pace=70.0,
        passing=60.0,
        shooting=80.0,
        pressing=55.0,
        defending=30.0,
        physicality=65.0,
        consistency=75.0,
    )


def _make_team_baseline(team_name: str = "Arsenal") -> TeamBaseline:
    return TeamBaseline(
        team_name=team_name,
        competition_id=2,
        season_id=27,
        played_match_ids=frozenset({1, 2, 3}),
        xg_for_per90=1.5,
        xg_against_per90=0.8,
        ppda=9.2,
        progressive_passes_per90=7.3,
        possession_pct=0.58,
        league_position=3,
        points_to_safety=20,
        points_to_title=-5,
        matches_remaining=10,
        cultural_inertia=0.7,
    )


def _make_manager_agent(
    squad_trust: dict[str, float] | None = None,
) -> ManagerAgent:
    return ManagerAgent(
        manager_name="Arsène Wenger",
        team_name="Arsenal",
        competition_id=2,
        season_id=27,
        tenure_match_ids=frozenset({1, 2, 3}),
        pressing_intensity=12.5,
        possession_preference=0.62,
        counter_tendency=0.38,
        preferred_formation="4-4-2",
        squad_trust=squad_trust or {},
    )


def _make_opponent_strengths() -> dict[str, OpponentStrength]:
    return {
        "Chelsea": OpponentStrength(
            opponent_name="Chelsea",
            xg_for_per90=1.3,
            xg_against_per90=0.9,
            elo_rating=1520.0,
        ),
        "Liverpool": OpponentStrength(
            opponent_name="Liverpool",
            xg_for_per90=1.8,
            xg_against_per90=0.7,
            elo_rating=1600.0,
        ),
    }


def _make_fixture_list() -> FixtureList:
    return FixtureList(
        team_name="Arsenal",
        trigger_week=29,
        fixtures=(
            Fixture(match_week=30, opponent_name="Chelsea", is_home=True),
            Fixture(match_week=31, opponent_name="Liverpool", is_home=False),
            Fixture(match_week=32, opponent_name="Everton", is_home=True),
        ),
    )


# ---------------------------------------------------------------------------
# TestDatabase (setup)
# ---------------------------------------------------------------------------


class TestDatabase:
    def test_in_memory_creation(self) -> None:
        db = Database(":memory:")
        db.close()

    def test_context_manager(self) -> None:
        with Database(":memory:") as db:
            assert db is not None

    def test_tables_created(self, db: Database) -> None:
        tables = {
            row[0]
            for row in db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "player_agents" in tables
        assert "team_baselines" in tables
        assert "manager_agents" in tables
        assert "opponent_strengths" in tables
        assert "fixture_lists" in tables
        assert "fixtures" in tables
        assert "db_meta" in tables
        assert "cascade_runs" in tables


class TestSchemaVersion:
    def test_new_db_gets_current_version(self, db: Database) -> None:
        row = db._conn.execute(
            "SELECT value FROM db_meta WHERE key = 'schema_version'"
        ).fetchone()
        assert row is not None
        assert int(row["value"]) == _SCHEMA_VERSION

    def test_reopening_same_version_succeeds(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.db"
            db1 = Database(path)
            db1.close()
            # Reopen — should not raise.
            db2 = Database(path)
            db2.close()

    def test_mismatched_version_raises(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.db"
            db1 = Database(path)
            # Tamper: set version to 999.
            db1._conn.execute(
                "UPDATE db_meta SET value = '999' WHERE key = 'schema_version'"
            )
            db1._conn.commit()
            db1.close()
            # Reopen — should raise.
            with pytest.raises(SchemaVersionError, match="999"):
                Database(path)

    def test_legacy_db_with_data_raises(self) -> None:
        import sqlite3
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy.db"
            # Create a legacy DB: has tables and data but no db_meta.
            conn = sqlite3.connect(str(path))
            conn.execute(
                "CREATE TABLE player_agents (player_id INTEGER PRIMARY KEY)"
            )
            conn.execute("INSERT INTO player_agents VALUES (1)")
            conn.commit()
            conn.close()
            # Opening via Database should detect legacy and raise.
            with pytest.raises(SchemaVersionError, match="Legacy"):
                Database(path)


# ---------------------------------------------------------------------------
# TestPlayerAgent
# ---------------------------------------------------------------------------


class TestPlayerAgent:
    def test_save_and_load_roundtrip(self, db: Database) -> None:
        agent = _make_player()
        db.save_player_agents([agent], competition_id=2, season_id=27)
        loaded = db.load_player_agents(competition_id=2, season_id=27)
        assert len(loaded) == 1
        assert loaded[0].player_id == agent.player_id
        assert loaded[0].player_name == agent.player_name

    def test_numeric_attributes_preserved(self, db: Database) -> None:
        agent = _make_player()
        db.save_player_agents([agent], competition_id=2, season_id=27)
        loaded = db.load_player_agents(competition_id=2, season_id=27)[0]
        assert loaded.pace == pytest.approx(70.0)
        assert loaded.shooting == pytest.approx(80.0)
        assert loaded.bench_streak == 0

    def test_enum_fields_preserved(self, db: Database) -> None:
        agent = _make_player()
        db.save_player_agents([agent], competition_id=2, season_id=27)
        loaded = db.load_player_agents(competition_id=2, season_id=27)[0]
        assert loaded.role_family == RoleFamily.FORWARD
        assert loaded.broad_position == BroadPosition.FW

    def test_multiple_agents(self, db: Database) -> None:
        agents = [_make_player(i) for i in range(1, 5)]
        db.save_player_agents(agents, competition_id=2, season_id=27)
        loaded = db.load_player_agents(competition_id=2, season_id=27)
        assert len(loaded) == 4

    def test_upsert_overwrites(self, db: Database) -> None:
        agent = _make_player()
        db.save_player_agents([agent], competition_id=2, season_id=27)
        updated = PlayerAgent(
            player_id=agent.player_id,
            player_name="Updated Name",
            team_name="Test Team",
            position_name=agent.position_name,
            role_family=agent.role_family,
            broad_position=agent.broad_position,
            pace=99.0,
            passing=agent.passing,
            shooting=agent.shooting,
            pressing=agent.pressing,
            defending=agent.defending,
            physicality=agent.physicality,
            consistency=agent.consistency,
        )
        db.save_player_agents([updated], competition_id=2, season_id=27)
        loaded = db.load_player_agents(competition_id=2, season_id=27)
        assert len(loaded) == 1
        assert loaded[0].player_name == "Updated Name"
        assert loaded[0].pace == pytest.approx(99.0)

    def test_empty_result_for_missing_season(self, db: Database) -> None:
        db.save_player_agents([_make_player()], competition_id=2, season_id=27)
        assert db.load_player_agents(competition_id=2, season_id=99) == []


# ---------------------------------------------------------------------------
# TestTeamBaseline
# ---------------------------------------------------------------------------


class TestTeamBaseline:
    def test_save_and_load_roundtrip(self, db: Database) -> None:
        baseline = _make_team_baseline()
        db.save_team_baseline(baseline)
        loaded = db.load_team_baseline("Arsenal", 2, 27)
        assert loaded is not None
        assert loaded.team_name == baseline.team_name
        assert loaded.league_position == baseline.league_position

    def test_played_match_ids_roundtrip(self, db: Database) -> None:
        baseline = _make_team_baseline()
        db.save_team_baseline(baseline)
        loaded = db.load_team_baseline("Arsenal", 2, 27)
        assert loaded is not None
        assert loaded.played_match_ids == frozenset({1, 2, 3})

    def test_ppda_nan_roundtrip(self, db: Database) -> None:
        baseline = TeamBaseline(
            team_name="Arsenal",
            competition_id=2,
            season_id=27,
            played_match_ids=frozenset({1}),
            xg_for_per90=1.0,
            xg_against_per90=1.0,
            ppda=float("nan"),
            progressive_passes_per90=5.0,
            possession_pct=0.5,
            league_position=5,
            points_to_safety=10,
            points_to_title=-10,
            matches_remaining=15,
        )
        db.save_team_baseline(baseline)
        loaded = db.load_team_baseline("Arsenal", 2, 27)
        assert loaded is not None
        assert math.isnan(loaded.ppda)

    def test_returns_none_when_missing(self, db: Database) -> None:
        assert db.load_team_baseline("Arsenal", 2, 99) is None

    def test_upsert_overwrites(self, db: Database) -> None:
        db.save_team_baseline(_make_team_baseline())
        updated = TeamBaseline(
            team_name="Arsenal",
            competition_id=2,
            season_id=27,
            played_match_ids=frozenset({1, 2, 3, 4}),
            xg_for_per90=2.0,
            xg_against_per90=0.5,
            ppda=8.0,
            progressive_passes_per90=9.0,
            possession_pct=0.65,
            league_position=1,
            points_to_safety=30,
            points_to_title=0,
            matches_remaining=5,
            cultural_inertia=0.9,
        )
        db.save_team_baseline(updated)
        loaded = db.load_team_baseline("Arsenal", 2, 27)
        assert loaded is not None
        assert loaded.league_position == 1
        assert loaded.xg_for_per90 == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# TestManagerAgent
# ---------------------------------------------------------------------------


class TestManagerAgent:
    def test_save_and_load_roundtrip(self, db: Database) -> None:
        agent = _make_manager_agent()
        db.save_manager_agent(agent)
        loaded = db.load_manager_agent("Arsène Wenger", "Arsenal", 2, 27)
        assert loaded is not None
        assert loaded.manager_name == agent.manager_name
        assert loaded.pressing_intensity == pytest.approx(agent.pressing_intensity)

    def test_tenure_match_ids_roundtrip(self, db: Database) -> None:
        db.save_manager_agent(_make_manager_agent())
        loaded = db.load_manager_agent("Arsène Wenger", "Arsenal", 2, 27)
        assert loaded is not None
        assert loaded.tenure_match_ids == frozenset({1, 2, 3})

    def test_preferred_formation_none(self, db: Database) -> None:
        agent = ManagerAgent(
            manager_name="Arsène Wenger",
            team_name="Arsenal",
            competition_id=2,
            season_id=27,
            tenure_match_ids=frozenset(),
            pressing_intensity=0.0,
            possession_preference=0.5,
            counter_tendency=0.5,
            preferred_formation=None,
        )
        db.save_manager_agent(agent)
        loaded = db.load_manager_agent("Arsène Wenger", "Arsenal", 2, 27)
        assert loaded is not None
        assert loaded.preferred_formation is None

    def test_squad_trust_roundtrip(self, db: Database) -> None:
        agent = _make_manager_agent(squad_trust={"Player A": 0.8, "Player B": 0.4})
        db.save_manager_agent(agent)
        loaded = db.load_manager_agent("Arsène Wenger", "Arsenal", 2, 27)
        assert loaded is not None
        assert loaded.squad_trust["Player A"] == pytest.approx(0.8)

    def test_returns_none_when_missing(self, db: Database) -> None:
        assert db.load_manager_agent("Unknown", "Arsenal", 2, 27) is None

    def test_upsert_overwrites(self, db: Database) -> None:
        db.save_manager_agent(_make_manager_agent())
        updated = ManagerAgent(
            manager_name="Arsène Wenger",
            team_name="Arsenal",
            competition_id=2,
            season_id=27,
            tenure_match_ids=frozenset({1, 2, 3, 4}),
            pressing_intensity=20.0,
            possession_preference=0.7,
            counter_tendency=0.3,
            preferred_formation="4-3-3",
        )
        db.save_manager_agent(updated)
        loaded = db.load_manager_agent("Arsène Wenger", "Arsenal", 2, 27)
        assert loaded is not None
        assert loaded.preferred_formation == "4-3-3"
        assert loaded.pressing_intensity == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# TestOpponentStrengths
# ---------------------------------------------------------------------------


class TestOpponentStrengths:
    def test_save_and_load_roundtrip(self, db: Database) -> None:
        strengths = _make_opponent_strengths()
        db.save_opponent_strengths(strengths, competition_id=2, season_id=27, trigger_week=29)
        loaded = db.load_opponent_strengths(competition_id=2, season_id=27, trigger_week=29)
        assert set(loaded.keys()) == {"Chelsea", "Liverpool"}

    def test_values_preserved(self, db: Database) -> None:
        strengths = _make_opponent_strengths()
        db.save_opponent_strengths(strengths, competition_id=2, season_id=27, trigger_week=29)
        loaded = db.load_opponent_strengths(competition_id=2, season_id=27, trigger_week=29)
        assert loaded["Chelsea"].elo_rating == pytest.approx(1520.0)
        assert loaded["Liverpool"].xg_for_per90 == pytest.approx(1.8)

    def test_returns_empty_for_missing_trigger(self, db: Database) -> None:
        db.save_opponent_strengths(
            _make_opponent_strengths(), competition_id=2, season_id=27, trigger_week=29
        )
        assert db.load_opponent_strengths(competition_id=2, season_id=27, trigger_week=99) == {}

    def test_upsert_overwrites(self, db: Database) -> None:
        db.save_opponent_strengths(
            _make_opponent_strengths(), competition_id=2, season_id=27, trigger_week=29
        )
        updated = {
            "Chelsea": OpponentStrength(
                opponent_name="Chelsea",
                xg_for_per90=2.0,
                xg_against_per90=0.5,
                elo_rating=1600.0,
            )
        }
        db.save_opponent_strengths(updated, competition_id=2, season_id=27, trigger_week=29)
        loaded = db.load_opponent_strengths(competition_id=2, season_id=27, trigger_week=29)
        assert loaded["Chelsea"].elo_rating == pytest.approx(1600.0)
        # Liverpool should still be present from the first save
        assert "Liverpool" in loaded


# ---------------------------------------------------------------------------
# TestFixtureList
# ---------------------------------------------------------------------------


class TestFixtureList:
    def test_save_and_load_roundtrip(self, db: Database) -> None:
        fl = _make_fixture_list()
        db.save_fixture_list(fl, competition_id=2, season_id=27)
        loaded = db.load_fixture_list("Arsenal", 2, 27, trigger_week=29)
        assert loaded is not None
        assert loaded.team_name == fl.team_name
        assert loaded.trigger_week == fl.trigger_week

    def test_fixture_count_preserved(self, db: Database) -> None:
        fl = _make_fixture_list()
        db.save_fixture_list(fl, competition_id=2, season_id=27)
        loaded = db.load_fixture_list("Arsenal", 2, 27, trigger_week=29)
        assert loaded is not None
        assert len(loaded.fixtures) == 3

    def test_fixture_order_preserved_by_ordinal(self, db: Database) -> None:
        fl = _make_fixture_list()
        db.save_fixture_list(fl, competition_id=2, season_id=27)
        loaded = db.load_fixture_list("Arsenal", 2, 27, trigger_week=29)
        assert loaded is not None
        assert loaded.fixtures[0].opponent_name == "Chelsea"
        assert loaded.fixtures[1].opponent_name == "Liverpool"
        assert loaded.fixtures[2].opponent_name == "Everton"

    def test_fixture_fields_preserved(self, db: Database) -> None:
        fl = _make_fixture_list()
        db.save_fixture_list(fl, competition_id=2, season_id=27)
        loaded = db.load_fixture_list("Arsenal", 2, 27, trigger_week=29)
        assert loaded is not None
        f = loaded.fixtures[0]
        assert f.match_week == 30
        assert f.is_home is True

    def test_is_home_false_preserved(self, db: Database) -> None:
        fl = _make_fixture_list()
        db.save_fixture_list(fl, competition_id=2, season_id=27)
        loaded = db.load_fixture_list("Arsenal", 2, 27, trigger_week=29)
        assert loaded is not None
        assert loaded.fixtures[1].is_home is False

    def test_empty_fixture_list_roundtrip(self, db: Database) -> None:
        fl = FixtureList(team_name="Arsenal", trigger_week=38, fixtures=())
        db.save_fixture_list(fl, competition_id=2, season_id=27)
        loaded = db.load_fixture_list("Arsenal", 2, 27, trigger_week=38)
        assert loaded is not None
        assert loaded.fixtures == ()

    def test_returns_none_when_missing(self, db: Database) -> None:
        assert db.load_fixture_list("Arsenal", 2, 27, trigger_week=99) is None

    def test_resave_replaces_fixtures(self, db: Database) -> None:
        fl = _make_fixture_list()
        db.save_fixture_list(fl, competition_id=2, season_id=27)
        updated = FixtureList(
            team_name="Arsenal",
            trigger_week=29,
            fixtures=(
                Fixture(match_week=30, opponent_name="Chelsea", is_home=True),
            ),
        )
        db.save_fixture_list(updated, competition_id=2, season_id=27)
        loaded = db.load_fixture_list("Arsenal", 2, 27, trigger_week=29)
        assert loaded is not None
        assert len(loaded.fixtures) == 1

    def test_loaded_fixture_list_is_frozen(self, db: Database) -> None:
        fl = _make_fixture_list()
        db.save_fixture_list(fl, competition_id=2, season_id=27)
        loaded = db.load_fixture_list("Arsenal", 2, 27, trigger_week=29)
        assert loaded is not None
        with pytest.raises((AttributeError, TypeError)):
            loaded.fixtures = ()  # type: ignore[misc]

    def test_loaded_fixture_is_frozen(self, db: Database) -> None:
        fl = _make_fixture_list()
        db.save_fixture_list(fl, competition_id=2, season_id=27)
        loaded = db.load_fixture_list("Arsenal", 2, 27, trigger_week=29)
        assert loaded is not None
        with pytest.raises((AttributeError, TypeError)):
            loaded.fixtures[0].is_home = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestLeagueContext
# ---------------------------------------------------------------------------


def _make_league_context() -> LeagueContext:
    return LeagueContext(
        competition_id=2,
        season_id=27,
        name="Premier League",
        avg_ppda=10.5,
        avg_progressive_passes_per90=45.0,
        avg_xg_per90=1.3,
        pressing_level="high",
        physicality_level="mid",
        tactical_complexity=None,
    )


class TestLeagueContext:
    def test_save_and_load_roundtrip(self, db: Database) -> None:
        ctx = _make_league_context()
        db.save_league_context(ctx)
        loaded = db.load_league_context(2, 27)
        assert loaded is not None
        assert loaded.competition_id == 2
        assert loaded.name == "Premier League"
        assert loaded.avg_ppda == pytest.approx(10.5)
        assert loaded.pressing_level == "high"
        assert loaded.tactical_complexity is None

    def test_returns_none_when_missing(self, db: Database) -> None:
        assert db.load_league_context(99, 99) is None

    def test_upsert(self, db: Database) -> None:
        ctx = _make_league_context()
        db.save_league_context(ctx)
        from dataclasses import replace
        updated = replace(ctx, avg_ppda=12.0)
        db.save_league_context(updated)
        loaded = db.load_league_context(2, 27)
        assert loaded is not None
        assert loaded.avg_ppda == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# TestCascadeEvents
# ---------------------------------------------------------------------------


def _make_cascade_events() -> list[CascadeEvent]:
    return [
        CascadeEvent(
            week=5,
            event_type="form_drop",
            affected_agent="Player A",
            cause_chain=("trust_decline",),
            magnitude=0.3,
            depth=2,
        ),
        CascadeEvent(
            week=5,
            event_type="trust_decline",
            affected_agent="Player A",
            cause_chain=("form_drop", "trust_decline"),
            magnitude=0.2,
            depth=3,
        ),
    ]


class TestCascadeEvents:
    def test_save_and_load_roundtrip(self, db: Database) -> None:
        events = _make_cascade_events()
        db.save_cascade_events("cmp1", "a_0", events)
        loaded = db.load_cascade_events("cmp1", "a_0")
        assert len(loaded) == 2
        assert loaded[0].event_type == "form_drop"
        assert loaded[0].cause_chain == ("trust_decline",)
        assert loaded[1].depth == 3

    def test_empty_list(self, db: Database) -> None:
        db.save_cascade_events("cmp1", "a_0", [])
        loaded = db.load_cascade_events("cmp1", "a_0")
        assert loaded == []

    def test_missing_key_returns_empty(self, db: Database) -> None:
        loaded = db.load_cascade_events("nonexistent", "run_0")
        assert loaded == []

    def test_order_preserved(self, db: Database) -> None:
        events = _make_cascade_events()
        db.save_cascade_events("cmp1", "a_0", events)
        loaded = db.load_cascade_events("cmp1", "a_0")
        assert loaded[0].event_type == "form_drop"
        assert loaded[1].event_type == "trust_decline"

    def test_different_runs_isolated(self, db: Database) -> None:
        e1 = [_make_cascade_events()[0]]
        e2 = [_make_cascade_events()[1]]
        db.save_cascade_events("cmp1", "a_0", e1)
        db.save_cascade_events("cmp1", "b_0", e2)
        assert len(db.load_cascade_events("cmp1", "a_0")) == 1
        assert len(db.load_cascade_events("cmp1", "b_0")) == 1

    def test_run_was_saved_true(self, db: Database) -> None:
        db.save_cascade_events("cmp1", "a_0", [])
        assert db.cascade_run_was_saved("cmp1", "a_0") is True

    def test_run_was_saved_false(self, db: Database) -> None:
        assert db.cascade_run_was_saved("nonexistent", "run_0") is False

    def test_saved_empty_distinguishable_from_never_saved(self, db: Database) -> None:
        # Save an empty run.
        db.save_cascade_events("cmp1", "a_0", [])
        # Saved empty: was_saved=True, events=[]
        assert db.cascade_run_was_saved("cmp1", "a_0") is True
        assert db.load_cascade_events("cmp1", "a_0") == []
        # Never saved: was_saved=False, events=[]
        assert db.cascade_run_was_saved("cmp1", "b_0") is False
        assert db.load_cascade_events("cmp1", "b_0") == []


# ---------------------------------------------------------------------------
# TestComparisonResult
# ---------------------------------------------------------------------------


def _make_comparison_result() -> ComparisonResult:
    agg_a = AggregatedResult(
        n_runs=3,
        total_points_mean=5.0,
        total_points_median=4.0,
        total_points_std=1.5,
        cascade_event_counts={"form_drop": 1.0},
        run_results=(),
    )
    agg_b = AggregatedResult(
        n_runs=3,
        total_points_mean=7.0,
        total_points_median=6.0,
        total_points_std=2.0,
        cascade_event_counts={"form_drop": 2.0, "trust_decline": 0.5},
        run_results=(),
    )
    delta = DeltaMetrics(
        points_mean_diff=2.0,
        points_median_diff=2.0,
        cascade_count_diff={"form_drop": 1.0, "trust_decline": 0.5},
    )
    return ComparisonResult(no_change=agg_a, with_change=agg_b, delta=delta)


class TestComparisonResult:
    def test_save_and_load_roundtrip(self, db: Database) -> None:
        cr = _make_comparison_result()
        db.save_comparison_result("test_key", cr, rng_seed=42, trigger_summary="test trigger", rng_policy="paired_split_v1")
        loaded = db.load_comparison_result("test_key")
        assert loaded is not None
        assert loaded.result.no_change.n_runs == 3
        assert loaded.result.no_change.total_points_mean == pytest.approx(5.0)
        assert loaded.result.with_change.total_points_mean == pytest.approx(7.0)
        assert loaded.result.delta.points_mean_diff == pytest.approx(2.0)

    def test_cascade_event_counts_preserved(self, db: Database) -> None:
        cr = _make_comparison_result()
        db.save_comparison_result("k", cr, rng_seed=42, trigger_summary="test trigger", rng_policy="paired_split_v1")
        loaded = db.load_comparison_result("k")
        assert loaded is not None
        assert loaded.result.with_change.cascade_event_counts["trust_decline"] == pytest.approx(0.5)

    def test_returns_none_when_missing(self, db: Database) -> None:
        assert db.load_comparison_result("nonexistent") is None

    def test_run_results_empty_on_load(self, db: Database) -> None:
        cr = _make_comparison_result()
        db.save_comparison_result("k", cr, rng_seed=42, trigger_summary="test trigger", rng_policy="paired_split_v1")
        loaded = db.load_comparison_result("k")
        assert loaded is not None
        assert loaded.result.no_change.run_results == ()
        assert loaded.result.with_change.run_results == ()

    def test_upsert(self, db: Database) -> None:
        cr = _make_comparison_result()
        db.save_comparison_result("k", cr, rng_seed=42, trigger_summary="test trigger", rng_policy="paired_split_v1")
        # Save again with different data
        updated_agg = AggregatedResult(
            n_runs=5,
            total_points_mean=9.0,
            total_points_median=8.0,
            total_points_std=1.0,
            cascade_event_counts={},
            run_results=(),
        )
        updated = ComparisonResult(
            no_change=updated_agg,
            with_change=updated_agg,
            delta=DeltaMetrics(0.0, 0.0, {}),
        )
        db.save_comparison_result("k", updated, rng_seed=99, trigger_summary="updated", rng_policy="paired_split_v1")
        loaded = db.load_comparison_result("k")
        assert loaded is not None
        assert loaded.result.no_change.n_runs == 5

    def test_metadata_roundtrip(self, db: Database) -> None:
        cr = _make_comparison_result()
        db.save_comparison_result(
            "meta_key", cr, rng_seed=123, trigger_summary="Manager change: A -> B", rng_policy="paired_split_v1"
        )
        loaded = db.load_comparison_result("meta_key")
        assert loaded is not None
        assert loaded.meta is not None
        assert loaded.meta.rng_seed == 123
        assert loaded.meta.n_runs == 3
        assert loaded.meta.trigger_summary == "Manager change: A -> B"
        assert loaded.meta.rng_policy == "paired_split_v1"

    def test_created_at_is_utc_iso8601(self, db: Database) -> None:
        cr = _make_comparison_result()
        db.save_comparison_result("ts_key", cr, rng_seed=42, trigger_summary="test trigger", rng_policy="paired_split_v1")
        loaded = db.load_comparison_result("ts_key")
        assert loaded is not None
        assert loaded.meta is not None
        # ISO 8601 UTC format: YYYY-MM-DDTHH:MM:SSZ
        assert loaded.meta.created_at.endswith("Z")
        assert "T" in loaded.meta.created_at
