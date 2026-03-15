"""SQLite storage for agent and simulation data.

Provides a Database class that persists and restores domain objects:

  M1 agent data:
  - PlayerAgent       (one per player per competition/season)
  - TeamBaseline      (one snapshot per team per competition/season)
  - ManagerAgent      (one per manager×team per competition/season)
  - OpponentStrength  (one per opponent per competition/season/trigger_week)
  - FixtureList       (one per team per competition/season/trigger_week)
  - LeagueContext     (one per competition/season)

  M2 simulation results:
  - CascadeEvent      (per comparison_key + run_id, ordered by ordinal)
  - ComparisonResult  (aggregated statistics per comparison_key;
                       run_results are excluded — use run_results=()
                       on load since full per-run data is too large
                       to persist and can be reproduced from the same seed)
  - ComparisonMeta    (rng_seed, n_runs, trigger_summary, created_at)

Identifier convention (M1):
  player_id, team_name, manager_name, and opponent_name are used as canonical
  identifiers. In M1 these map directly to StatsBomb player IDs and team name
  strings. StatsBomb team names must be spelled exactly as they appear in the
  StatsBomb data (e.g. "Manchester United"). Any future normalisation layer
  (e.g. a separate key column) can be added in a later milestone without
  schema-breaking changes.

Snapshot policy:
  TeamBaseline and ManagerAgent store one snapshot per
  (team/manager, competition_id, season_id). Re-saving overwrites the
  previous snapshot via UPSERT (ON CONFLICT DO UPDATE).
  OpponentStrength and FixtureList are scoped to trigger_week so multiple
  snapshots per season are possible.

Serialisation:
  frozenset[int]       → TEXT (JSON-encoded sorted list)
  dict[str, float]     → TEXT (JSON-encoded object)
  RoleFamily / BroadPosition enum → TEXT (.value)
  float("nan") / ppda  → NULL
  bool (is_home)       → INTEGER (0 / 1)
  str | None           → TEXT / NULL

Note on cascade_events:
  load_cascade_events() returns [] for both "explicitly saved empty run"
  and "never saved". There is no header table to distinguish these cases.
  If that distinction is needed for auditing, a header table can be added
  in a future milestone.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

# ---------------------------------------------------------------------------
# Comparison metadata types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComparisonMeta:
    """Metadata for a saved comparison result.

    Attributes:
        rng_seed:        Random seed used for reproducibility.
        n_runs:          Number of simulation runs per branch.
        trigger_summary: Human-readable trigger description. This is a
                         display label, not a machine-parseable reconstruction
                         of the trigger. Do not use it to rebuild triggers.
        created_at:      UTC ISO 8601 timestamp of when the result was saved.
    """

    rng_seed: int
    n_runs: int
    trigger_summary: str
    created_at: str


@dataclass(frozen=True)
class ComparisonResultWithMeta:
    """ComparisonResult paired with its persistence metadata.

    Attributes:
        result: The aggregated A/B comparison result.
        meta:   Persistence metadata (seed, runs, trigger, timestamp).
                None when loaded from legacy rows that lack metadata.
    """

    result: ComparisonResult
    meta: ComparisonMeta | None


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS player_agents (
    player_id             INTEGER NOT NULL,
    competition_id        INTEGER NOT NULL,
    season_id             INTEGER NOT NULL,
    player_name           TEXT    NOT NULL,
    position_name         TEXT    NOT NULL,
    role_family           TEXT    NOT NULL,
    broad_position        TEXT    NOT NULL,
    pace                  REAL    NOT NULL,
    passing               REAL    NOT NULL,
    shooting              REAL    NOT NULL,
    pressing              REAL    NOT NULL,
    defending             REAL    NOT NULL,
    physicality           REAL    NOT NULL,
    consistency           REAL    NOT NULL,
    tactical_adaptability REAL    NOT NULL,
    leadership            REAL    NOT NULL,
    pressure_resistance   REAL    NOT NULL,
    current_form          REAL    NOT NULL,
    fatigue               REAL    NOT NULL,
    tactical_understanding REAL   NOT NULL,
    manager_trust         REAL    NOT NULL,
    bench_streak          INTEGER NOT NULL,
    PRIMARY KEY (player_id, competition_id, season_id)
);

CREATE TABLE IF NOT EXISTS team_baselines (
    team_name             TEXT    NOT NULL,
    competition_id        INTEGER NOT NULL,
    season_id             INTEGER NOT NULL,
    played_match_ids      TEXT    NOT NULL,
    xg_for_per90          REAL    NOT NULL,
    xg_against_per90      REAL    NOT NULL,
    ppda                  REAL,
    progressive_passes_per90 REAL NOT NULL,
    possession_pct        REAL    NOT NULL,
    league_position       INTEGER NOT NULL,
    points_to_safety      INTEGER NOT NULL,
    points_to_title       INTEGER NOT NULL,
    matches_remaining     INTEGER NOT NULL,
    cultural_inertia      REAL    NOT NULL,
    PRIMARY KEY (team_name, competition_id, season_id)
);

CREATE TABLE IF NOT EXISTS manager_agents (
    manager_name          TEXT    NOT NULL,
    team_name             TEXT    NOT NULL,
    competition_id        INTEGER NOT NULL,
    season_id             INTEGER NOT NULL,
    tenure_match_ids      TEXT    NOT NULL,
    pressing_intensity    REAL    NOT NULL,
    possession_preference REAL    NOT NULL,
    counter_tendency      REAL    NOT NULL,
    preferred_formation   TEXT,
    implementation_speed  REAL    NOT NULL,
    youth_development     REAL    NOT NULL,
    style_stubbornness    REAL    NOT NULL,
    job_security          REAL    NOT NULL,
    squad_trust           TEXT    NOT NULL,
    PRIMARY KEY (manager_name, team_name, competition_id, season_id)
);

CREATE TABLE IF NOT EXISTS opponent_strengths (
    opponent_name         TEXT    NOT NULL,
    competition_id        INTEGER NOT NULL,
    season_id             INTEGER NOT NULL,
    trigger_week          INTEGER NOT NULL,
    xg_for_per90          REAL    NOT NULL,
    xg_against_per90      REAL    NOT NULL,
    elo_rating            REAL    NOT NULL,
    PRIMARY KEY (opponent_name, competition_id, season_id, trigger_week)
);

CREATE TABLE IF NOT EXISTS fixture_lists (
    team_name             TEXT    NOT NULL,
    competition_id        INTEGER NOT NULL,
    season_id             INTEGER NOT NULL,
    trigger_week          INTEGER NOT NULL,
    PRIMARY KEY (team_name, competition_id, season_id, trigger_week)
);

CREATE TABLE IF NOT EXISTS fixtures (
    team_name             TEXT    NOT NULL,
    competition_id        INTEGER NOT NULL,
    season_id             INTEGER NOT NULL,
    trigger_week          INTEGER NOT NULL,
    ordinal               INTEGER NOT NULL,
    match_week            INTEGER NOT NULL,
    opponent_name         TEXT    NOT NULL,
    is_home               INTEGER NOT NULL,
    PRIMARY KEY (team_name, competition_id, season_id, trigger_week, ordinal)
);

CREATE TABLE IF NOT EXISTS league_contexts (
    competition_id        INTEGER NOT NULL,
    season_id             INTEGER NOT NULL,
    name                  TEXT    NOT NULL,
    avg_ppda              REAL    NOT NULL,
    avg_progressive_passes_per90 REAL NOT NULL,
    avg_xg_per90          REAL    NOT NULL,
    pressing_level        TEXT,
    physicality_level     TEXT,
    tactical_complexity   TEXT,
    PRIMARY KEY (competition_id, season_id)
);

CREATE TABLE IF NOT EXISTS cascade_events (
    comparison_key        TEXT    NOT NULL,
    run_id                TEXT    NOT NULL,
    ordinal               INTEGER NOT NULL,
    week                  INTEGER NOT NULL,
    event_type            TEXT    NOT NULL,
    affected_agent        TEXT    NOT NULL,
    cause_chain           TEXT    NOT NULL,
    magnitude             REAL    NOT NULL,
    depth                 INTEGER NOT NULL,
    PRIMARY KEY (comparison_key, run_id, ordinal)
);

CREATE TABLE IF NOT EXISTS comparison_results (
    comparison_key        TEXT NOT NULL PRIMARY KEY,
    no_change_json        TEXT NOT NULL,
    with_change_json      TEXT NOT NULL,
    delta_json            TEXT NOT NULL,
    rng_seed              INTEGER,
    n_runs                INTEGER,
    trigger_summary       TEXT,
    created_at            TEXT
);
"""

# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _encode_frozenset(ids: frozenset[int]) -> str:
    return json.dumps(sorted(ids))


def _decode_frozenset(text: str) -> frozenset[int]:
    return frozenset(json.loads(text))


def _encode_dict(d: dict[str, float]) -> str:
    return json.dumps(d)


def _decode_dict(text: str) -> dict[str, float]:
    return dict(json.loads(text))


def _encode_aggregated(agg: AggregatedResult) -> str:
    """Serialise AggregatedResult to JSON (excluding run_results)."""
    return json.dumps(
        {
            "n_runs": agg.n_runs,
            "total_points_mean": agg.total_points_mean,
            "total_points_median": agg.total_points_median,
            "total_points_std": agg.total_points_std,
            "cascade_event_counts": agg.cascade_event_counts,
        }
    )


def _decode_aggregated(text: str) -> AggregatedResult:
    """Deserialise AggregatedResult from JSON. run_results is empty tuple."""
    d = json.loads(text)
    return AggregatedResult(
        n_runs=int(d["n_runs"]),
        total_points_mean=float(d["total_points_mean"]),
        total_points_median=float(d["total_points_median"]),
        total_points_std=float(d["total_points_std"]),
        cascade_event_counts={
            str(k): float(v) for k, v in d["cascade_event_counts"].items()
        },
        run_results=(),
    )


def _encode_delta(delta: DeltaMetrics) -> str:
    """Serialise DeltaMetrics to JSON."""
    return json.dumps(
        {
            "points_mean_diff": delta.points_mean_diff,
            "points_median_diff": delta.points_median_diff,
            "cascade_count_diff": delta.cascade_count_diff,
        }
    )


def _decode_delta(text: str) -> DeltaMetrics:
    """Deserialise DeltaMetrics from JSON."""
    d = json.loads(text)
    return DeltaMetrics(
        points_mean_diff=float(d["points_mean_diff"]),
        points_median_diff=float(d["points_median_diff"]),
        cascade_count_diff={
            str(k): float(v) for k, v in d["cascade_count_diff"].items()
        },
    )


def _encode_float(value: float) -> float | None:
    """Convert NaN to None (SQLite NULL)."""
    return None if math.isnan(value) else value


def _decode_float(value: Any) -> float:
    """Convert SQLite NULL back to NaN."""
    return float("nan") if value is None else float(value)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


class Database:
    """SQLite-backed store for M1 agent data.

    Usage:
        db = Database("path/to/data.db")
        db.save_player_agents(agents, competition_id=2, season_id=27)
        agents = db.load_player_agents(competition_id=2, season_id=27)
        db.close()

    Or as a context manager:
        with Database("path/to/data.db") as db:
            db.save_team_baseline(baseline)

    Passing ":memory:" creates a transient in-memory database (useful for
    tests and one-off scripts).
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript(_DDL)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -----------------------------------------------------------------------
    # PlayerAgent
    # -----------------------------------------------------------------------

    def save_player_agents(
        self,
        agents: list[PlayerAgent],
        competition_id: int,
        season_id: int,
    ) -> None:
        """Persist a list of PlayerAgent instances for a competition/season.

        Re-saving an agent with the same (player_id, competition_id, season_id)
        overwrites all columns.
        """
        rows = [
            (
                a.player_id, competition_id, season_id,
                a.player_name, a.position_name,
                a.role_family.value, a.broad_position.value,
                a.pace, a.passing, a.shooting, a.pressing,
                a.defending, a.physicality, a.consistency,
                a.tactical_adaptability, a.leadership, a.pressure_resistance,
                a.current_form, a.fatigue, a.tactical_understanding,
                a.manager_trust, a.bench_streak,
            )
            for a in agents
        ]
        self._conn.executemany(
            """
            INSERT INTO player_agents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(player_id, competition_id, season_id) DO UPDATE SET
                player_name=excluded.player_name,
                position_name=excluded.position_name,
                role_family=excluded.role_family,
                broad_position=excluded.broad_position,
                pace=excluded.pace, passing=excluded.passing,
                shooting=excluded.shooting, pressing=excluded.pressing,
                defending=excluded.defending, physicality=excluded.physicality,
                consistency=excluded.consistency,
                tactical_adaptability=excluded.tactical_adaptability,
                leadership=excluded.leadership,
                pressure_resistance=excluded.pressure_resistance,
                current_form=excluded.current_form, fatigue=excluded.fatigue,
                tactical_understanding=excluded.tactical_understanding,
                manager_trust=excluded.manager_trust,
                bench_streak=excluded.bench_streak
            """,
            rows,
        )
        self._conn.commit()

    def load_player_agents(
        self,
        competition_id: int,
        season_id: int,
    ) -> list[PlayerAgent]:
        """Load all PlayerAgent instances for a competition/season."""
        rows = self._conn.execute(
            "SELECT * FROM player_agents WHERE competition_id=? AND season_id=?",
            (competition_id, season_id),
        ).fetchall()
        return [
            PlayerAgent(
                player_id=int(r["player_id"]),
                player_name=str(r["player_name"]),
                position_name=str(r["position_name"]),
                role_family=RoleFamily(r["role_family"]),
                broad_position=BroadPosition(r["broad_position"]),
                pace=float(r["pace"]),
                passing=float(r["passing"]),
                shooting=float(r["shooting"]),
                pressing=float(r["pressing"]),
                defending=float(r["defending"]),
                physicality=float(r["physicality"]),
                consistency=float(r["consistency"]),
                tactical_adaptability=float(r["tactical_adaptability"]),
                leadership=float(r["leadership"]),
                pressure_resistance=float(r["pressure_resistance"]),
                current_form=float(r["current_form"]),
                fatigue=float(r["fatigue"]),
                tactical_understanding=float(r["tactical_understanding"]),
                manager_trust=float(r["manager_trust"]),
                bench_streak=int(r["bench_streak"]),
            )
            for r in rows
        ]

    # -----------------------------------------------------------------------
    # TeamBaseline
    # -----------------------------------------------------------------------

    def save_team_baseline(self, baseline: TeamBaseline) -> None:
        """Persist a TeamBaseline snapshot.

        One snapshot per (team_name, competition_id, season_id). Re-saving
        overwrites the previous snapshot.
        """
        self._conn.execute(
            """
            INSERT INTO team_baselines VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(team_name, competition_id, season_id) DO UPDATE SET
                played_match_ids=excluded.played_match_ids,
                xg_for_per90=excluded.xg_for_per90,
                xg_against_per90=excluded.xg_against_per90,
                ppda=excluded.ppda,
                progressive_passes_per90=excluded.progressive_passes_per90,
                possession_pct=excluded.possession_pct,
                league_position=excluded.league_position,
                points_to_safety=excluded.points_to_safety,
                points_to_title=excluded.points_to_title,
                matches_remaining=excluded.matches_remaining,
                cultural_inertia=excluded.cultural_inertia
            """,
            (
                baseline.team_name,
                baseline.competition_id,
                baseline.season_id,
                _encode_frozenset(baseline.played_match_ids),
                baseline.xg_for_per90,
                baseline.xg_against_per90,
                _encode_float(baseline.ppda),
                baseline.progressive_passes_per90,
                baseline.possession_pct,
                baseline.league_position,
                baseline.points_to_safety,
                baseline.points_to_title,
                baseline.matches_remaining,
                baseline.cultural_inertia,
            ),
        )
        self._conn.commit()

    def load_team_baseline(
        self,
        team_name: str,
        competition_id: int,
        season_id: int,
    ) -> TeamBaseline | None:
        """Load a TeamBaseline snapshot. Returns None if not found."""
        r = self._conn.execute(
            """
            SELECT * FROM team_baselines
            WHERE team_name=? AND competition_id=? AND season_id=?
            """,
            (team_name, competition_id, season_id),
        ).fetchone()
        if r is None:
            return None
        return TeamBaseline(
            team_name=str(r["team_name"]),
            competition_id=int(r["competition_id"]),
            season_id=int(r["season_id"]),
            played_match_ids=_decode_frozenset(r["played_match_ids"]),
            xg_for_per90=float(r["xg_for_per90"]),
            xg_against_per90=float(r["xg_against_per90"]),
            ppda=_decode_float(r["ppda"]),
            progressive_passes_per90=float(r["progressive_passes_per90"]),
            possession_pct=float(r["possession_pct"]),
            league_position=int(r["league_position"]),
            points_to_safety=int(r["points_to_safety"]),
            points_to_title=int(r["points_to_title"]),
            matches_remaining=int(r["matches_remaining"]),
            cultural_inertia=float(r["cultural_inertia"]),
        )

    # -----------------------------------------------------------------------
    # ManagerAgent
    # -----------------------------------------------------------------------

    def save_manager_agent(self, agent: ManagerAgent) -> None:
        """Persist a ManagerAgent snapshot.

        One snapshot per (manager_name, team_name, competition_id, season_id).
        Re-saving overwrites the previous snapshot.
        """
        self._conn.execute(
            """
            INSERT INTO manager_agents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(manager_name, team_name, competition_id, season_id) DO UPDATE SET
                tenure_match_ids=excluded.tenure_match_ids,
                pressing_intensity=excluded.pressing_intensity,
                possession_preference=excluded.possession_preference,
                counter_tendency=excluded.counter_tendency,
                preferred_formation=excluded.preferred_formation,
                implementation_speed=excluded.implementation_speed,
                youth_development=excluded.youth_development,
                style_stubbornness=excluded.style_stubbornness,
                job_security=excluded.job_security,
                squad_trust=excluded.squad_trust
            """,
            (
                agent.manager_name,
                agent.team_name,
                agent.competition_id,
                agent.season_id,
                _encode_frozenset(agent.tenure_match_ids),
                agent.pressing_intensity,
                agent.possession_preference,
                agent.counter_tendency,
                agent.preferred_formation,
                agent.implementation_speed,
                agent.youth_development,
                agent.style_stubbornness,
                agent.job_security,
                _encode_dict(agent.squad_trust),
            ),
        )
        self._conn.commit()

    def load_manager_agent(
        self,
        manager_name: str,
        team_name: str,
        competition_id: int,
        season_id: int,
    ) -> ManagerAgent | None:
        """Load a ManagerAgent snapshot. Returns None if not found."""
        r = self._conn.execute(
            """
            SELECT * FROM manager_agents
            WHERE manager_name=? AND team_name=? AND competition_id=? AND season_id=?
            """,
            (manager_name, team_name, competition_id, season_id),
        ).fetchone()
        if r is None:
            return None
        return ManagerAgent(
            manager_name=str(r["manager_name"]),
            team_name=str(r["team_name"]),
            competition_id=int(r["competition_id"]),
            season_id=int(r["season_id"]),
            tenure_match_ids=_decode_frozenset(r["tenure_match_ids"]),
            pressing_intensity=float(r["pressing_intensity"]),
            possession_preference=float(r["possession_preference"]),
            counter_tendency=float(r["counter_tendency"]),
            preferred_formation=r["preferred_formation"],
            implementation_speed=float(r["implementation_speed"]),
            youth_development=float(r["youth_development"]),
            style_stubbornness=float(r["style_stubbornness"]),
            job_security=float(r["job_security"]),
            squad_trust=_decode_dict(r["squad_trust"]),
        )

    # -----------------------------------------------------------------------
    # OpponentStrength
    # -----------------------------------------------------------------------

    def save_opponent_strengths(
        self,
        strengths: dict[str, OpponentStrength],
        competition_id: int,
        season_id: int,
        trigger_week: int,
    ) -> None:
        """Persist a dict of OpponentStrength snapshots for a trigger point."""
        rows = [
            (
                s.opponent_name, competition_id, season_id, trigger_week,
                s.xg_for_per90, s.xg_against_per90, s.elo_rating,
            )
            for s in strengths.values()
        ]
        self._conn.executemany(
            """
            INSERT INTO opponent_strengths VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(opponent_name, competition_id, season_id, trigger_week) DO UPDATE SET
                xg_for_per90=excluded.xg_for_per90,
                xg_against_per90=excluded.xg_against_per90,
                elo_rating=excluded.elo_rating
            """,
            rows,
        )
        self._conn.commit()

    def load_opponent_strengths(
        self,
        competition_id: int,
        season_id: int,
        trigger_week: int,
    ) -> dict[str, OpponentStrength]:
        """Load all OpponentStrength snapshots for a trigger point."""
        rows = self._conn.execute(
            """
            SELECT * FROM opponent_strengths
            WHERE competition_id=? AND season_id=? AND trigger_week=?
            """,
            (competition_id, season_id, trigger_week),
        ).fetchall()
        return {
            str(r["opponent_name"]): OpponentStrength(
                opponent_name=str(r["opponent_name"]),
                xg_for_per90=float(r["xg_for_per90"]),
                xg_against_per90=float(r["xg_against_per90"]),
                elo_rating=float(r["elo_rating"]),
            )
            for r in rows
        }

    # -----------------------------------------------------------------------
    # FixtureList
    # -----------------------------------------------------------------------

    def save_fixture_list(
        self,
        fixture_list: FixtureList,
        competition_id: int,
        season_id: int,
    ) -> None:
        """Persist a FixtureList.

        Two-table write (fixture_lists header + fixtures rows) is performed
        in a single transaction so a mid-write failure never leaves an
        inconsistent state. The header row is what distinguishes a saved
        empty FixtureList from a missing one on load.

        The ordinal column records each Fixture's position in the tuple
        (0-indexed) to guarantee exact round-trip ordering on load.
        Re-saving deletes existing fixture rows before reinserting.
        """
        key = (
            fixture_list.team_name,
            competition_id,
            season_id,
            fixture_list.trigger_week,
        )
        rows = [
            (
                fixture_list.team_name,
                competition_id,
                season_id,
                fixture_list.trigger_week,
                ordinal,
                f.match_week,
                f.opponent_name,
                int(f.is_home),
            )
            for ordinal, f in enumerate(fixture_list.fixtures)
        ]
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO fixture_lists VALUES (?,?,?,?)
                ON CONFLICT(team_name, competition_id, season_id, trigger_week)
                DO UPDATE SET trigger_week=excluded.trigger_week
                """,
                key,
            )
            self._conn.execute(
                "DELETE FROM fixtures WHERE team_name=? AND competition_id=? AND season_id=? AND trigger_week=?",
                key,
            )
            self._conn.executemany(
                "INSERT INTO fixtures VALUES (?,?,?,?,?,?,?,?)",
                rows,
            )

    def load_fixture_list(
        self,
        team_name: str,
        competition_id: int,
        season_id: int,
        trigger_week: int,
    ) -> FixtureList | None:
        """Load a FixtureList. Returns None if not found.

        The fixture_lists header is checked first to distinguish a saved
        empty FixtureList from a missing one. Fixtures are returned in
        ordinal order, preserving the original tuple sequence.
        """
        header = self._conn.execute(
            """
            SELECT 1 FROM fixture_lists
            WHERE team_name=? AND competition_id=? AND season_id=? AND trigger_week=?
            """,
            (team_name, competition_id, season_id, trigger_week),
        ).fetchone()
        if header is None:
            return None
        rows = self._conn.execute(
            """
            SELECT * FROM fixtures
            WHERE team_name=? AND competition_id=? AND season_id=? AND trigger_week=?
            ORDER BY ordinal
            """,
            (team_name, competition_id, season_id, trigger_week),
        ).fetchall()
        fixtures = tuple(
            Fixture(
                match_week=int(r["match_week"]),
                opponent_name=str(r["opponent_name"]),
                is_home=bool(r["is_home"]),
            )
            for r in rows
        )
        return FixtureList(
            team_name=team_name,
            trigger_week=trigger_week,
            fixtures=fixtures,
        )

    # -----------------------------------------------------------------------
    # LeagueContext
    # -----------------------------------------------------------------------

    def save_league_context(self, context: LeagueContext) -> None:
        """Persist a LeagueContext snapshot.

        One snapshot per (competition_id, season_id). Re-saving overwrites.
        """
        self._conn.execute(
            """
            INSERT INTO league_contexts VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT(competition_id, season_id) DO UPDATE SET
                name=excluded.name,
                avg_ppda=excluded.avg_ppda,
                avg_progressive_passes_per90=excluded.avg_progressive_passes_per90,
                avg_xg_per90=excluded.avg_xg_per90,
                pressing_level=excluded.pressing_level,
                physicality_level=excluded.physicality_level,
                tactical_complexity=excluded.tactical_complexity
            """,
            (
                context.competition_id,
                context.season_id,
                context.name,
                context.avg_ppda,
                context.avg_progressive_passes_per90,
                context.avg_xg_per90,
                context.pressing_level,
                context.physicality_level,
                context.tactical_complexity,
            ),
        )
        self._conn.commit()

    def load_league_context(
        self,
        competition_id: int,
        season_id: int,
    ) -> LeagueContext | None:
        """Load a LeagueContext. Returns None if not found."""
        r = self._conn.execute(
            """
            SELECT * FROM league_contexts
            WHERE competition_id=? AND season_id=?
            """,
            (competition_id, season_id),
        ).fetchone()
        if r is None:
            return None
        return LeagueContext(
            competition_id=int(r["competition_id"]),
            season_id=int(r["season_id"]),
            name=str(r["name"]),
            avg_ppda=float(r["avg_ppda"]),
            avg_progressive_passes_per90=float(r["avg_progressive_passes_per90"]),
            avg_xg_per90=float(r["avg_xg_per90"]),
            pressing_level=r["pressing_level"],
            physicality_level=r["physicality_level"],
            tactical_complexity=r["tactical_complexity"],
        )

    # -----------------------------------------------------------------------
    # CascadeEvent
    # -----------------------------------------------------------------------

    def save_cascade_events(
        self,
        comparison_key: str,
        run_id: str,
        events: list[CascadeEvent],
    ) -> None:
        """Persist a list of CascadeEvent for a specific run.

        Events are stored with an ordinal to preserve ordering.
        Re-saving deletes existing events for the same key+run_id.

        Args:
            comparison_key: Identifies the comparison this run belongs to.
            run_id:         Identifies the specific run. Recommended format:
                            "{branch}_{run_index}" where branch is "a" (no
                            change) or "b" (with trigger), and run_index is
                            0-based. Examples: "a_0", "b_3".
            events:         Ordered list of cascade events.
        """
        pk = (comparison_key, run_id)
        rows = [
            (
                comparison_key,
                run_id,
                ordinal,
                e.week,
                e.event_type,
                e.affected_agent,
                json.dumps(e.cause_chain),
                e.magnitude,
                e.depth,
            )
            for ordinal, e in enumerate(events)
        ]
        with self._conn:
            self._conn.execute(
                "DELETE FROM cascade_events WHERE comparison_key=? AND run_id=?",
                pk,
            )
            self._conn.executemany(
                "INSERT INTO cascade_events VALUES (?,?,?,?,?,?,?,?,?)",
                rows,
            )

    def load_cascade_events(
        self,
        comparison_key: str,
        run_id: str,
    ) -> list[CascadeEvent]:
        """Load CascadeEvent list for a run. Returns empty list if not found."""
        rows = self._conn.execute(
            """
            SELECT * FROM cascade_events
            WHERE comparison_key=? AND run_id=?
            ORDER BY ordinal
            """,
            (comparison_key, run_id),
        ).fetchall()
        return [
            CascadeEvent(
                week=int(r["week"]),
                event_type=str(r["event_type"]),
                affected_agent=str(r["affected_agent"]),
                cause_chain=tuple(json.loads(r["cause_chain"])),
                magnitude=float(r["magnitude"]),
                depth=int(r["depth"]),
            )
            for r in rows
        ]

    # -----------------------------------------------------------------------
    # ComparisonResult
    # -----------------------------------------------------------------------

    def save_comparison_result(
        self,
        comparison_key: str,
        result: ComparisonResult,
        *,
        rng_seed: int,
        trigger_summary: str,
    ) -> None:
        """Persist a ComparisonResult's aggregated statistics with metadata.

        run_results are excluded from persistence (too large). On load,
        run_results is restored as an empty tuple. Full per-run data can
        be reproduced by re-running with the same seed.

        Args:
            comparison_key:  Unique identifier for this comparison.
            result:          The ComparisonResult to persist.
            rng_seed:        Random seed used for reproducibility.
            trigger_summary: Human-readable trigger description (display
                             label only; not for trigger reconstruction).
        """
        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        n_runs = result.no_change.n_runs

        self._conn.execute(
            """
            INSERT INTO comparison_results VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(comparison_key) DO UPDATE SET
                no_change_json=excluded.no_change_json,
                with_change_json=excluded.with_change_json,
                delta_json=excluded.delta_json,
                rng_seed=excluded.rng_seed,
                n_runs=excluded.n_runs,
                trigger_summary=excluded.trigger_summary,
                created_at=excluded.created_at
            """,
            (
                comparison_key,
                _encode_aggregated(result.no_change),
                _encode_aggregated(result.with_change),
                _encode_delta(result.delta),
                rng_seed,
                n_runs,
                trigger_summary,
                created_at,
            ),
        )
        self._conn.commit()

    def load_comparison_result(
        self,
        comparison_key: str,
    ) -> ComparisonResultWithMeta | None:
        """Load a ComparisonResult with metadata. Returns None if not found.

        run_results on both AggregatedResult instances are empty tuples
        since per-run SimulationResult data is not persisted.

        Returns ComparisonResultWithMeta where meta is None for legacy
        rows that lack metadata columns.
        """
        r = self._conn.execute(
            """
            SELECT * FROM comparison_results WHERE comparison_key=?
            """,
            (comparison_key,),
        ).fetchone()
        if r is None:
            return None

        result = ComparisonResult(
            no_change=_decode_aggregated(r["no_change_json"]),
            with_change=_decode_aggregated(r["with_change_json"]),
            delta=_decode_delta(r["delta_json"]),
        )

        # Build metadata (None for legacy rows without metadata).
        if r["rng_seed"] is not None and r["created_at"] is not None:
            meta = ComparisonMeta(
                rng_seed=int(r["rng_seed"]),
                n_runs=int(r["n_runs"]),
                trigger_summary=str(r["trigger_summary"] or ""),
                created_at=str(r["created_at"]),
            )
        else:
            meta = None

        return ComparisonResultWithMeta(result=result, meta=meta)
