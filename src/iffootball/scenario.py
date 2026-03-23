"""Season-scenario definition and execution.

Connects season cache DB to the comparison engine. A ScenarioDefinition
describes a what-if scenario (manager change, player add, player remove)
and run_scenario() executes it against the season cache.

Usage:
    db = Database("data/season_cache/premier_league_2015-16.db")
    rules = SimulationRules.load("config/simulation_rules")

    scenario = ScenarioDefinition(
        team_name="Chelsea",
        competition_id=2,
        season_id=27,
        scenario_type="manager_change",
        alt_manager_name="Jürgen Klopp",
    )
    result = run_scenario(scenario, db, rules, n_runs=20, rng_seed=42)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal

import numpy as np

from iffootball.agents.fixture import OpponentStrength
from iffootball.agents.manager import ManagerAgent
from iffootball.agents.player import PlayerAgent
from iffootball.agents.team import TeamBaseline
from iffootball.agents.trigger import ManagerChangeTrigger, TransferInTrigger
from iffootball.config import SimulationRules
from iffootball.simulation.comparison import ComparisonResult, run_comparison
from iffootball.simulation.engine import Simulation, SimulationResult
from iffootball.simulation.turning_point import RuleBasedHandler
from iffootball.storage.db import Database


# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioDefinition:
    """A season-level what-if scenario.

    Describes a single change to a team's season baseline. The baseline
    manager is not stored here — it is resolved at runtime from the
    manager_agents table (season-start manager for the team).

    Attributes:
        team_name:             Target team (StatsBomb spelling).
        competition_id:        StatsBomb competition ID.
        season_id:             StatsBomb season ID.
        scenario_type:         One of "manager_change", "player_add", "player_remove".
        alt_manager_name:      Incoming manager name (manager_change only).
        alt_manager_team_name: Team the incoming manager is sourced from.
                               Required for unique profile resolution when
                               multiple managers share the same name.
        player_id:             Target player ID (player_add / player_remove).
        player_name:           Target player display name (player_add / player_remove).
        expected_role:         Role for incoming player (player_add only).
    """

    team_name: str
    competition_id: int
    season_id: int
    scenario_type: Literal["manager_change", "player_add", "player_remove"]
    alt_manager_name: str | None = None
    alt_manager_team_name: str | None = None
    player_id: int | None = None
    player_name: str | None = None
    expected_role: str | None = None  # "starter" / "rotation" / "squad"

    def __post_init__(self) -> None:
        if self.scenario_type == "manager_change":
            if not self.alt_manager_name:
                raise ValueError(
                    "manager_change scenario requires alt_manager_name"
                )
        elif self.scenario_type in ("player_add", "player_remove"):
            if self.player_id is None:
                raise ValueError(
                    f"{self.scenario_type} scenario requires player_id"
                )
        else:
            raise ValueError(f"Unknown scenario_type: {self.scenario_type}")

    @property
    def scenario_key(self) -> str:
        """Generate a comparison key for this scenario."""
        safe_team = self.team_name.replace(" ", "_").lower()
        base = f"{safe_team}_{self.competition_id}-{self.season_id}"
        if self.scenario_type == "manager_change":
            safe_mgr = (self.alt_manager_name or "").replace(" ", "_").lower()
            return f"{base}_mgr_{safe_mgr}"
        elif self.scenario_type == "player_add":
            return f"{base}_add_{self.player_id}"
        else:
            return f"{base}_remove_{self.player_id}"


# ---------------------------------------------------------------------------
# Season cache loader
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TeamSeasonState:
    """Baseline state for a team loaded from season cache.

    All fields needed to run a comparison.
    """

    squad: list[PlayerAgent]
    baseline: TeamBaseline
    manager: ManagerAgent
    fixture_list: object  # FixtureList (avoiding circular for now)
    opponent_strengths: dict[str, OpponentStrength]


def load_team_season_state(
    db: Database,
    team_name: str,
    competition_id: int,
    season_id: int,
) -> TeamSeasonState:
    """Load a team's baseline state from the season cache DB.

    Resolves the baseline manager as the season-start manager
    (first match of the season).

    Raises:
        ValueError: if required data is missing from the DB.
    """
    squad = db.load_player_agents(competition_id, season_id, team_name=team_name)
    if not squad:
        raise ValueError(
            f"No player agents found for {team_name} "
            f"(comp={competition_id}, season={season_id})"
        )

    baseline = db.load_team_baseline(team_name, competition_id, season_id)
    if baseline is None:
        raise ValueError(f"No team baseline found for {team_name}")

    fixture_list = db.load_fixture_list(team_name, competition_id, season_id)
    if fixture_list is None:
        raise ValueError(f"No fixture list found for {team_name}")

    opponent_strengths = db.load_opponent_strengths(competition_id, season_id)
    if not opponent_strengths:
        raise ValueError("No opponent strengths found")

    # Resolve baseline manager (season-start).
    # The season-start manager is persisted in db_meta during
    # initialize_season() as "season_start_manager:{team_name}".
    conn = db._conn  # noqa: SLF001
    meta_row = conn.execute(
        "SELECT value FROM db_meta WHERE key=?",
        (f"season_start_manager:{team_name}",),
    ).fetchone()
    if meta_row is None:
        raise ValueError(
            f"No season-start manager recorded for {team_name}. "
            f"Rebuild the season cache."
        )
    best_name = str(meta_row["value"])

    manager = db.load_manager_agent(
        best_name, team_name, competition_id, season_id
    )
    if manager is None:
        raise ValueError(f"Manager {best_name} not found for {team_name}")

    return TeamSeasonState(
        squad=squad,
        baseline=baseline,
        manager=manager,
        fixture_list=fixture_list,
        opponent_strengths=opponent_strengths,
    )


# ---------------------------------------------------------------------------
# Scenario execution
# ---------------------------------------------------------------------------


def _build_manager_change_trigger(
    scenario: ScenarioDefinition,
    db: Database,
    baseline_manager_name: str,
) -> ManagerChangeTrigger:
    """Build a ManagerChangeTrigger from a manager_change scenario."""
    assert scenario.alt_manager_name is not None

    # Resolve incoming manager profile from DB (candidate master).
    # Use alt_manager_team_name for unique resolution when available.
    incoming_profile: ManagerAgent | None = None
    if scenario.alt_manager_team_name:
        incoming_profile = db.load_manager_agent(
            scenario.alt_manager_name,
            scenario.alt_manager_team_name,
            scenario.competition_id,
            scenario.season_id,
        )
    else:
        # Fallback: search by name only (first match).
        conn = db._conn  # noqa: SLF001
        rows = conn.execute(
            "SELECT manager_name, team_name FROM manager_agents "
            "WHERE manager_name=? AND competition_id=? AND season_id=?",
            (scenario.alt_manager_name, scenario.competition_id, scenario.season_id),
        ).fetchall()
        if rows:
            incoming_profile = db.load_manager_agent(
                str(rows[0]["manager_name"]),
                str(rows[0]["team_name"]),
                scenario.competition_id,
                scenario.season_id,
            )

    return ManagerChangeTrigger(
        outgoing_manager_name=baseline_manager_name,
        incoming_manager_name=scenario.alt_manager_name,
        transition_type="pre_season",
        applied_at=0,
        incoming_profile=incoming_profile,
    )


def _build_player_add_trigger(
    scenario: ScenarioDefinition,
    db: Database,
) -> TransferInTrigger:
    """Build a TransferInTrigger from a player_add scenario."""
    assert scenario.player_id is not None

    # Find the player in another team's roster.
    all_players = db.load_player_agents(
        scenario.competition_id, scenario.season_id
    )
    player = next(
        (p for p in all_players if p.player_id == scenario.player_id),
        None,
    )
    if player is None:
        raise ValueError(
            f"Player ID {scenario.player_id} not found in season cache"
        )
    if player.team_name == scenario.team_name:
        raise ValueError(
            f"Player {player.player_name} (ID {scenario.player_id}) "
            f"already belongs to {scenario.team_name}. "
            f"player_add requires a player from another team."
        )

    return TransferInTrigger(
        player_name=scenario.player_name or player.player_name,
        expected_role=scenario.expected_role or "starter",
        applied_at=0,
        player=player,
    )


def run_scenario(
    scenario: ScenarioDefinition,
    db: Database,
    rules: SimulationRules,
    n_runs: int = 20,
    rng_seed: int = 42,
) -> ComparisonResult:
    """Execute a season-scenario comparison.

    Loads the team's baseline state from the season cache DB, constructs
    the appropriate trigger or squad modification, and runs a paired A/B
    comparison.

    For manager_change and player_add, Branch A is baseline and Branch B
    applies a trigger at week 0 (pre-season).

    For player_remove, Branch A is baseline (full squad) and Branch B
    runs with the player excluded from the starting squad.

    Args:
        scenario: The scenario to execute.
        db:       Season cache database.
        rules:    Simulation rules config.
        n_runs:   Number of paired runs.
        rng_seed: Base seed for reproducibility.

    Returns:
        ComparisonResult with A/B results and delta.
    """
    from iffootball.agents.fixture import FixtureList

    state = load_team_season_state(
        db, scenario.team_name, scenario.competition_id, scenario.season_id
    )
    fixture_list: FixtureList = state.fixture_list  # type: ignore[assignment]
    handler = RuleBasedHandler(rules)

    if scenario.scenario_type == "manager_change":
        trigger = _build_manager_change_trigger(
            scenario, db, state.manager.manager_name
        )
        return run_comparison(
            team=state.baseline,
            squad=state.squad,
            manager=state.manager,
            fixture_list=fixture_list,
            opponent_strengths=state.opponent_strengths,
            rules=rules,
            handler=handler,
            trigger=trigger,
            n_runs=n_runs,
            rng_seed=rng_seed,
        )

    elif scenario.scenario_type == "player_add":
        add_trigger = _build_player_add_trigger(scenario, db)
        return run_comparison(
            team=state.baseline,
            squad=state.squad,
            manager=state.manager,
            fixture_list=fixture_list,
            opponent_strengths=state.opponent_strengths,
            rules=rules,
            handler=handler,
            trigger=add_trigger,
            n_runs=n_runs,
            rng_seed=rng_seed,
        )

    else:
        # player_remove: Branch A = full squad, Branch B = squad minus player.
        assert scenario.player_id is not None
        squad_b = [
            p for p in state.squad if p.player_id != scenario.player_id
        ]
        if len(squad_b) == len(state.squad):
            raise ValueError(
                f"Player ID {scenario.player_id} not in "
                f"{scenario.team_name} squad"
            )

        # Run paired comparison manually (no trigger, different squads).
        base_ss = np.random.SeedSequence(rng_seed)
        results_a: list[SimulationResult] = []
        results_b: list[SimulationResult] = []

        for run_ss in base_ss.spawn(n_runs):
            match_ss, action_ss_a, action_ss_b = run_ss.spawn(3)

            sim_a = Simulation(
                team=state.baseline,
                squad=copy.deepcopy(state.squad),
                manager=copy.deepcopy(state.manager),
                fixture_list=fixture_list,
                opponent_strengths=state.opponent_strengths,
                rules=rules,
                handler=handler,
                match_rng=np.random.default_rng(match_ss),
                action_rng=np.random.default_rng(action_ss_a),
            )
            results_a.append(sim_a.run())

            sim_b = Simulation(
                team=state.baseline,
                squad=copy.deepcopy(squad_b),
                manager=copy.deepcopy(state.manager),
                fixture_list=fixture_list,
                opponent_strengths=state.opponent_strengths,
                rules=rules,
                handler=handler,
                match_rng=np.random.default_rng(match_ss),
                action_rng=np.random.default_rng(action_ss_b),
            )
            results_b.append(sim_b.run())

        from iffootball.simulation.comparison import (
            _aggregate,
            _calc_delta,
        )

        agg_a = _aggregate(results_a)
        agg_b = _aggregate(results_b)
        delta = _calc_delta(agg_a, agg_b)

        return ComparisonResult(
            no_change=agg_a,
            with_change=agg_b,
            delta=delta,
        )
