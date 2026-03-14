"""Branch A/B parallel comparison.

Runs the same initial state N times with and without a trigger, then
aggregates and compares the results.

Seed and reproducibility:
    A base rng is created from rng_seed. For each run i, base_rng.spawn(2)
    produces two independent child generators: one for Branch A (no change)
    and one for Branch B (with trigger). The child generators have
    independent, non-overlapping random streams — they do NOT share the
    same sequence. This means match-by-match Poisson draws differ between
    branches even at the same fixture. The comparison captures the
    aggregate statistical effect of the trigger across N runs, not a
    paired match-level difference. All results are fully reproducible
    from the same rng_seed.

Delta computation:
    DeltaMetrics uses the union of all event types observed across both
    branches. Event types present in only one branch get 0.0 for the
    other, ensuring no key is silently dropped.
"""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass

import numpy as np

from iffootball.agents.fixture import FixtureList, OpponentStrength
from iffootball.agents.manager import ManagerAgent
from iffootball.agents.player import PlayerAgent
from iffootball.agents.team import TeamBaseline
from iffootball.agents.trigger import ChangeTrigger
from iffootball.config import SimulationRules
from iffootball.simulation.engine import Simulation, SimulationResult
from iffootball.simulation.turning_point import TurningPointHandler


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AggregatedResult:
    """Statistics from N simulation runs of a single branch.

    Attributes:
        n_runs:               Number of runs executed.
        total_points_mean:    Mean total points across runs.
        total_points_median:  Median total points across runs.
        total_points_std:     Standard deviation of total points.
        cascade_event_counts: Mean frequency of each event_type across
                              runs (event_type -> mean count per run).
        run_results:          Individual SimulationResult per run,
                              retained for deeper analysis (e.g. M3
                              radar charts).
    """

    n_runs: int
    total_points_mean: float
    total_points_median: float
    total_points_std: float
    cascade_event_counts: dict[str, float]
    run_results: tuple[SimulationResult, ...]


@dataclass(frozen=True)
class DeltaMetrics:
    """Difference between Branch B (with_change) and Branch A (no_change).

    All values are B - A. Positive means the trigger increased the metric.

    Attributes:
        points_mean_diff:    Difference in mean total points.
        points_median_diff:  Difference in median total points.
        cascade_count_diff:  Difference in mean cascade event frequency
                             per event_type. Computed over the union of
                             all event types from both branches.
    """

    points_mean_diff: float
    points_median_diff: float
    cascade_count_diff: dict[str, float]


@dataclass(frozen=True)
class ComparisonResult:
    """Complete A/B comparison result.

    Attributes:
        no_change:   Branch A aggregated results (baseline, no trigger).
        with_change: Branch B aggregated results (trigger applied).
        delta:       B - A difference metrics.
    """

    no_change: AggregatedResult
    with_change: AggregatedResult
    delta: DeltaMetrics


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _aggregate(results: list[SimulationResult]) -> AggregatedResult:
    """Aggregate N SimulationResult instances into summary statistics."""
    n = len(results)
    total_points = [
        sum(mr.points_earned for mr in r.match_results) for r in results
    ]
    points_arr = np.array(total_points, dtype=float)

    # Cascade event frequency: count per event_type per run, then average.
    type_counts_per_run: list[Counter[str]] = []
    for r in results:
        counter: Counter[str] = Counter()
        for ce in r.cascade_events:
            counter[ce.event_type] += 1
        type_counts_per_run.append(counter)

    all_types: set[str] = set()
    for c in type_counts_per_run:
        all_types.update(c.keys())

    cascade_means: dict[str, float] = {}
    for et in sorted(all_types):
        cascade_means[et] = sum(c[et] for c in type_counts_per_run) / n

    return AggregatedResult(
        n_runs=n,
        total_points_mean=float(np.mean(points_arr)),
        total_points_median=float(np.median(points_arr)),
        total_points_std=float(np.std(points_arr, ddof=0)),
        cascade_event_counts=cascade_means,
        run_results=tuple(results),
    )


def _calc_delta(a: AggregatedResult, b: AggregatedResult) -> DeltaMetrics:
    """Compute B - A difference metrics.

    cascade_count_diff uses the union of event types from both branches.
    Missing types default to 0.0.
    """
    all_types = set(a.cascade_event_counts) | set(b.cascade_event_counts)
    cascade_diff: dict[str, float] = {}
    for et in sorted(all_types):
        val_a = a.cascade_event_counts.get(et, 0.0)
        val_b = b.cascade_event_counts.get(et, 0.0)
        cascade_diff[et] = val_b - val_a

    return DeltaMetrics(
        points_mean_diff=b.total_points_mean - a.total_points_mean,
        points_median_diff=b.total_points_median - a.total_points_median,
        cascade_count_diff=cascade_diff,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_comparison(
    team: TeamBaseline,
    squad: list[PlayerAgent],
    manager: ManagerAgent,
    fixture_list: FixtureList,
    opponent_strengths: dict[str, OpponentStrength],
    rules: SimulationRules,
    handler: TurningPointHandler,
    trigger: ChangeTrigger,
    n_runs: int,
    rng_seed: int,
) -> ComparisonResult:
    """Run N parallel A/B comparisons and return aggregated results.

    For each run, two independent child generators are spawned from
    base_rng. Branch A runs without the trigger; Branch B runs with it.
    The child generators produce independent (not identical) random
    streams, so match-level outcomes differ between branches. The
    comparison captures the aggregate statistical effect of the trigger
    across N runs.

    Args:
        team:               Team baseline (read-only, shared).
        squad:              Initial squad state (deepcopied per run).
        manager:            Initial manager state (deepcopied per run).
        fixture_list:       Fixtures to simulate (read-only, shared).
        opponent_strengths: Opponent data (read-only, shared).
        rules:              Simulation rules config.
        handler:            Turning point handler.
        trigger:            Change trigger to apply in Branch B.
        n_runs:             Number of simulation runs per branch.
        rng_seed:           Base seed for reproducibility.

    Returns:
        ComparisonResult with aggregated A/B results and delta.
    """
    base_rng = np.random.default_rng(rng_seed)

    results_a: list[SimulationResult] = []
    results_b: list[SimulationResult] = []

    for _ in range(n_runs):
        child_a, child_b = base_rng.spawn(2)

        # Branch A: no trigger
        sim_a = Simulation(
            team=team,
            squad=copy.deepcopy(squad),
            manager=copy.deepcopy(manager),
            fixture_list=fixture_list,
            opponent_strengths=opponent_strengths,
            rules=rules,
            handler=handler,
            rng=np.random.default_rng(child_a),
        )
        results_a.append(sim_a.run())

        # Branch B: with trigger
        sim_b = Simulation(
            team=team,
            squad=copy.deepcopy(squad),
            manager=copy.deepcopy(manager),
            fixture_list=fixture_list,
            opponent_strengths=opponent_strengths,
            rules=rules,
            handler=handler,
            rng=np.random.default_rng(child_b),
        )
        sim_b.apply_trigger(trigger)
        results_b.append(sim_b.run())

    agg_a = _aggregate(results_a)
    agg_b = _aggregate(results_b)
    delta = _calc_delta(agg_a, agg_b)

    return ComparisonResult(
        no_change=agg_a,
        with_change=agg_b,
        delta=delta,
    )
