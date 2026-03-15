"""Radar chart data extraction and normalization.

Transforms ComparisonResult + context into normalized radar chart axes
for Branch A/B overlay visualization.

Axes:
    xG/90              — Simulation output: mean expected_goals_for across
                         N runs (affected by agent state factor).
    xGA/90 (baseline)  — Fixed baseline: mean expected_goals_against. Current
                         model does not simulate defensive manager impact.
    PPDA (est.)        — Tactical estimate from manager pressing_intensity.
    Possession (est.)  — Tactical estimate from manager possession_preference.
    Prog Passes (est.) — Tactical estimate from manager profile + formation.

Normalization:
    All axes are normalized to 0.0–1.0 where higher = better for the team.
    PPDA and xGA/90 are inverted (lower raw = better → higher normalized).
    League average maps to approximately 0.5 on each axis.
"""

from __future__ import annotations

from dataclasses import dataclass

from iffootball.agents.league import LeagueContext
from iffootball.agents.manager import ManagerAgent
from iffootball.agents.team import TeamBaseline
from iffootball.simulation.comparison import ComparisonResult
from iffootball.visualization.tactical_estimate import (
    estimate_possession,
    estimate_ppda,
    estimate_progressive_passes,
)

# Axis labels in display order.
AXIS_LABELS: tuple[str, ...] = (
    "xG/90",
    "xGA/90 (baseline)",
    "PPDA (est.)",
    "Possession (est.)",
    "Prog Passes (est.)",
)

# Default normalization ranges centered on typical league values.
# Each tuple is (min_val, max_val). Values outside are clipped.
_DEFAULT_RANGES: dict[str, tuple[float, float]] = {
    "xg_for": (0.5, 3.0),
    "xg_against": (0.5, 3.0),
    "ppda": (5.0, 20.0),
    "possession": (0.30, 0.70),
    "prog_passes": (20.0, 80.0),
}

# Axes where lower raw value = better for team (normalized as inverted).
_INVERTED_AXES = {"xg_against", "ppda"}

# League average possession (approximation when not available).
_LEAGUE_AVG_POSSESSION = 0.5


@dataclass(frozen=True)
class RadarAxes:
    """Normalized radar chart values for a single branch.

    All values are 0.0–1.0. Higher = better for the team.
    """

    xg_for: float
    xg_against: float
    ppda: float
    possession: float
    prog_passes: float

    def values(self) -> tuple[float, ...]:
        """Return axis values in AXIS_LABELS order."""
        return (
            self.xg_for,
            self.xg_against,
            self.ppda,
            self.possession,
            self.prog_passes,
        )


@dataclass(frozen=True)
class RadarChartData:
    """Complete radar chart data for A/B comparison.

    Attributes:
        branch_a: Normalized axes for Branch A (no change).
        branch_b: Normalized axes for Branch B (with change).
        labels:   Axis labels in display order.
    """

    branch_a: RadarAxes
    branch_b: RadarAxes
    labels: tuple[str, ...] = AXIS_LABELS


def build_normalization_ranges(
    league: LeagueContext,
) -> dict[str, tuple[float, float]]:
    """Build normalization ranges centered on league averages.

    Uses league averages as the midpoint and applies a symmetric spread.
    Falls back to defaults when league data is unavailable (0.0).
    """
    ranges = dict(_DEFAULT_RANGES)

    if league.avg_xg_per90 > 0.0:
        center = league.avg_xg_per90
        ranges["xg_for"] = (max(0.1, center - 1.0), center + 1.0)
        ranges["xg_against"] = (max(0.1, center - 1.0), center + 1.0)

    if league.avg_ppda > 0.0:
        center = league.avg_ppda
        ranges["ppda"] = (max(1.0, center - 5.0), center + 5.0)

    if league.avg_progressive_passes_per90 > 0.0:
        center = league.avg_progressive_passes_per90
        ranges["prog_passes"] = (max(0.0, center - 20.0), center + 20.0)

    return ranges


def _normalize(
    value: float,
    min_val: float,
    max_val: float,
    *,
    invert: bool = False,
) -> float:
    """Normalize a value to 0.0–1.0, clipping to [min_val, max_val].

    Args:
        value:   Raw metric value.
        min_val: Lower bound of the normalization range.
        max_val: Upper bound of the normalization range.
        invert:  If True, lower raw values map to higher normalized values.
    """
    span = max_val - min_val
    if span <= 0.0:
        return 0.5

    clipped = max(min_val, min(max_val, value))
    normalized = (clipped - min_val) / span

    if invert:
        normalized = 1.0 - normalized

    return normalized


def _mean_expected_goals(
    comparison: ComparisonResult,
    *,
    branch: str,
    metric: str,
) -> float:
    """Compute mean expected goals across all runs and matches.

    Args:
        comparison: A/B comparison result.
        branch:     "no_change" or "with_change".
        metric:     "expected_goals_for" or "expected_goals_against".
    """
    agg = getattr(comparison, branch)
    total = 0.0
    count = 0
    for run_result in agg.run_results:
        for mr in run_result.match_results:
            total += getattr(mr, metric)
            count += 1
    return total / count if count > 0 else 0.0


def extract_radar_data(
    comparison: ComparisonResult,
    baseline: TeamBaseline,
    incoming_manager: ManagerAgent | None,
    league: LeagueContext,
) -> RadarChartData:
    """Extract and normalize radar chart data from comparison results.

    Branch A uses pre-trigger baseline values directly for tactical axes.
    Branch B uses tactical estimates derived from the incoming manager's
    profile (or neutral defaults when incoming_manager is None).

    Args:
        comparison:        A/B comparison result with run_results retained.
        baseline:          Pre-trigger team baseline.
        incoming_manager:  Manager for Branch B. None uses neutral defaults.
        league:            League context for normalization and estimates.

    Returns:
        RadarChartData with normalized Branch A/B axes.
    """
    ranges = build_normalization_ranges(league)

    # --- xG/90 (simulation output) ---
    xg_for_a = _mean_expected_goals(
        comparison, branch="no_change", metric="expected_goals_for"
    )
    xg_for_b = _mean_expected_goals(
        comparison, branch="with_change", metric="expected_goals_for"
    )

    # --- xGA/90 (baseline fixed — current model does not simulate
    #     defensive impact of manager changes) ---
    xg_against_fixed = baseline.xg_against_per90

    # --- Tactical estimates ---
    # Branch A: use pre-trigger baseline values directly (no estimation).
    ppda_a = baseline.ppda
    poss_a = baseline.possession_pct
    prog_a = baseline.progressive_passes_per90

    # Branch B: incoming manager (or neutral default).
    if incoming_manager is not None:
        b_manager = incoming_manager
    else:
        b_manager = _neutral_manager_stub()

    ppda_b = estimate_ppda(baseline, b_manager, league, is_new_manager=True)
    poss_b = estimate_possession(
        baseline, b_manager, _LEAGUE_AVG_POSSESSION, is_new_manager=True
    )
    prog_b = estimate_progressive_passes(
        baseline, b_manager, league, is_new_manager=True
    )

    # --- Normalize ---
    def norm(key: str, val: float) -> float:
        lo, hi = ranges[key]
        return _normalize(val, lo, hi, invert=(key in _INVERTED_AXES))

    branch_a = RadarAxes(
        xg_for=norm("xg_for", xg_for_a),
        xg_against=norm("xg_against", xg_against_fixed),
        ppda=norm("ppda", ppda_a),
        possession=norm("possession", poss_a),
        prog_passes=norm("prog_passes", prog_a),
    )
    branch_b = RadarAxes(
        xg_for=norm("xg_for", xg_for_b),
        xg_against=norm("xg_against", xg_against_fixed),
        ppda=norm("ppda", ppda_b),
        possession=norm("possession", poss_b),
        prog_passes=norm("prog_passes", prog_b),
    )

    return RadarChartData(branch_a=branch_a, branch_b=branch_b)


def _neutral_manager_stub() -> ManagerAgent:
    """Create a minimal ManagerAgent with neutral defaults.

    Used when incoming_profile is None (same defaults as engine fallback).
    """
    return ManagerAgent(
        manager_name="",
        team_name="",
        competition_id=0,
        season_id=0,
        tenure_match_ids=frozenset(),
        pressing_intensity=50.0,
        possession_preference=0.5,
        counter_tendency=0.5,
        preferred_formation="4-4-2",
    )
