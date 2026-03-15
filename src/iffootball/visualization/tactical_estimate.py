"""Tactical metric estimates from manager profile.

Estimates team performance metrics (PPDA, possession, progressive passes)
based on the manager's tactical profile relative to the team baseline and
league average. These are labelled as estimates (est.), not simulation
outputs, because the simulation engine does not model these metrics
directly.

Approach:
    Each estimate blends the team baseline toward the league average, then
    adjusts for the manager's tactical profile. This reflects the idea that
    a new manager shifts the team's style toward league norms first (loss
    of the old manager's imprint), then toward their own tendencies.

    blend = baseline * (1 - regression_weight) + league_avg * regression_weight
    estimate = blend * adjustment_from_manager_profile

    regression_weight is 0.0 for the original manager (no change) and
    higher for a new manager (tactical reset).
"""

from __future__ import annotations

from iffootball.agents.league import LeagueContext
from iffootball.agents.manager import ManagerAgent
from iffootball.agents.team import TeamBaseline

# Weight for regression toward league average when a new manager arrives.
# 0.0 = keep baseline as-is; 1.0 = full regression to league avg.
_NEW_MANAGER_REGRESSION = 0.3

# Reference pressing_intensity for neutral PPDA (no directional shift).
_NEUTRAL_PRESSING = 50.0

# Reference possession_preference for neutral possession (no shift).
_NEUTRAL_POSSESSION = 0.5


def estimate_ppda(
    baseline: TeamBaseline,
    manager: ManagerAgent,
    league: LeagueContext,
    *,
    is_new_manager: bool,
) -> float:
    """Estimate PPDA from manager pressing intensity.

    Higher pressing_intensity → lower PPDA (more aggressive pressing).
    Uses league average PPDA as the regression anchor.

    Args:
        baseline:       Team baseline with original PPDA.
        manager:        Manager whose pressing_intensity drives the estimate.
        league:         League context with avg_ppda.
        is_new_manager: True if this manager replaced the original.

    Returns:
        Estimated PPDA (lower = more pressing). Clamped to [1.0, inf).
    """
    if league.avg_ppda <= 0.0:
        return baseline.ppda

    regression = _NEW_MANAGER_REGRESSION if is_new_manager else 0.0
    blended = baseline.ppda * (1.0 - regression) + league.avg_ppda * regression

    # pressing_intensity above neutral → PPDA decreases (more pressing).
    # pressing_intensity below neutral → PPDA increases (less pressing).
    intensity_ratio = manager.pressing_intensity / _NEUTRAL_PRESSING
    # Invert: higher intensity → lower PPDA.
    if intensity_ratio > 0.0:
        adjusted = blended / intensity_ratio
    else:
        adjusted = blended

    return max(1.0, adjusted)


def estimate_possession(
    baseline: TeamBaseline,
    manager: ManagerAgent,
    league_avg_possession: float,
    *,
    is_new_manager: bool,
) -> float:
    """Estimate possession % from manager possession preference.

    Args:
        baseline:              Team baseline with original possession_pct.
        manager:               Manager whose possession_preference drives
                               the estimate.
        league_avg_possession: League average possession (typically ~0.5).
        is_new_manager:        True if this manager replaced the original.

    Returns:
        Estimated possession percentage (0.0–1.0).
    """
    regression = _NEW_MANAGER_REGRESSION if is_new_manager else 0.0
    blended = (
        baseline.possession_pct * (1.0 - regression)
        + league_avg_possession * regression
    )

    # Shift blended value toward manager's preference.
    pref_delta = manager.possession_preference - _NEUTRAL_POSSESSION
    adjusted = blended + pref_delta * 0.3  # damped shift

    return max(0.0, min(1.0, adjusted))


def estimate_progressive_passes(
    baseline: TeamBaseline,
    manager: ManagerAgent,
    league: LeagueContext,
    *,
    is_new_manager: bool,
) -> float:
    """Estimate progressive passes per 90 from manager profile.

    Progressive passing is influenced by possession preference (more
    possession → more passing opportunities) and formation (wider
    formations tend to create more progressive passing lanes).

    Args:
        baseline:       Team baseline with original progressive_passes_per90.
        manager:        Manager whose profile drives the estimate.
        league:         League context with avg_progressive_passes_per90.
        is_new_manager: True if this manager replaced the original.

    Returns:
        Estimated progressive passes per 90. Clamped to [0.0, inf).
    """
    if league.avg_progressive_passes_per90 <= 0.0:
        return baseline.progressive_passes_per90

    regression = _NEW_MANAGER_REGRESSION if is_new_manager else 0.0
    blended = (
        baseline.progressive_passes_per90 * (1.0 - regression)
        + league.avg_progressive_passes_per90 * regression
    )

    # Possession preference factor: higher possession → more opportunities.
    poss_factor = 1.0 + (manager.possession_preference - _NEUTRAL_POSSESSION) * 0.4

    # Formation factor: formations with more midfielders/forwards tend to
    # produce more progressive passes.
    formation_factor = _formation_progression_factor(manager.preferred_formation)

    adjusted = blended * poss_factor * formation_factor
    return max(0.0, adjusted)


def _formation_progression_factor(formation: str | None) -> float:
    """Return a multiplicative factor based on formation shape.

    Formations with more midfielders or attacking players get a slight
    boost. Returns 1.0 for unknown or neutral formations.
    """
    if formation is None:
        return 1.0

    parts = formation.split("-")
    if len(parts) < 3:
        return 1.0

    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return 1.0

    # Sum of midfield + forward players (exclude defenders).
    attacking_players = sum(nums[1:])

    # 6 is a neutral midpoint (e.g., 4-3-3 has 6, 3-5-2 has 7, 5-4-1 has 5).
    return 1.0 + (attacking_players - 6) * 0.03
