"""Match result determination via Poisson model.

Simulates a single match by computing expected goals from team/opponent
baselines and player agent states, then sampling actual goals from
Poisson distributions.

Expected goals formula:
    expected_goals_for     = team.xg_for_per90
                             * opponent.xg_against_per90
                             * agent_state_factor
    expected_goals_against = opponent.xg_for_per90
                             * team.xg_against_per90

agent_state_factor reflects the average condition of the starting XI:
    avg_form    = mean(current_form for each starter) / 0.5
    avg_fatigue = mean(fatigue for each starter)
    factor      = avg_form * (1.0 - avg_fatigue * fatigue_penalty_weight)
    clamped to [0.5, 1.5]

The factor centres at 1.0 when all starters have neutral form (0.5) and
zero fatigue. Higher form or lower fatigue increases expected goals;
the reverse decreases them. Clamping prevents unrealistic extremes.

Reproducibility:
    Pass a seeded numpy.random.Generator to simulate_match() for
    deterministic results across runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iffootball.agents.fixture import Fixture, OpponentStrength
from iffootball.agents.player import PlayerAgent
from iffootball.agents.team import TeamBaseline
from iffootball.config import AdaptationConfig, MatchConfig

# agent_state_factor bounds to prevent unrealistic match outcomes.
_STATE_FACTOR_MIN = 0.5
_STATE_FACTOR_MAX = 1.5

# Neutral current_form value (PlayerAgent default, 0.0-1.0 scale).
_NEUTRAL_FORM = 0.5


@dataclass(frozen=True)
class MatchResult:
    """Result of a single simulated match.

    Attributes:
        home_goals:             Goals scored by the home team.
        away_goals:             Goals scored by the away team.
        is_home:                Whether the simulated team played at home.
        points_earned:          Points earned by the simulated team (3/1/0).
        expected_goals_for:     Poisson lambda used for the simulated team's
                                goals (after home advantage adjustment).
        expected_goals_against: Poisson lambda used for the opponent's goals
                                (after home advantage adjustment).
    """

    home_goals: int
    away_goals: int
    is_home: bool
    points_earned: int
    expected_goals_for: float
    expected_goals_against: float


def calc_agent_state_factor(
    starters: list[PlayerAgent],
    fatigue_penalty_weight: float,
) -> float:
    """Compute the agent state factor from the starting XI condition.

    Returns a multiplicative factor (clamped to [0.5, 1.5]) that adjusts
    the team's expected goals based on player form and fatigue.

    When starters is empty, returns 1.0 (neutral).

    Args:
        starters:               Starting XI players.
        fatigue_penalty_weight: How much fatigue reduces the factor
                                (from AdaptationConfig).
    """
    if not starters:
        return 1.0

    avg_form = sum(p.current_form for p in starters) / len(starters)
    avg_fatigue = sum(p.fatigue for p in starters) / len(starters)

    factor = (avg_form / _NEUTRAL_FORM) * (
        1.0 - avg_fatigue * fatigue_penalty_weight
    )

    return max(_STATE_FACTOR_MIN, min(_STATE_FACTOR_MAX, factor))


def simulate_match(
    team: TeamBaseline,
    opponent: OpponentStrength,
    starters: list[PlayerAgent],
    fixture: Fixture,
    adaptation: AdaptationConfig,
    match_config: MatchConfig,
    rng: np.random.Generator,
) -> MatchResult:
    """Simulate a single match and return the result.

    Expected goals are computed from team/opponent baselines scaled by the
    agent state factor, then adjusted for home advantage and sampled via
    Poisson distribution.

    Args:
        team:         Simulated team's baseline metrics.
        opponent:     Opponent's strength snapshot at trigger point.
        starters:     Starting XI of the simulated team.
        fixture:      Fixture being played (provides is_home).
        adaptation:   Adaptation config (provides fatigue_penalty_weight).
        match_config: Match config (provides home_advantage_factor).
        rng:          Seeded random generator for reproducibility.

    Returns:
        MatchResult with goals, home/away flag, and points earned.
    """
    state_factor = calc_agent_state_factor(
        starters, adaptation.fatigue_penalty_weight
    )

    expected_for = (
        team.xg_for_per90 * opponent.xg_against_per90 * state_factor
    )
    expected_against = opponent.xg_for_per90 * team.xg_against_per90

    # Home advantage: boost expected goals for the home side.
    if fixture.is_home:
        expected_for *= match_config.home_advantage_factor
    else:
        expected_against *= match_config.home_advantage_factor

    # PAIRED CONTRACT: exactly 2 RNG calls in this fixed order.
    # Changing the count or order breaks paired A/B comparison guarantees.
    # NOTE: numpy Poisson consumption is lambda-dependent internally.
    # When A/B have different lambdas (post-trigger), this causes RNG
    # desync for subsequent fixtures. See test_poisson_consumption_is_lambda_dependent.
    goals_for = int(rng.poisson(max(expected_for, 0.0)))
    goals_against = int(rng.poisson(max(expected_against, 0.0)))

    # Map to home/away goals based on fixture.
    if fixture.is_home:
        home_goals = goals_for
        away_goals = goals_against
    else:
        home_goals = goals_against
        away_goals = goals_for

    # Points for the simulated team.
    if goals_for > goals_against:
        points = 3
    elif goals_for == goals_against:
        points = 1
    else:
        points = 0

    return MatchResult(
        home_goals=home_goals,
        away_goals=away_goals,
        is_home=fixture.is_home,
        points_earned=points,
        expected_goals_for=expected_for,
        expected_goals_against=expected_against,
    )
