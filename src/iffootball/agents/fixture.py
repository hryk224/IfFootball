"""Fixture and opponent strength domain models.

FixtureList and OpponentStrength are shared between Branch A/B to ensure
a fair parallel comparison: both branches face the same opponents in the
same order.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fixture:
    """A single remaining fixture for the simulated team.

    Attributes:
        match_week:    League round number.
        opponent_name: StatsBomb team name of the opponent.
                       Corresponds to opponent_id in the architecture
                       spec; M1 uses the StatsBomb team name string
                       as the identifier.
        is_home:       True if the simulated team plays at home.
    """

    match_week: int
    opponent_name: str
    is_home: bool


@dataclass(frozen=True)
class FixtureList:
    """Ordered list of fixtures for a team's season.

    Frozen and immutable so it can be safely shared between Branch A/B
    without risk of accidental mutation during simulation.

    Attributes:
        team_name: StatsBomb team name of the simulated team.
        fixtures:  Fixtures sorted by match_week ascending, then
                   match_id ascending within the same week.
    """

    team_name: str
    fixtures: tuple[Fixture, ...]


@dataclass(frozen=True)
class OpponentStrength:
    """Static strength snapshot of an opponent at the trigger point.

    Used as a fixed input to trial match result calculation. The opponent
    is not simulated; only their historical metrics up to trigger_week
    are used.

    This is intentionally lighter than TeamBaseline: no simulation state,
    no league standing context. Only the metrics needed for the Poisson
    match result model.

    Corresponds to OpponentStrength in the architecture spec.
    opponent_name maps to opponent_id (architecture uses str ID;
    M1 uses StatsBomb team name string).

    Attributes:
        opponent_name:    StatsBomb team name.
        xg_for_per90:     Opponent's attacking strength (xG / 90 min).
                          Computed over matches up to trigger_week.
        xg_against_per90: Opponent's defensive weakness (xGA / 90 min).
                          Computed over matches up to trigger_week.
        elo_rating:       Elo rating at trigger_week.
                          Initialised at 1500; updated from match results
                          chronologically up to trigger_week with K=20.
    """

    opponent_name: str
    xg_for_per90: float
    xg_against_per90: float
    elo_rating: float
