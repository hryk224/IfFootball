"""Player agent domain model.

Defines the position taxonomy and PlayerAgent dataclass used throughout the simulation.

Position taxonomy (3 layers):
  position_name (str)  ->  RoleFamily  ->  BroadPosition
  StatsBomb original       9 categories    4 categories

- position_name: the canonical StatsBomb position string, preserved without loss.
- role_family:   mid-level grouping used as the primary axis for rule authoring.
- broad_position: coarse grouping for simple rules that do not need role granularity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class BroadPosition(str, Enum):
    """Coarse 4-category position classification for simple simulation rules."""

    GK = "GK"
    DF = "DF"
    MF = "MF"
    FW = "FW"


class RoleFamily(str, Enum):
    """Mid-level position grouping that preserves tactically meaningful distinctions.

    Used as the primary axis for rule authoring. More granular than BroadPosition
    but still abstract enough to avoid coupling rules to exact StatsBomb strings.
    """

    GOALKEEPER = "goalkeeper"
    CENTER_BACK = "center_back"
    FULL_BACK = "full_back"
    # Wing backs have distinct wide/forward responsibilities from full backs and center backs;
    # treating them as generic defenders would lose pressing, crossing, and overlap behaviour.
    WING_BACK = "wing_back"
    DEFENSIVE_MIDFIELDER = "defensive_midfielder"
    CENTRAL_MIDFIELDER = "central_midfielder"
    ATTACKING_MIDFIELDER = "attacking_midfielder"
    # Wide attacking players (wingers) classified as MF in BroadPosition for M1.
    WINGER = "winger"
    FORWARD = "forward"


@dataclass
class PlayerAgent:
    """Represents a player within the simulation.

    Attributes:
        position_name: StatsBomb original position string adopted at initialisation.
            This is the *representative* position — the most frequently occurring
            position in the player's match events (measured by event count, not
            time-weighted). Not a per-event snapshot.
        role_family: Mid-level role derived from position_name. Use this for most
            simulation rules.
        broad_position: Coarse 4-category classification derived from role_family.
            Use for rules that need only GK/DF/MF/FW granularity.

        Technical attributes (percentile 0–100, relative to players with ≥900 min):
            pace, passing, shooting, pressing, defending, physicality, consistency

        Adaptation attributes (fixed placeholder values for M1):
            tactical_adaptability, leadership, pressure_resistance

        Dynamic state (reset to initial values each simulation run, 0.0-1.0 scale):
            current_form, fatigue, tactical_understanding, manager_trust, bench_streak
    """

    player_id: int
    player_name: str
    position_name: str
    role_family: RoleFamily
    broad_position: BroadPosition

    # Technical attributes — percentile normalised within same league/season cohort
    pace: float
    passing: float
    shooting: float
    pressing: float
    defending: float
    physicality: float
    consistency: float

    # Adaptation attributes — placeholder fixed values until M2 estimation logic
    tactical_adaptability: float = field(default=50.0)
    leadership: float = field(default=50.0)
    pressure_resistance: float = field(default=50.0)

    # Dynamic state — initialised to neutral; updated each match simulation step.
    # Scale: 0.0-1.0 (fatigue and dynamic attributes).
    current_form: float = field(default=0.5)
    fatigue: float = field(default=0.0)
    tactical_understanding: float = field(default=0.5)
    manager_trust: float = field(default=0.5)
    bench_streak: int = field(default=0)
