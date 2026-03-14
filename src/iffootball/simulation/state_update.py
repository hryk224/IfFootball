"""Weekly state update functions (simulation loop steps 4-7).

Updates player and manager dynamic state after each match week:
  Step 4: fatigue
  Step 5: tactical_understanding (via adaptation curve)
  Step 6: manager_trust
  Step 7: job_security

All parameter values come from SimulationRules (no hardcoded thresholds).

Scale note:
    PlayerAgent dynamic attributes (current_form, manager_trust,
    tactical_understanding) are currently on a 0-100 scale.
    Config values (tactical_understanding_gain) are on 0.0-1.0.
    Conversions use * 100.0 where needed. These will be removed when
    PlayerAgent migrates to 0.0-1.0 in a future task.
"""

from __future__ import annotations

from iffootball.agents.manager import ManagerAgent
from iffootball.agents.player import BroadPosition, PlayerAgent
from iffootball.config import SimulationRules
from iffootball.simulation.lineup_selection import parse_formation


# ---------------------------------------------------------------------------
# Step 4: Fatigue update
# ---------------------------------------------------------------------------


def update_fatigue(
    squad: list[PlayerAgent],
    starter_ids: set[int],
    rules: SimulationRules,
) -> None:
    """Update fatigue for all players after a match week.

    Starters gain fatigue; non-starters recover.
    Fatigue is clamped to [0.0, 1.0].

    Args:
        squad:       All players in the squad.
        starter_ids: player_id set of players who started this week.
        rules:       Simulation rules config.
    """
    for p in squad:
        if p.player_id in starter_ids:
            p.fatigue = min(1.0, p.fatigue + rules.adaptation.base_fatigue_increase)
        else:
            p.fatigue = max(0.0, p.fatigue - rules.adaptation.base_fatigue_recovery)


# ---------------------------------------------------------------------------
# Step 5: Tactical understanding update
# ---------------------------------------------------------------------------


def calc_tactical_familiarity(
    player: PlayerAgent,
    preferred_formation: str | None,
) -> float:
    """Compute how familiar a player is with the manager's formation.

    BroadPosition-based coarse approximation: if the formation has slots
    for the player's BroadPosition, familiarity is 1.0; otherwise 0.5
    (the player must adapt to a role outside their natural position group).

    GK always returns 1.0 (every formation uses a goalkeeper).

    This is a simplified proxy. A more granular approach using RoleFamily
    would better capture e.g. winger-vs-central-midfielder transitions,
    but is deferred to a future refinement.

    Args:
        player:              The player to evaluate.
        preferred_formation: Manager's preferred formation string
                             (e.g. "4-3-3"). None uses default.
    """
    if player.broad_position == BroadPosition.GK:
        return 1.0

    slots = parse_formation(preferred_formation)
    if slots.get(player.broad_position, 0) > 0:
        return 1.0
    return 0.5


def calc_adaptation_rate(
    player: PlayerAgent,
    manager: ManagerAgent,
) -> float:
    """Compute weekly adaptation rate for tactical understanding gain.

    rate = (tactical_adaptability / 100) * (implementation_speed / 100) * familiarity

    Returns a value in [0.0, 1.0] that scales the weekly
    tactical_understanding_gain from config.
    """
    base_rate = player.tactical_adaptability / 100.0
    manager_factor = manager.implementation_speed / 100.0
    familiarity = calc_tactical_familiarity(
        player, manager.preferred_formation
    )
    return base_rate * manager_factor * familiarity


def update_tactical_understanding(
    squad: list[PlayerAgent],
    manager: ManagerAgent,
    rules: SimulationRules,
) -> None:
    """Update tactical_understanding for all players.

    gain = adaptation_rate * tactical_understanding_gain * 100.0
    (config gain is 0.0-1.0 scale; PlayerAgent attribute is 0-100)

    Capped at 100.0.

    Args:
        squad:   All players in the squad.
        manager: Current manager.
        rules:   Simulation rules config.
    """
    gain_config = rules.adaptation.tactical_understanding_gain
    for p in squad:
        rate = calc_adaptation_rate(p, manager)
        # Convert 0.0-1.0 config gain to 0-100 attribute scale.
        delta = rate * gain_config * 100.0
        p.tactical_understanding = min(100.0, p.tactical_understanding + delta)


# ---------------------------------------------------------------------------
# Step 6: Manager trust update
# ---------------------------------------------------------------------------


def update_manager_trust(
    squad: list[PlayerAgent],
    starter_ids: set[int],
    rules: SimulationRules,
) -> None:
    """Update manager_trust for all players after a match week.

    Starters gain trust; non-starters lose trust.
    Trust is clamped to [0.0, 100.0] (0-100 scale).

    Args:
        squad:       All players in the squad.
        starter_ids: player_id set of players who started this week.
        rules:       Simulation rules config.
    """
    for p in squad:
        if p.player_id in starter_ids:
            p.manager_trust = min(
                100.0,
                p.manager_trust + rules.adaptation.trust_increase_on_start,
            )
        else:
            p.manager_trust = max(
                0.0,
                p.manager_trust - rules.adaptation.trust_decrease_on_bench,
            )


# ---------------------------------------------------------------------------
# Step 7: Job security update
# ---------------------------------------------------------------------------

# Maximum possible points over the recent window (5 wins * 3 pts).
_MAX_RECENT_POINTS = 15.0


def update_job_security(
    manager: ManagerAgent,
    recent_points: list[int],
) -> None:
    """Update manager's job_security from recent match results.

    job_security = sum(recent_points) / 15.0
    where recent_points is the last 5 match results (3/1/0 each).

    Clamped to [0.0, 1.0].

    If recent_points is empty, job_security is unchanged.

    Args:
        manager:       Current manager (mutated in place).
        recent_points: Points earned in recent matches (up to 5).
    """
    if not recent_points:
        return

    total = sum(recent_points)
    manager.job_security = max(0.0, min(1.0, total / _MAX_RECENT_POINTS))
