"""Weekly state update functions (simulation loop steps 4-7).

Updates player and manager dynamic state after each match week:
  Step 4: fatigue
  Step 5: tactical_understanding (via adaptation curve)
  Step 6: manager_trust
  Step 7: job_security

All parameter values come from SimulationRules (no hardcoded thresholds).

Scale: PlayerAgent dynamic attributes (current_form, manager_trust,
tactical_understanding) and all config values are on 0.0-1.0.
Technical/adaptation attributes (tactical_adaptability, implementation_speed)
remain 0-100 and are normalised within calc_adaptation_rate().
"""

from __future__ import annotations

from iffootball.agents.manager import ManagerAgent
from iffootball.agents.player import BroadPosition, PlayerAgent
from iffootball.config import AdaptationConfig, SimulationRules
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
# Step 4b: Current form update
# ---------------------------------------------------------------------------


def update_current_form(
    squad: list[PlayerAgent],
    starter_ids: set[int],
    points_earned: int,
    rules: SimulationRules,
) -> None:
    """Update current_form for starters based on match result.

    Starters gain form on a win and lose form on a loss. Draws have no
    effect. Non-starters keep their current form unchanged.

    Form represents recent team-result momentum: "winning lifts
    confidence, losing erodes it". It does NOT overlap with trust
    (manager's selection preference) or fatigue (physical load).

    Form is clamped to [0.0, 1.0].

    Args:
        squad:         All players in the squad.
        starter_ids:   player_id set of players who started this week.
        points_earned: Points from this match (3=win, 1=draw, 0=loss).
        rules:         Simulation rules config.
    """
    if points_earned == 3:
        delta = rules.adaptation.form_boost_on_win
    elif points_earned == 0:
        delta = -rules.adaptation.form_drop_on_loss
    else:
        return  # Draw: no form change.

    for p in squad:
        if p.player_id in starter_ids:
            p.current_form = max(0.0, min(1.0, p.current_form + delta))


# ---------------------------------------------------------------------------
# Step 5: Tactical understanding update
# ---------------------------------------------------------------------------


def calc_initial_understanding(
    manager: ManagerAgent,
    config: AdaptationConfig,
) -> float:
    """Compute initial tactical_understanding for a new tactical context.

    Used when a player enters a new tactical environment: either an
    existing squad receiving a new manager, or a new signing joining
    the current manager's system.

    initial = base + (implementation_speed / 100) * speed_bonus

    Higher implementation_speed means the manager communicates tactics
    faster, giving the player a head start. This is a one-time offset;
    the weekly gain rate (calc_adaptation_rate) is a separate mechanism
    that also uses implementation_speed.

    The result is clamped to [0.0, 1.0].

    Args:
        manager: The current or newly appointed manager.
        config:  Adaptation config with base and speed_bonus parameters.
    """
    speed_factor = manager.implementation_speed / 100.0
    value = config.initial_understanding_base + speed_factor * config.initial_understanding_speed_bonus
    return max(0.0, min(1.0, value))


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

    tactical_adaptability and implementation_speed are 0-100 scale
    attributes; dividing by 100 normalises them to [0.0, 1.0].

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

    gain = adaptation_rate * tactical_understanding_gain
    Both are on 0.0-1.0 scale. Capped at 1.0.

    Args:
        squad:   All players in the squad.
        manager: Current manager.
        rules:   Simulation rules config.
    """
    gain_config = rules.adaptation.tactical_understanding_gain
    for p in squad:
        rate = calc_adaptation_rate(p, manager)
        delta = rate * gain_config
        p.tactical_understanding = min(1.0, p.tactical_understanding + delta)


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
    Trust is clamped to [0.0, 1.0].

    Args:
        squad:       All players in the squad.
        starter_ids: player_id set of players who started this week.
        rules:       Simulation rules config.
    """
    for p in squad:
        if p.player_id in starter_ids:
            p.manager_trust = min(
                1.0,
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
