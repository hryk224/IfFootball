"""Lineup selection logic.

Selects the starting XI from the squad based on the manager's preferred
formation and a per-player selection_score. Players not selected have
their bench_streak incremented; selected players have it reset to 0.

Selection score formula:
    tactical_fit = (pressing_fit + possession_fit) / 2.0
    where:
        pressing_fit  = player.pressing * normalised_pressing_intensity
        possession_fit = player.passing * manager.possession_preference

    score = tactical_fit
            + player.manager_trust
            + player.current_form
            - player.fatigue * fatigue_penalty_weight * 100.0
            - low_understanding_penalty

    low_understanding_penalty is applied only when:
        1. matches_since_appointment is not None
        2. matches_since_appointment <= rules.turning_points.player.short_term_window
        3. player.tactical_understanding < rules.turning_points.player.tactical_understanding_low * 100.0

    The 100.0 scaling converts 0.0-1.0 config thresholds to the 0-100
    scale used by PlayerAgent attributes. This will be removed when
    PlayerAgent dynamic state migrates to 0.0-1.0 in player-state-update.

    penalty value = (threshold - player.tactical_understanding)
"""

from __future__ import annotations

from dataclasses import dataclass

from iffootball.agents.manager import ManagerAgent
from iffootball.agents.player import BroadPosition, PlayerAgent
from iffootball.config import SimulationRules

# Default formation when manager has no preferred_formation.
_DEFAULT_FORMATION = "4-4-2"

# Normalisation factor for pressing_intensity (raw Pressure events/90).
# Typical PL range is ~40-80; dividing by 100 maps to roughly 0.0-0.8,
# comparable in scale to possession_preference (0.0-1.0).
_PRESSING_NORMALISATION = 100.0


# ---------------------------------------------------------------------------
# Formation parsing
# ---------------------------------------------------------------------------


def parse_formation(formation: str | None) -> dict[BroadPosition, int]:
    """Parse a formation string into position slot counts.

    Format: "{DF}-{MF}-{FW}" (e.g. "4-3-3", "3-5-2").
    GK is always 1 and not included in the string.

    Returns:
        Dict mapping BroadPosition to required count.
        GK is always 1.

    Falls back to _DEFAULT_FORMATION ("4-4-2") if formation is None.

    Raises:
        ValueError: if the formation string cannot be parsed into
                    exactly 3 integer segments.
    """
    raw = formation if formation is not None else _DEFAULT_FORMATION
    parts = raw.split("-")
    if len(parts) != 3:
        raise ValueError(
            f"Formation must have 3 segments (DF-MF-FW), got: {raw!r}"
        )
    try:
        df_count, mf_count, fw_count = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        raise ValueError(
            f"Formation segments must be integers, got: {raw!r}"
        ) from None

    return {
        BroadPosition.GK: 1,
        BroadPosition.DF: df_count,
        BroadPosition.MF: mf_count,
        BroadPosition.FW: fw_count,
    }


# ---------------------------------------------------------------------------
# Selection score
# ---------------------------------------------------------------------------


def calc_selection_score(
    player: PlayerAgent,
    manager: ManagerAgent,
    rules: SimulationRules,
    matches_since_appointment: int | None,
) -> float:
    """Compute selection score for a player under the given manager.

    Higher score means more likely to be selected as a starter.

    Args:
        player:                    The player to evaluate.
        manager:                   Current manager's tactical profile.
        rules:                     Simulation rules config.
        matches_since_appointment: Matches since the current manager was
                                   appointed. None if this is the original
                                   manager (no appointment event).
    """
    # Tactical fit: how well the player suits the manager's style.
    normalised_pressing = manager.pressing_intensity / _PRESSING_NORMALISATION
    pressing_fit = player.pressing * normalised_pressing
    possession_fit = player.passing * manager.possession_preference
    tactical_fit = (pressing_fit + possession_fit) / 2.0

    # Fatigue penalty (config-driven weight).
    fatigue_penalty = (
        player.fatigue
        * rules.adaptation.fatigue_penalty_weight
        * 100.0  # scale to match 0-100 attribute range
    )

    # Low tactical understanding penalty (short-term window only).
    low_understanding_penalty = 0.0
    tp = rules.turning_points.player
    if matches_since_appointment is not None:
        if matches_since_appointment <= tp.short_term_window:
            # Convert 0.0-1.0 threshold to 0-100 scale.
            threshold = tp.tactical_understanding_low * 100.0
            if player.tactical_understanding < threshold:
                low_understanding_penalty = (
                    threshold - player.tactical_understanding
                )

    return (
        tactical_fit
        + player.manager_trust
        + player.current_form
        - fatigue_penalty
        - low_understanding_penalty
    )


# ---------------------------------------------------------------------------
# Lineup selection
# ---------------------------------------------------------------------------


@dataclass
class LineupResult:
    """Result of lineup selection.

    Attributes:
        starters: Selected starting XI (up to 11 players).
        benched:  Players not selected.
    """

    starters: list[PlayerAgent]
    benched: list[PlayerAgent]


def select_lineup(
    squad: list[PlayerAgent],
    manager: ManagerAgent,
    rules: SimulationRules,
    matches_since_appointment: int | None = None,
) -> LineupResult:
    """Select the starting XI from the squad.

    Process:
    1. Parse the manager's preferred_formation into position slots.
    2. For each BroadPosition slot, pick the top-scoring players by
       calc_selection_score.
    3. If a slot cannot be filled from its own position, remaining slots
       are filled from the highest-scoring unselected players regardless
       of position (GK excluded from overflow).
    4. Update bench_streak: reset to 0 for starters, increment for benched.

    Args:
        squad:                     All available players.
        manager:                   Current manager.
        rules:                     Simulation rules config.
        matches_since_appointment: Matches since appointment (None if
                                   original manager).

    Returns:
        LineupResult with starters and benched lists.
        bench_streak is updated on each PlayerAgent in place.
    """
    slots = parse_formation(manager.preferred_formation)

    # Pre-compute scores for all players.
    scores: dict[int, float] = {
        p.player_id: calc_selection_score(
            p, manager, rules, matches_since_appointment
        )
        for p in squad
    }

    # Group players by BroadPosition.
    by_position: dict[BroadPosition, list[PlayerAgent]] = {
        bp: [] for bp in BroadPosition
    }
    for p in squad:
        by_position[p.broad_position].append(p)

    # Sort each group by score descending (tie-break: player_id ascending).
    for players in by_position.values():
        players.sort(key=lambda p: (-scores[p.player_id], p.player_id))

    selected_ids: set[int] = set()
    starters: list[PlayerAgent] = []

    # Fill each position slot from the matching group.
    for bp in (
        BroadPosition.GK,
        BroadPosition.DF,
        BroadPosition.MF,
        BroadPosition.FW,
    ):
        needed = slots.get(bp, 0)
        available = [
            p for p in by_position[bp] if p.player_id not in selected_ids
        ]
        picked = available[:needed]
        starters.extend(picked)
        selected_ids.update(p.player_id for p in picked)

    # Fill remaining slots from highest-scoring unselected (non-GK).
    total_needed = sum(slots.values())
    if len(starters) < total_needed:
        remaining = [
            p
            for p in squad
            if p.player_id not in selected_ids
            and p.broad_position != BroadPosition.GK
        ]
        remaining.sort(key=lambda p: (-scores[p.player_id], p.player_id))
        for p in remaining:
            if len(starters) >= total_needed:
                break
            starters.append(p)
            selected_ids.add(p.player_id)

    # Update bench_streak.
    benched: list[PlayerAgent] = []
    for p in squad:
        if p.player_id in selected_ids:
            p.bench_streak = 0
        else:
            p.bench_streak += 1
            benched.append(p)

    return LineupResult(starters=starters, benched=benched)
