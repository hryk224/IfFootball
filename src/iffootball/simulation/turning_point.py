"""Turning point detection and rule-based action distribution.

Turning points (TPs) are moments in the simulation where a player's or
manager's situation triggers a behavioural response. Detection uses
config-driven thresholds; the response is an ActionDistribution that
the weekly loop samples to update state.

Phase 1/2 boundary:
    TurningPointHandler is a Protocol. The current implementation
    (RuleBasedHandler) returns fixed distributions. A future Phase 2
    implementation can swap in LLM-sampled distributions without
    changing the caller interface.

Actions:
    adapt    — player adapts to the new system
    resist   — player resists change (performance drops)
    transfer — player signals desire to leave
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from iffootball.agents.manager import ManagerAgent
from iffootball.agents.player import PlayerAgent
from iffootball.config import SimulationRules

# ---------------------------------------------------------------------------
# Action distribution
# ---------------------------------------------------------------------------

VALID_ACTIONS: frozenset[str] = frozenset({"adapt", "resist", "transfer"})


@dataclass
class ActionDistribution:
    """Probability distribution over player actions.

    choices maps action names to probabilities. On construction:
      - Unknown action keys are rejected.
      - Negative probabilities are rejected.
      - Zero total is rejected.
      - Probabilities are normalised to sum to 1.0.
    """

    choices: dict[str, float]

    def __post_init__(self) -> None:
        unknown = set(self.choices) - VALID_ACTIONS
        if unknown:
            raise ValueError(f"Unknown action keys: {unknown}")
        if any(v < 0 for v in self.choices.values()):
            raise ValueError("Negative probability values are not allowed")
        total = sum(self.choices.values())
        if total == 0:
            raise ValueError("Total probability must be > 0")
        # Normalise so probabilities sum to 1.0.
        self.choices = {k: v / total for k, v in self.choices.items()}


# ---------------------------------------------------------------------------
# Simulation context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimContext:
    """Read-only context passed to turning point handlers.

    Attributes:
        current_week:              Current match week number.
        matches_since_appointment: Matches since the current manager was
                                   appointed. None if this is the original
                                   manager (no appointment event).
        manager:                   Current manager state.
        recent_points:             Points earned in recent matches
                                   (most recent last, up to 5).
    """

    current_week: int
    matches_since_appointment: int | None
    manager: ManagerAgent
    recent_points: tuple[int, ...] = ()


# ---------------------------------------------------------------------------
# Handler protocol (Phase 1/2 boundary)
# ---------------------------------------------------------------------------


class TurningPointHandler(Protocol):
    """Interface for turning point response handlers.

    Phase 1: RuleBasedHandler (deterministic distributions).
    Phase 2: LLM-based handler (sampled distributions).
    """

    def handle(
        self, agent: PlayerAgent, context: SimContext,
        *, is_starter: bool = True,
    ) -> ActionDistribution: ...


# ---------------------------------------------------------------------------
# Turning point detection
# ---------------------------------------------------------------------------


def detect_player_turning_points(
    player: PlayerAgent,
    context: SimContext,
    rules: SimulationRules,
    *,
    is_starter: bool = True,
) -> list[str]:
    """Detect active turning points for a player.

    Returns a list of TP type strings (may be empty).
    All thresholds come from rules.turning_points.player.

    TP types:
        "bench_streak":      bench_streak >= bench_streak_threshold
        "low_understanding": tactical_understanding below threshold
                             AND within short_term_window of appointment
                             AND player is a starter this week.
                             Benched players still experience low
                             understanding but do not trigger the
                             behavioural response (resist/adapt/transfer).

    Args:
        player:     The player to evaluate.
        context:    Current simulation context.
        rules:      Simulation rules config.
        is_starter: Whether the player started this week's match.
                    low_understanding TP only fires for starters.

    All values are on 0.0-1.0 scale.
    """
    tp_config = rules.turning_points.player
    tps: list[str] = []

    if player.bench_streak >= tp_config.bench_streak_threshold:
        tps.append("bench_streak")

    if is_starter and context.matches_since_appointment is not None:
        if context.matches_since_appointment <= tp_config.short_term_window:
            if player.tactical_understanding < tp_config.tactical_understanding_low:
                tps.append("low_understanding")

    return tps


def detect_manager_turning_points(
    manager: ManagerAgent,
    rules: SimulationRules,
) -> list[str]:
    """Detect active turning points for the manager.

    Returns a list of TP type strings (may be empty).
    All thresholds come from rules.turning_points.manager.

    TP types:
        "job_security_warning":  job_security < warning threshold
                                 AND style_stubbornness < stubbornness threshold
        "job_security_critical": job_security < critical threshold
    """
    tp_config = rules.turning_points.manager
    tps: list[str] = []

    if manager.job_security < tp_config.job_security_critical:
        tps.append("job_security_critical")
    elif manager.job_security < tp_config.job_security_warning:
        if manager.style_stubbornness < tp_config.style_stubbornness_threshold:
            tps.append("job_security_warning")

    return tps


# ---------------------------------------------------------------------------
# Rule-based handler
# ---------------------------------------------------------------------------


class RuleBasedHandler:
    """Rule-based turning point response handler (Phase 1).

    Returns fixed ActionDistribution based on player state and detected
    turning points. The distributions are provisional and subject to
    community review (documented in docs/simulation-rules.md at M4).
    """

    def __init__(self, rules: SimulationRules) -> None:
        self._rules = rules

    def handle(
        self, agent: PlayerAgent, context: SimContext,
        *, is_starter: bool = True,
    ) -> ActionDistribution:
        """Return an action distribution for the given player and context.

        Decision logic (priority order):
          1. bench_streak TP + low trust → bench_streak_low_trust distribution
          2. low_understanding TP → low_understanding distribution
          3. No TP or unmatched → default distribution

        All distributions are loaded from config (action_distribution section
        in turning_points.toml).
        """
        tps = detect_player_turning_points(
            agent, context, self._rules, is_starter=is_starter,
        )
        ad = self._rules.turning_points.action_distribution

        trust_low = self._rules.turning_points.player.trust_low
        if "bench_streak" in tps and agent.manager_trust < trust_low:
            return ActionDistribution(dict(ad.bench_streak_low_trust))

        if "low_understanding" in tps:
            return ActionDistribution(dict(ad.low_understanding))

        return ActionDistribution(dict(ad.default))
