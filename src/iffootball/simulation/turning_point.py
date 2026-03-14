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
        self, agent: PlayerAgent, context: SimContext
    ) -> ActionDistribution: ...


# ---------------------------------------------------------------------------
# Turning point detection
# ---------------------------------------------------------------------------


def detect_player_turning_points(
    player: PlayerAgent,
    context: SimContext,
    rules: SimulationRules,
) -> list[str]:
    """Detect active turning points for a player.

    Returns a list of TP type strings (may be empty).
    All thresholds come from rules.turning_points.player.

    TP types:
        "bench_streak":      bench_streak >= bench_streak_threshold
        "low_understanding": tactical_understanding below threshold
                             AND within short_term_window of appointment

    Note: tactical_understanding threshold is converted from 0.0-1.0
    config scale to 0-100 PlayerAgent scale.
    """
    tp_config = rules.turning_points.player
    tps: list[str] = []

    if player.bench_streak >= tp_config.bench_streak_threshold:
        tps.append("bench_streak")

    if context.matches_since_appointment is not None:
        if context.matches_since_appointment <= tp_config.short_term_window:
            threshold = tp_config.tactical_understanding_low * 100.0
            if player.tactical_understanding < threshold:
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
        self, agent: PlayerAgent, context: SimContext
    ) -> ActionDistribution:
        """Return an action distribution for the given player and context.

        Decision logic:
          1. bench_streak TP + low trust → resist-heavy
          2. low_understanding TP → adapt-heavy (confusion but trying)
          3. No TP → strong adapt (stable adaptation)
        """
        tps = detect_player_turning_points(agent, context, self._rules)

        trust_low = self._rules.turning_points.player.trust_low
        if "bench_streak" in tps and agent.manager_trust < trust_low:
            return ActionDistribution(
                {"resist": 0.6, "adapt": 0.3, "transfer": 0.1}
            )

        if "low_understanding" in tps:
            return ActionDistribution(
                {"adapt": 0.5, "resist": 0.4, "transfer": 0.1}
            )

        return ActionDistribution(
            {"adapt": 0.8, "resist": 0.2, "transfer": 0.0}
        )
