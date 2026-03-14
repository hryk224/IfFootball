"""Simulation rules configuration loader.

Reads TOML config files from a directory and returns typed, frozen
dataclass instances. All simulation parameters live in config files,
never hardcoded in Python source.

Config directory layout:
    config/simulation_rules/
        adaptation.toml       — weekly state update parameters
        turning_points.toml   — turning point detection thresholds
        match.toml            — match result calculation parameters

Each file uses top-level keys (no wrapper section). The turning_points
file uses [player] and [manager] sections.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Adaptation config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdaptationConfig:
    """Weekly state update parameters.

    All values are on a 0.0-1.0 scale unless noted otherwise.

    Attributes:
        base_fatigue_increase:      Fatigue added per match played.
        base_fatigue_recovery:      Fatigue recovered per match not played.
        tactical_understanding_gain: Base weekly gain in tactical
                                     understanding (before adaptation_rate
                                     scaling).
        fatigue_penalty_weight:     How much fatigue reduces the
                                     agent_state_factor in match result
                                     calculation (0.0-1.0).
        trust_increase_on_start:    manager_trust increase when selected
                                     as starter (0.0-1.0).
        trust_decrease_on_bench:    manager_trust decrease when benched
                                     (0.0-1.0).
    """

    base_fatigue_increase: float
    base_fatigue_recovery: float
    tactical_understanding_gain: float
    fatigue_penalty_weight: float
    trust_increase_on_start: float
    trust_decrease_on_bench: float

    def __post_init__(self) -> None:
        if self.base_fatigue_increase < 0:
            raise ValueError(
                f"base_fatigue_increase must be >= 0, "
                f"got {self.base_fatigue_increase}"
            )
        if self.base_fatigue_recovery < 0:
            raise ValueError(
                f"base_fatigue_recovery must be >= 0, "
                f"got {self.base_fatigue_recovery}"
            )
        if self.tactical_understanding_gain < 0:
            raise ValueError(
                f"tactical_understanding_gain must be >= 0, "
                f"got {self.tactical_understanding_gain}"
            )
        if not 0.0 <= self.fatigue_penalty_weight <= 1.0:
            raise ValueError(
                f"fatigue_penalty_weight must be in [0.0, 1.0], "
                f"got {self.fatigue_penalty_weight}"
            )
        if not 0.0 <= self.trust_increase_on_start <= 1.0:
            raise ValueError(
                f"trust_increase_on_start must be in [0.0, 1.0], "
                f"got {self.trust_increase_on_start}"
            )
        if not 0.0 <= self.trust_decrease_on_bench <= 1.0:
            raise ValueError(
                f"trust_decrease_on_bench must be in [0.0, 1.0], "
                f"got {self.trust_decrease_on_bench}"
            )


# ---------------------------------------------------------------------------
# Match config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchConfig:
    """Match result calculation parameters.

    Attributes:
        home_advantage_factor: Multiplier for expected goals when the
                               simulated team plays at home (or for
                               the opponent when away). Must be > 0.
    """

    home_advantage_factor: float

    def __post_init__(self) -> None:
        if self.home_advantage_factor <= 0:
            raise ValueError(
                f"home_advantage_factor must be > 0, "
                f"got {self.home_advantage_factor}"
            )


# ---------------------------------------------------------------------------
# Turning point config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlayerTurningPointConfig:
    """Player-level turning point thresholds.

    Attributes:
        bench_streak_threshold:     Consecutive benched matches before
                                    a turning point fires.
        tactical_understanding_low: tactical_understanding below this
                                    value triggers short-term confusion
                                    (0.0-1.0 scale).
        short_term_window:          Matches after appointment during which
                                    low tactical_understanding fires a TP.
        trust_low:                  manager_trust below this value combined
                                    with bench_streak TP triggers resist-heavy
                                    response (0.0-1.0).
    """

    bench_streak_threshold: int
    tactical_understanding_low: float
    short_term_window: int
    trust_low: float

    def __post_init__(self) -> None:
        if self.bench_streak_threshold < 1:
            raise ValueError(
                f"bench_streak_threshold must be >= 1, "
                f"got {self.bench_streak_threshold}"
            )
        if not 0.0 <= self.tactical_understanding_low <= 1.0:
            raise ValueError(
                f"tactical_understanding_low must be in [0.0, 1.0], "
                f"got {self.tactical_understanding_low}"
            )
        if self.short_term_window < 1:
            raise ValueError(
                f"short_term_window must be >= 1, "
                f"got {self.short_term_window}"
            )
        if not 0.0 <= self.trust_low <= 1.0:
            raise ValueError(
                f"trust_low must be in [0.0, 1.0], "
                f"got {self.trust_low}"
            )


@dataclass(frozen=True)
class ManagerTurningPointConfig:
    """Manager-level turning point thresholds.

    Attributes:
        job_security_warning:         job_security below this triggers
                                      defensive tactical shift (0.0-1.0).
        job_security_critical:        job_security below this generates
                                      a dismissal event (0.0-1.0).
        style_stubbornness_threshold: style_stubbornness at or above this
                                      prevents tactical shifts (0-100).
    """

    job_security_warning: float
    job_security_critical: float
    style_stubbornness_threshold: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.job_security_warning <= 1.0:
            raise ValueError(
                f"job_security_warning must be in [0.0, 1.0], "
                f"got {self.job_security_warning}"
            )
        if not 0.0 <= self.job_security_critical <= 1.0:
            raise ValueError(
                f"job_security_critical must be in [0.0, 1.0], "
                f"got {self.job_security_critical}"
            )
        if self.job_security_critical > self.job_security_warning:
            raise ValueError(
                f"job_security_critical ({self.job_security_critical}) "
                f"must be <= job_security_warning ({self.job_security_warning})"
            )
        if self.style_stubbornness_threshold < 0:
            raise ValueError(
                f"style_stubbornness_threshold must be >= 0, "
                f"got {self.style_stubbornness_threshold}"
            )


@dataclass(frozen=True)
class TurningPointConfig:
    """Combined player and manager turning point thresholds."""

    player: PlayerTurningPointConfig
    manager: ManagerTurningPointConfig


# ---------------------------------------------------------------------------
# Top-level rules container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulationRules:
    """All simulation rule parameters, loaded from TOML config files.

    Use SimulationRules.load(config_dir) to read from disk.
    """

    adaptation: AdaptationConfig
    turning_points: TurningPointConfig
    match: MatchConfig

    @staticmethod
    def load(config_dir: str | Path) -> SimulationRules:
        """Load simulation rules from a config directory.

        Expects the directory to contain:
            adaptation.toml
            turning_points.toml
            match.toml

        Args:
            config_dir: Path to the directory containing TOML files.

        Returns:
            SimulationRules with all parameters populated.

        Raises:
            FileNotFoundError: if a required TOML file is missing.
            KeyError: if a required key is missing from a TOML file.
            ValueError: if a value fails validation.
        """
        config_dir = Path(config_dir)

        adaptation = _load_adaptation(config_dir / "adaptation.toml")
        turning_points = _load_turning_points(
            config_dir / "turning_points.toml"
        )
        match = _load_match(config_dir / "match.toml")

        return SimulationRules(
            adaptation=adaptation,
            turning_points=turning_points,
            match=match,
        )


# ---------------------------------------------------------------------------
# Internal loaders
# ---------------------------------------------------------------------------


def _load_adaptation(path: Path) -> AdaptationConfig:
    """Load AdaptationConfig from a TOML file."""
    data = _read_toml(path)
    return AdaptationConfig(
        base_fatigue_increase=float(data["base_fatigue_increase"]),
        base_fatigue_recovery=float(data["base_fatigue_recovery"]),
        tactical_understanding_gain=float(
            data["tactical_understanding_gain"]
        ),
        fatigue_penalty_weight=float(data["fatigue_penalty_weight"]),
        trust_increase_on_start=float(data["trust_increase_on_start"]),
        trust_decrease_on_bench=float(data["trust_decrease_on_bench"]),
    )


def _load_match(path: Path) -> MatchConfig:
    """Load MatchConfig from a TOML file."""
    data = _read_toml(path)
    return MatchConfig(
        home_advantage_factor=float(data["home_advantage_factor"]),
    )


def _load_turning_points(path: Path) -> TurningPointConfig:
    """Load TurningPointConfig from a TOML file."""
    data = _read_toml(path)

    player_data = data["player"]
    manager_data = data["manager"]

    player = PlayerTurningPointConfig(
        bench_streak_threshold=int(player_data["bench_streak_threshold"]),
        tactical_understanding_low=float(
            player_data["tactical_understanding_low"]
        ),
        short_term_window=int(player_data["short_term_window"]),
        trust_low=float(player_data["trust_low"]),
    )

    manager = ManagerTurningPointConfig(
        job_security_warning=float(manager_data["job_security_warning"]),
        job_security_critical=float(manager_data["job_security_critical"]),
        style_stubbornness_threshold=float(
            manager_data["style_stubbornness_threshold"]
        ),
    )

    return TurningPointConfig(player=player, manager=manager)


def _read_toml(path: Path) -> dict[str, Any]:
    """Read and parse a TOML file."""
    with path.open("rb") as f:
        return tomllib.load(f)
