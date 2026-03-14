"""Tests for simulation rules configuration loader."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from iffootball.config import (
    AdaptationConfig,
    ManagerTurningPointConfig,
    PlayerTurningPointConfig,
    SimulationRules,
    TurningPointConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_toml(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content), encoding="utf-8")


def _make_config_dir(
    tmp_path: Path,
    adaptation: str | None = None,
    turning_points: str | None = None,
) -> Path:
    """Create a config directory with default or custom TOML files."""
    config_dir = tmp_path / "simulation_rules"
    config_dir.mkdir()

    if adaptation is None:
        adaptation = """\
            base_fatigue_increase = 0.05
            base_fatigue_recovery = 0.03
            tactical_understanding_gain = 0.04
        """
    _write_toml(config_dir / "adaptation.toml", adaptation)

    if turning_points is None:
        turning_points = """\
            [player]
            bench_streak_threshold = 3
            tactical_understanding_low = 0.40
            short_term_window = 4

            [manager]
            job_security_warning = 0.30
            job_security_critical = 0.10
            style_stubbornness_threshold = 80
        """
    _write_toml(config_dir / "turning_points.toml", turning_points)

    return config_dir


# ---------------------------------------------------------------------------
# SimulationRules.load — happy path
# ---------------------------------------------------------------------------


class TestSimulationRulesLoad:
    def test_loads_from_directory(self, tmp_path: Path) -> None:
        config_dir = _make_config_dir(tmp_path)
        rules = SimulationRules.load(config_dir)

        assert isinstance(rules, SimulationRules)
        assert isinstance(rules.adaptation, AdaptationConfig)
        assert isinstance(rules.turning_points, TurningPointConfig)

    def test_adaptation_values(self, tmp_path: Path) -> None:
        config_dir = _make_config_dir(tmp_path)
        rules = SimulationRules.load(config_dir)

        assert rules.adaptation.base_fatigue_increase == 0.05
        assert rules.adaptation.base_fatigue_recovery == 0.03
        assert rules.adaptation.tactical_understanding_gain == 0.04

    def test_turning_points_player_values(self, tmp_path: Path) -> None:
        config_dir = _make_config_dir(tmp_path)
        rules = SimulationRules.load(config_dir)

        p = rules.turning_points.player
        assert p.bench_streak_threshold == 3
        assert p.tactical_understanding_low == 0.40
        assert p.short_term_window == 4

    def test_turning_points_manager_values(self, tmp_path: Path) -> None:
        config_dir = _make_config_dir(tmp_path)
        rules = SimulationRules.load(config_dir)

        m = rules.turning_points.manager
        assert m.job_security_warning == 0.30
        assert m.job_security_critical == 0.10
        assert m.style_stubbornness_threshold == 80

    def test_frozen(self, tmp_path: Path) -> None:
        config_dir = _make_config_dir(tmp_path)
        rules = SimulationRules.load(config_dir)

        with pytest.raises(AttributeError):
            rules.adaptation = AdaptationConfig(0.1, 0.1, 0.1)  # type: ignore[misc]

    def test_loads_from_real_config(self) -> None:
        """Verify that the actual config files in the repo are loadable."""
        config_dir = (
            Path(__file__).parents[1] / "config" / "simulation_rules"
        )
        rules = SimulationRules.load(config_dir)
        assert rules.adaptation.base_fatigue_increase > 0
        assert rules.turning_points.player.bench_streak_threshold >= 1


# ---------------------------------------------------------------------------
# Missing file / missing key
# ---------------------------------------------------------------------------


class TestSimulationRulesLoadErrors:
    def test_missing_adaptation_file(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "simulation_rules"
        config_dir.mkdir()
        _write_toml(
            config_dir / "turning_points.toml",
            """\
            [player]
            bench_streak_threshold = 3
            tactical_understanding_low = 0.40
            short_term_window = 4
            [manager]
            job_security_warning = 0.30
            job_security_critical = 0.10
            style_stubbornness_threshold = 80
            """,
        )
        with pytest.raises(FileNotFoundError):
            SimulationRules.load(config_dir)

    def test_missing_key_in_adaptation(self, tmp_path: Path) -> None:
        config_dir = _make_config_dir(
            tmp_path,
            adaptation="""\
                base_fatigue_increase = 0.05
                base_fatigue_recovery = 0.03
            """,
        )
        with pytest.raises(KeyError):
            SimulationRules.load(config_dir)

    def test_missing_player_section(self, tmp_path: Path) -> None:
        config_dir = _make_config_dir(
            tmp_path,
            turning_points="""\
                [manager]
                job_security_warning = 0.30
                job_security_critical = 0.10
                style_stubbornness_threshold = 80
            """,
        )
        with pytest.raises(KeyError):
            SimulationRules.load(config_dir)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestAdaptationConfigValidation:
    def test_negative_fatigue_increase(self) -> None:
        with pytest.raises(ValueError, match="base_fatigue_increase"):
            AdaptationConfig(
                base_fatigue_increase=-0.01,
                base_fatigue_recovery=0.03,
                tactical_understanding_gain=0.04,
            )

    def test_negative_fatigue_recovery(self) -> None:
        with pytest.raises(ValueError, match="base_fatigue_recovery"):
            AdaptationConfig(
                base_fatigue_increase=0.05,
                base_fatigue_recovery=-0.01,
                tactical_understanding_gain=0.04,
            )

    def test_negative_understanding_gain(self) -> None:
        with pytest.raises(ValueError, match="tactical_understanding_gain"):
            AdaptationConfig(
                base_fatigue_increase=0.05,
                base_fatigue_recovery=0.03,
                tactical_understanding_gain=-0.01,
            )

    def test_zero_values_are_valid(self) -> None:
        config = AdaptationConfig(
            base_fatigue_increase=0.0,
            base_fatigue_recovery=0.0,
            tactical_understanding_gain=0.0,
        )
        assert config.base_fatigue_increase == 0.0


class TestPlayerTurningPointValidation:
    def test_bench_streak_threshold_zero(self) -> None:
        with pytest.raises(ValueError, match="bench_streak_threshold"):
            PlayerTurningPointConfig(
                bench_streak_threshold=0,
                tactical_understanding_low=0.40,
                short_term_window=4,
            )

    def test_tactical_understanding_low_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="tactical_understanding_low"):
            PlayerTurningPointConfig(
                bench_streak_threshold=3,
                tactical_understanding_low=1.5,
                short_term_window=4,
            )

    def test_short_term_window_zero(self) -> None:
        with pytest.raises(ValueError, match="short_term_window"):
            PlayerTurningPointConfig(
                bench_streak_threshold=3,
                tactical_understanding_low=0.40,
                short_term_window=0,
            )


class TestManagerTurningPointValidation:
    def test_warning_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="job_security_warning"):
            ManagerTurningPointConfig(
                job_security_warning=1.5,
                job_security_critical=0.10,
                style_stubbornness_threshold=80,
            )

    def test_critical_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="job_security_critical"):
            ManagerTurningPointConfig(
                job_security_warning=0.30,
                job_security_critical=-0.1,
                style_stubbornness_threshold=80,
            )

    def test_critical_greater_than_warning(self) -> None:
        with pytest.raises(ValueError, match="job_security_critical.*<="):
            ManagerTurningPointConfig(
                job_security_warning=0.10,
                job_security_critical=0.30,
                style_stubbornness_threshold=80,
            )

    def test_negative_stubbornness_threshold(self) -> None:
        with pytest.raises(ValueError, match="style_stubbornness_threshold"):
            ManagerTurningPointConfig(
                job_security_warning=0.30,
                job_security_critical=0.10,
                style_stubbornness_threshold=-1,
            )
