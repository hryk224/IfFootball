"""Tests for player radar chart rendering."""

from __future__ import annotations

import tempfile
from pathlib import Path

from iffootball.visualization.player_impact import PlayerImpact
from iffootball.visualization.player_radar import (
    create_player_radar_figure,
    render_player_radars,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_impact() -> PlayerImpact:
    return PlayerImpact(
        player_id=7,
        player_name="Test Player",
        impact_score=0.42,
        mean_form_a=0.55,
        mean_form_b=0.40,
        mean_fatigue_a=0.15,
        mean_fatigue_b=0.30,
        mean_understanding_a=0.60,
        mean_understanding_b=0.30,
        mean_trust_a=0.55,
        mean_trust_b=0.45,
    )


def _sample_impacts() -> list[PlayerImpact]:
    return [
        _sample_impact(),
        PlayerImpact(
            player_id=10,
            player_name="Another Player",
            impact_score=0.25,
            mean_form_a=0.50,
            mean_form_b=0.45,
            mean_fatigue_a=0.10,
            mean_fatigue_b=0.20,
            mean_understanding_a=0.55,
            mean_understanding_b=0.35,
            mean_trust_a=0.50,
            mean_trust_b=0.50,
        ),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCreatePlayerRadarFigure:
    def test_returns_figure(self) -> None:
        fig = create_player_radar_figure(_sample_impact())
        assert fig is not None
        assert len(fig.axes) >= 1

    def test_title_contains_player_name(self) -> None:
        fig = create_player_radar_figure(_sample_impact())
        title = fig.axes[0].get_title()
        assert "Test Player" in title


class TestRenderPlayerRadars:
    def test_saves_files_for_each_player(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = render_player_radars(_sample_impacts(), tmpdir)
            assert len(paths) == 2
            for p in paths:
                assert p.exists()
                assert p.stat().st_size > 0

    def test_filenames_use_player_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = render_player_radars(_sample_impacts(), tmpdir)
            names = {p.name for p in paths}
            assert "player_7.png" in names
            assert "player_10.png" in names

    def test_svg_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = render_player_radars(
                [_sample_impact()], tmpdir, fmt="svg"
            )
            assert paths[0].suffix == ".svg"
            assert paths[0].exists()

    def test_empty_list_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = render_player_radars([], tmpdir)
            assert paths == []

    def test_creates_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "sub" / "dir"
            paths = render_player_radars([_sample_impact()], nested)
            assert len(paths) == 1
            assert paths[0].exists()
