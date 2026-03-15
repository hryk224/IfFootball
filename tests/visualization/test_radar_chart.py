"""Tests for radar chart rendering."""

from __future__ import annotations

import tempfile
from pathlib import Path

from iffootball.visualization.radar_chart import create_radar_figure, render_radar_chart
from iffootball.visualization.radar_data import RadarAxes, RadarChartData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_data() -> RadarChartData:
    return RadarChartData(
        branch_a=RadarAxes(
            xg_for=0.6,
            xg_against=0.5,
            ppda=0.55,
            possession=0.6,
            prog_passes=0.55,
        ),
        branch_b=RadarAxes(
            xg_for=0.45,
            xg_against=0.5,
            ppda=0.7,
            possession=0.65,
            prog_passes=0.6,
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCreateRadarFigure:
    def test_returns_figure(self) -> None:
        fig = create_radar_figure(_sample_data())
        assert fig is not None
        # Figure should have at least one axes (the polar plot).
        assert len(fig.axes) >= 1


class TestRenderRadarChart:
    def test_saves_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = render_radar_chart(
                _sample_data(), Path(tmpdir) / "test.png"
            )
            assert path.exists()
            assert path.suffix == ".png"
            assert path.stat().st_size > 0

    def test_saves_svg(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = render_radar_chart(
                _sample_data(), Path(tmpdir) / "test.svg"
            )
            assert path.exists()
            assert path.suffix == ".svg"
            assert path.stat().st_size > 0

    def test_creates_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = render_radar_chart(
                _sample_data(), Path(tmpdir) / "sub" / "dir" / "test.png"
            )
            assert path.exists()
