"""Radar chart rendering for Branch A/B team comparison.

Renders a polar radar chart overlaying two branches (no_change vs
with_change) using normalized 0.0–1.0 axes. Axes where lower raw
values are better (PPDA, xGA/90) are pre-inverted by radar_data so
that "outer = better" holds for all axes.

Output formats: PNG and SVG via matplotlib.

Usage:
    from iffootball.visualization.radar_chart import render_radar_chart
    from iffootball.visualization.radar_data import extract_radar_data

    data = extract_radar_data(comparison, baseline, mgr_a, mgr_b, league)
    render_radar_chart(data, output_path="output/team_radar.png")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend for file output.

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from iffootball.visualization.radar_data import RadarChartData  # noqa: E402

# Style constants.
_BRANCH_A_COLOR = "#2563eb"  # Blue
_BRANCH_B_COLOR = "#dc2626"  # Red
_BRANCH_A_ALPHA = 0.25
_BRANCH_B_ALPHA = 0.25
_LINE_WIDTH = 2.0
_FIGURE_SIZE = (8, 8)
_TITLE_FONTSIZE = 14
_LABEL_FONTSIZE = 10


def create_radar_figure(
    data: RadarChartData,
    *,
    title: str = "Team Comparison: Branch A vs Branch B",
) -> Figure:
    """Create a matplotlib Figure with the radar chart.

    Args:
        data:  RadarChartData with normalized Branch A/B axes.
        title: Chart title.

    Returns:
        matplotlib Figure object (not yet saved).
    """
    labels = list(data.labels)
    n_axes = len(labels)

    # Compute angle for each axis.
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()

    # Close the polygon by repeating the first value.
    values_a = list(data.branch_a.values()) + [data.branch_a.values()[0]]
    values_b = list(data.branch_b.values()) + [data.branch_b.values()[0]]
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(
        figsize=_FIGURE_SIZE, subplot_kw={"projection": "polar"}
    )

    # Branch A (no change).
    ax.plot(
        angles_closed,
        values_a,
        color=_BRANCH_A_COLOR,
        linewidth=_LINE_WIDTH,
        label="Branch A (no change)",
    )
    ax.fill(angles_closed, values_a, color=_BRANCH_A_COLOR, alpha=_BRANCH_A_ALPHA)

    # Branch B (with change).
    ax.plot(
        angles_closed,
        values_b,
        color=_BRANCH_B_COLOR,
        linewidth=_LINE_WIDTH,
        label="Branch B (with change)",
    )
    ax.fill(angles_closed, values_b, color=_BRANCH_B_COLOR, alpha=_BRANCH_B_ALPHA)

    # Axis labels.
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=_LABEL_FONTSIZE)

    # Radial ticks (0.0 to 1.0).
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8, alpha=0.6)

    ax.set_title(title, fontsize=_TITLE_FONTSIZE, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    # Caption noting data source classification.
    fig.text(
        0.5,
        0.02,
        "xG/90: simulation output | xGA/90: fixed baseline (current model) | "
        "PPDA, Possession, Prog Passes: tactical estimates",
        ha="center",
        fontsize=8,
        alpha=0.6,
        style="italic",
    )

    fig.tight_layout(rect=(0, 0.05, 1, 1))

    return fig


def render_radar_chart(
    data: RadarChartData,
    output_path: str | Path,
    *,
    title: str = "Team Comparison: Branch A vs Branch B",
    dpi: int = 150,
) -> Path:
    """Render and save a radar chart to a file.

    Output format is determined by the file extension (.png or .svg).

    Args:
        data:        RadarChartData with normalized Branch A/B axes.
        output_path: Destination file path.
        title:       Chart title.
        dpi:         Resolution for raster output (PNG).

    Returns:
        Resolved Path of the saved file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = create_radar_figure(data, title=title)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return path.resolve()
