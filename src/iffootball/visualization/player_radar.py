"""Player impact radar chart rendering.

Renders per-player radar charts comparing dynamic state between Branch A
and Branch B. Each chart has 4 axes (all 0.0–1.0 scale):
    Form, Fatigue (inverted), Tactical Understanding, Manager Trust.

Fatigue is inverted so that "outer = better" holds for all axes,
consistent with the team radar chart convention.

Usage:
    from iffootball.visualization.player_impact import rank_player_impact
    from iffootball.visualization.player_radar import render_player_radars

    impacts = rank_player_impact(comparison, top_n=5)
    paths = render_player_radars(impacts, output_dir="output/players")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from iffootball.visualization.player_impact import PlayerImpact  # noqa: E402

# Axis configuration.
_AXIS_LABELS = ("Form", "Fatigue", "Tactical\nUnderstanding", "Manager\nTrust")
_N_AXES = len(_AXIS_LABELS)

# Style constants.
_BRANCH_A_COLOR = "#2563eb"
_BRANCH_B_COLOR = "#dc2626"
_FILL_ALPHA = 0.25
_LINE_WIDTH = 2.0
_FIGURE_SIZE = (4, 4)


def _extract_values(impact: PlayerImpact, branch: str) -> list[float]:
    """Extract radar values for a branch, inverting fatigue.

    Returns [form, 1-fatigue, understanding, trust] — all 0.0–1.0
    where higher = better.
    """
    if branch == "a":
        return [
            impact.mean_form_a,
            1.0 - impact.mean_fatigue_a,  # invert: low fatigue = good
            impact.mean_understanding_a,
            impact.mean_trust_a,
        ]
    return [
        impact.mean_form_b,
        1.0 - impact.mean_fatigue_b,
        impact.mean_understanding_b,
        impact.mean_trust_b,
    ]


def create_player_radar_figure(impact: PlayerImpact) -> Figure:
    """Create a radar chart figure for a single player.

    Args:
        impact: PlayerImpact with mean dynamic state for both branches.

    Returns:
        matplotlib Figure with the radar chart.
    """
    angles = np.linspace(0, 2 * np.pi, _N_AXES, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    values_a = _extract_values(impact, "a")
    values_b = _extract_values(impact, "b")
    values_a_closed = values_a + [values_a[0]]
    values_b_closed = values_b + [values_b[0]]

    fig, ax = plt.subplots(
        figsize=_FIGURE_SIZE, subplot_kw={"projection": "polar"}
    )

    # Branch A.
    ax.plot(
        angles_closed,
        values_a_closed,
        color=_BRANCH_A_COLOR,
        linewidth=_LINE_WIDTH,
        label="Branch A (no change)",
    )
    ax.fill(angles_closed, values_a_closed, color=_BRANCH_A_COLOR, alpha=_FILL_ALPHA)

    # Branch B.
    ax.plot(
        angles_closed,
        values_b_closed,
        color=_BRANCH_B_COLOR,
        linewidth=_LINE_WIDTH,
        label="Branch B (with change)",
    )
    ax.fill(angles_closed, values_b_closed, color=_BRANCH_B_COLOR, alpha=_FILL_ALPHA)

    ax.set_xticks(angles)
    ax.set_xticklabels(list(_AXIS_LABELS), fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8, alpha=0.6)

    title = f"{impact.player_name} (impact: {impact.impact_score:.3f})"
    ax.set_title(title, fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    fig.tight_layout()
    return fig


def render_player_radars(
    impacts: list[PlayerImpact],
    output_dir: str | Path,
    *,
    fmt: str = "png",
    dpi: int = 150,
) -> list[Path]:
    """Render radar charts for a list of impacted players.

    Creates one file per player in the output directory, named by
    player_id for uniqueness.

    Args:
        impacts:    List of PlayerImpact (typically from rank_player_impact).
        output_dir: Directory to save chart files.
        fmt:        Output format ("png" or "svg").
        dpi:        Resolution for raster output.

    Returns:
        List of Paths to saved chart files, in the same order as impacts.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for impact in impacts:
        fig = create_player_radar_figure(impact)
        filename = f"player_{impact.player_id}.{fmt}"
        filepath = out / filename
        fig.savefig(str(filepath), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(filepath.resolve())

    return paths
