"""Report-level risk audit.

Scans the overall comparison and explanation for risks that could
mislead the reader. Unlike LimitationsDisclosure (structural model
constraints) or HypothesisCritique (per-hypothesis weaknesses), this
module performs a cross-cutting audit of the final report's reliability.

Design principles:
    - Rule-based, no LLM. Same input produces same flags.
    - Max 1 flag per category, max 3 total.
    - Each flag is tied to a specific signal, not generic advice.
    - Output is rendered as a fixed Markdown section appended after
      the LLM-generated report, before What to Watch / Next Steps.

Risk categories:
    overstatement:    Points diff is small relative to run-to-run variance.
    unstable_basis:   High cascade activity makes results threshold-sensitive.
    data_reliability: A featured player has partial-tier (fallback) data.

Severity thresholds:
    overstatement:    |diff|/std < 0.2 -> high, < 0.5 -> medium
    unstable_basis:   form_drop+trust_decline > 80/run -> high, > 40 -> medium
    data_reliability: partial in top 1 -> high, in top 3 -> medium
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from iffootball.agents.player import SampleTier
from iffootball.simulation.comparison import ComparisonResult
from iffootball.simulation.structured_explanation import StructuredExplanation

_MAX_FLAGS = 3

# Severity icons for markdown rendering.
_SEVERITY_ICON = {"high": "\u26a0\ufe0f", "medium": "\u26a0"}


@dataclass(frozen=True)
class RiskFlag:
    """A risk identified in the overall report.

    Attributes:
        category:      Type of risk.
        description:   Human-readable explanation.
        signal_source: Which data point triggered this flag.
        severity:      "high" (may mislead) or "medium" (caution).
    """

    category: Literal["overstatement", "unstable_basis", "data_reliability"]
    description: str
    signal_source: str
    severity: Literal["high", "medium"]


# Category priority for stable ordering (lower = higher priority).
_CATEGORY_ORDER: dict[str, int] = {
    "overstatement": 0,
    "unstable_basis": 1,
    "data_reliability": 2,
}

# Severity ordering (high before medium).
_SEVERITY_ORDER: dict[str, int] = {"high": 0, "medium": 1}


def generate_risk_audit(
    comparison: ComparisonResult,
    explanation: StructuredExplanation,
) -> tuple[RiskFlag, ...]:
    """Generate risk flags from comparison and explanation.

    Returns up to _MAX_FLAGS flags (max 1 per category), ordered by
    severity then category.

    Args:
        comparison:  A/B comparison result.
        explanation: Completed StructuredExplanation.

    Returns:
        Tuple of RiskFlag, ordered by severity then category.
    """
    candidates: list[RiskFlag] = []

    # --- overstatement ---
    points_diff = comparison.delta.points_mean_diff
    points_std = comparison.no_change.total_points_std
    if points_std > 0:
        ratio = abs(points_diff) / points_std
        if ratio < 0.2:
            candidates.append(
                RiskFlag(
                    category="overstatement",
                    description=(
                        f"The points difference ({points_diff:+.1f}) is very "
                        f"small relative to run-to-run variance (std {points_std:.1f}). "
                        f"Any directional conclusion from this comparison is unreliable."
                    ),
                    signal_source=f"|diff|/std={ratio:.2f}",
                    severity="high",
                )
            )
        elif ratio < 0.5:
            candidates.append(
                RiskFlag(
                    category="overstatement",
                    description=(
                        f"The points difference ({points_diff:+.1f}) is modest "
                        f"relative to variance (std {points_std:.1f}). "
                        f"Interpret directional claims with caution."
                    ),
                    signal_source=f"|diff|/std={ratio:.2f}",
                    severity="medium",
                )
            )

    # --- unstable_basis ---
    cascade_b = comparison.with_change.cascade_event_counts
    form_drop = cascade_b.get("form_drop", 0.0)
    trust_decline = cascade_b.get("trust_decline", 0.0)
    cascade_total = form_drop + trust_decline
    if cascade_total > 80:
        candidates.append(
            RiskFlag(
                category="unstable_basis",
                description=(
                    f"High cascade activity (form_drop + trust_decline = "
                    f"{cascade_total:.0f}/run) indicates the results are "
                    f"sensitive to turning point thresholds and action "
                    f"distribution settings."
                ),
                signal_source=f"form_drop+trust_decline={cascade_total:.0f}/run",
                severity="high",
            )
        )
    elif cascade_total > 40:
        candidates.append(
            RiskFlag(
                category="unstable_basis",
                description=(
                    f"Moderate cascade activity (form_drop + trust_decline = "
                    f"{cascade_total:.0f}/run) suggests some sensitivity to "
                    f"turning point configuration."
                ),
                signal_source=f"form_drop+trust_decline={cascade_total:.0f}/run",
                severity="medium",
            )
        )

    # --- data_reliability ---
    impacts = explanation.player_impacts
    if impacts:
        top1_partial = impacts[0].sample_tier == SampleTier.PARTIAL
        top3_partial = any(
            pi.sample_tier == SampleTier.PARTIAL for pi in impacts[:3]
        )
        if top1_partial:
            candidates.append(
                RiskFlag(
                    category="data_reliability",
                    description=(
                        f"The top-impact player ({impacts[0].player_name}) "
                        f"has limited playing time data (partial tier). "
                        f"Their attribute estimates are fallback values, "
                        f"not full statistics."
                    ),
                    signal_source=(
                        f"top_player={impacts[0].player_name}, "
                        f"tier=partial"
                    ),
                    severity="high",
                )
            )
        elif top3_partial:
            partial_names = [
                pi.player_name
                for pi in impacts[:3]
                if pi.sample_tier == SampleTier.PARTIAL
            ]
            candidates.append(
                RiskFlag(
                    category="data_reliability",
                    description=(
                        f"Featured players with limited data: "
                        f"{', '.join(partial_names)}. Their impact "
                        f"rankings may be unreliable."
                    ),
                    signal_source=f"partial_in_top3={partial_names}",
                    severity="medium",
                )
            )

    # Deduplicate by category, sort by severity then category.
    seen: set[str] = set()
    deduped: list[RiskFlag] = []
    candidates.sort(
        key=lambda f: (
            _SEVERITY_ORDER.get(f.severity, 99),
            _CATEGORY_ORDER.get(f.category, 99),
        )
    )
    for f in candidates:
        if f.category not in seen:
            seen.add(f.category)
            deduped.append(f)

    return tuple(deduped[:_MAX_FLAGS])


def render_risk_audit_markdown(
    flags: tuple[RiskFlag, ...],
) -> str:
    """Render risk flags as a Markdown section.

    Returns empty string if no flags.
    """
    if not flags:
        return ""

    lines = ["## Risk Review", ""]
    for f in flags:
        icon = _SEVERITY_ICON.get(f.severity, "")
        lines.append(f"- {icon} **{f.category}**: {f.description}")
        lines.append(f"  *Signal: {f.signal_source}*")
        lines.append("")

    return "\n".join(lines)
