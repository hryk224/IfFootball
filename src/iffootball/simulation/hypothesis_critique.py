"""Hypothesis generation with mandatory critique.

Extracts hypothesis candidates from comparison results and structured
explanations, then attaches critique points to each. Hypotheses are
never emitted without at least one critique — this prevents the module
from becoming a standalone narrative generator.

Design principles:
    - Rule-based, no LLM. Same input produces same output.
    - Hypotheses are "cause candidates" at the causal-chain level.
    - Each hypothesis inherits a label (data/analysis/hypothesis) from
      its strongest supporting evidence. When sources mix, the weakest
      label is adopted (conservative: data < analysis < hypothesis).
    - Critiques aim for 3 aspects (refutability, missing_data,
      alternative) but require at least 1. Not all aspects can always
      be filled meaningfully by rules alone.
    - Output is rendered as a fixed Markdown section appended after
      the LLM-generated report (not mixed into LLM payload).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from iffootball.simulation.comparison import ComparisonResult
from iffootball.simulation.structured_explanation import (
    EvidenceLabel,
    StructuredExplanation,
)

# Max hypotheses to generate.
_MAX_HYPOTHESES = 3

# Thresholds.
_SMALL_POINTS_DIFF = 1.5
_HIGH_CASCADE_THRESHOLD = 15.0
_DOMINANT_IMPACT_RATIO = 1.3

# Label strength ordering (lower index = stronger evidence).
_LABEL_STRENGTH: dict[EvidenceLabel, int] = {
    "data": 0,
    "analysis": 1,
    "hypothesis": 2,
}


def _weakest_label(*labels: EvidenceLabel) -> EvidenceLabel:
    """Return the weakest (most uncertain) label from the inputs."""
    return max(labels, key=lambda l: _LABEL_STRENGTH[l])


@dataclass(frozen=True)
class CritiquePoint:
    """A single criticism of a hypothesis.

    Attributes:
        aspect: Type of critique.
        point:  Natural language description of the criticism.
    """

    aspect: Literal["refutability", "missing_data", "alternative"]
    point: str


@dataclass(frozen=True)
class HypothesisCritique:
    """A hypothesis paired with its criticisms.

    Invariant: critiques is never empty.

    Attributes:
        hypothesis_id:       Unique identifier (e.g. "hyp-001").
        claim:               What the hypothesis asserts.
        supporting_evidence: Which signal supports it.
        label:               Inherited from source evidence (weakest if mixed).
        related_step_ids:    Links to CausalStep IDs.
        critiques:           At least 1 critique point.
    """

    hypothesis_id: str
    claim: str
    supporting_evidence: str
    label: EvidenceLabel
    related_step_ids: tuple[str, ...]
    critiques: tuple[CritiquePoint, ...]

    def __post_init__(self) -> None:
        if not self.critiques:
            raise ValueError(
                f"HypothesisCritique {self.hypothesis_id} must have "
                f"at least one critique point."
            )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_hypothesis_critiques(
    comparison: ComparisonResult,
    explanation: StructuredExplanation,
) -> tuple[HypothesisCritique, ...]:
    """Generate hypothesis-critique pairs from comparison and explanation.

    Returns up to _MAX_HYPOTHESES pairs, each with at least 1 critique.

    Args:
        comparison:  A/B comparison result (provides points diff, cascade counts).
        explanation: Completed StructuredExplanation (provides causal chain,
                     player impacts, highlights).

    Returns:
        Tuple of HypothesisCritique, ordered by hypothesis_id.
    """
    results: list[HypothesisCritique] = []
    hyp_idx = 0

    # --- Hypothesis 1: Main diff driver ---
    points_diff = comparison.delta.points_mean_diff
    points_std_a = comparison.no_change.total_points_std
    if explanation.highlights:
        top_highlight = explanation.highlights[0]
        hyp_idx += 1
        critiques: list[CritiquePoint] = []

        critiques.append(
            CritiquePoint(
                aspect="refutability",
                point=(
                    f"If a manager with a similar profile produces a "
                    f"different points diff, this metric is not the "
                    f"primary driver."
                ),
            )
        )
        if points_std_a > 0 and abs(points_diff) < points_std_a:
            critiques.append(
                CritiquePoint(
                    aspect="alternative",
                    point=(
                        f"The diff ({points_diff:+.1f}) is within one "
                        f"standard deviation ({points_std_a:.1f}), so "
                        f"fixture randomness could explain the result."
                    ),
                )
            )
        critiques.append(
            CritiquePoint(
                aspect="missing_data",
                point=(
                    "Per-match breakdown is not available in the "
                    "aggregated comparison; the trend within the "
                    "fixture sequence is unknown."
                ),
            )
        )

        # Label: highlights come from simulation_output (data) but the
        # claim interprets them (analysis).
        label = _weakest_label("data", "analysis")

        results.append(
            HypothesisCritique(
                hypothesis_id=f"hyp-{hyp_idx:03d}",
                claim=(
                    f"The primary driver of the "
                    f"{points_diff:+.1f} points difference is "
                    f"{top_highlight.metric_name}."
                ),
                supporting_evidence=(
                    f"{top_highlight.metric_name}: "
                    f"diff={top_highlight.diff:+.1f}"
                ),
                label=label,
                related_step_ids=(),
                critiques=tuple(critiques),
            )
        )

    # --- Hypothesis 2: Cascade-driven suppression ---
    cascade_b = comparison.with_change.cascade_event_counts
    confusion = cascade_b.get("tactical_confusion", 0.0)
    form_drop = cascade_b.get("form_drop", 0.0)
    if confusion > _HIGH_CASCADE_THRESHOLD or form_drop > _HIGH_CASCADE_THRESHOLD:
        dominant_event = "tactical_confusion" if confusion >= form_drop else "form_drop"
        dominant_count = max(confusion, form_drop)
        hyp_idx += 1
        critiques = []

        # Find related step IDs.
        related = tuple(
            s.step_id
            for s in explanation.causal_chain
            if s.event_type == dominant_event
        )[:3]

        critiques.append(
            CritiquePoint(
                aspect="refutability",
                point=(
                    f"If {dominant_event} events are reduced (e.g. by "
                    f"adjusting thresholds) without changing the points "
                    f"outcome, the cascade is noise rather than signal."
                ),
            )
        )
        critiques.append(
            CritiquePoint(
                aspect="missing_data",
                point=(
                    "The action distribution at turning points is "
                    "rule-based with fixed probabilities; actual player "
                    "behaviour may differ."
                ),
            )
        )

        results.append(
            HypothesisCritique(
                hypothesis_id=f"hyp-{hyp_idx:03d}",
                claim=(
                    f"High {dominant_event} rate ({dominant_count:.0f}/run) "
                    f"suppresses the team's performance in the "
                    f"transition period."
                ),
                supporting_evidence=(
                    f"{dominant_event}={dominant_count:.1f}/run "
                    f"in Branch B"
                ),
                label="analysis",
                related_step_ids=related,
                critiques=tuple(critiques),
            )
        )

    # --- Hypothesis 3: Player concentration ---
    impacts = explanation.player_impacts
    if len(impacts) >= 2:
        top = impacts[0]
        second = impacts[1]
        if second.impact_score > 0 and (
            top.impact_score / second.impact_score >= _DOMINANT_IMPACT_RATIO
        ):
            hyp_idx += 1
            critiques = []

            critiques.append(
                CritiquePoint(
                    aspect="refutability",
                    point=(
                        f"If {top.player_name} is removed from the "
                        f"scenario (e.g. injury trigger) and the overall "
                        f"outcome barely changes, their dominance in the "
                        f"ranking is misleading."
                    ),
                )
            )
            if top.sample_tier.value == "partial":
                critiques.append(
                    CritiquePoint(
                        aspect="missing_data",
                        point=(
                            f"{top.player_name} has limited playing time "
                            f"data (partial tier); their attributes are "
                            f"fallback estimates, not full statistics."
                        ),
                    )
                )

            # Label depends on whether the top player is full or partial.
            label = "analysis" if top.sample_tier.value == "full" else "hypothesis"

            results.append(
                HypothesisCritique(
                    hypothesis_id=f"hyp-{hyp_idx:03d}",
                    claim=(
                        f"{top.player_name}'s state changes dominate "
                        f"the impact ranking (score {top.impact_score:.3f} "
                        f"vs next {second.impact_score:.3f})."
                    ),
                    supporting_evidence=(
                        f"impact_ratio="
                        f"{top.impact_score / second.impact_score:.2f}"
                    ),
                    label=label,
                    related_step_ids=tuple(top.related_step_ids),
                    critiques=tuple(critiques),
                )
            )

    return tuple(results[:_MAX_HYPOTHESES])


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_critiques_markdown(
    critiques: tuple[HypothesisCritique, ...],
) -> str:
    """Render hypothesis-critique pairs as a Markdown section.

    Returns empty string if no critiques.
    """
    if not critiques:
        return ""

    lines = ["## Hypothesis Review", ""]
    for hc in critiques:
        lines.append(f"### {hc.claim}")
        lines.append(f"*[{hc.label}] {hc.supporting_evidence}*")
        lines.append("")
        for cp in hc.critiques:
            lines.append(f"- **{cp.aspect}**: {cp.point}")
        lines.append("")

    return "\n".join(lines)
