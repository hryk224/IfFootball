"""Next scenario suggestions from comparison results.

Generates rule-based suggestions for what scenario to explore next,
based on signals from ComparisonResult and StructuredExplanation.
Unlike ValidationSignal (which asks "what to observe in this scenario"),
ScenarioSuggestion asks "what different scenario to try next".

Design principles:
    - Rule-based, no LLM. Same input produces same suggestions.
    - Max 1 suggestion per category, max 3 total.
    - Suggestions reference specific signals, not generic advice.
    - No concrete manager names; describe characteristics instead.
    - Output is rendered as a fixed Markdown section appended after
      the LLM-generated report (not mixed into LLM payload).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from iffootball.simulation.comparison import ComparisonResult
from iffootball.simulation.structured_explanation import StructuredExplanation

# Category priority for stable ordering (lower = higher priority).
_CATEGORY_ORDER: dict[str, int] = {
    "alternative_manager": 0,
    "trigger_timing": 1,
    "focus_player": 2,
}

# Thresholds for suggestion triggers.
_SMALL_POINTS_DIFF = 1.0
_HIGH_CASCADE_EVENT_THRESHOLD = 15.0
_DOMINANT_IMPACT_RATIO = 1.5  # top player impact / second player impact


@dataclass(frozen=True)
class ScenarioSuggestion:
    """A suggested next scenario to explore.

    Attributes:
        category:      Type of suggestion.
        suggestion:    What to try next (no concrete names).
        reason:        Why this is suggested (linked to data signal).
        signal_source: Which data point triggered this suggestion.
        priority:      Lower = higher priority. Used for ordering.
    """

    category: Literal["alternative_manager", "trigger_timing", "focus_player"]
    suggestion: str
    reason: str
    signal_source: str
    priority: int


def generate_scenario_suggestions(
    comparison: ComparisonResult,
    explanation: StructuredExplanation,
) -> tuple[ScenarioSuggestion, ...]:
    """Generate next-scenario suggestions from comparison results.

    Returns up to 3 suggestions (max 1 per category), ordered by
    priority then category order.

    Args:
        comparison:  A/B comparison result.
        explanation: Completed StructuredExplanation.

    Returns:
        Tuple of ScenarioSuggestion, ordered by priority.
    """
    candidates: list[ScenarioSuggestion] = []

    # --- alternative_manager ---
    points_diff = abs(comparison.delta.points_mean_diff)
    if points_diff < _SMALL_POINTS_DIFF:
        candidates.append(
            ScenarioSuggestion(
                category="alternative_manager",
                suggestion=(
                    "Consider comparing with a manager whose tactical "
                    "profile differs more strongly from the baseline — "
                    "for example, one with significantly higher or lower "
                    "pressing intensity."
                ),
                reason=(
                    f"The points difference is small "
                    f"({comparison.delta.points_mean_diff:+.1f}), "
                    f"suggesting the current incoming manager's profile "
                    f"is close to the baseline."
                ),
                signal_source=(
                    f"points_mean_diff={comparison.delta.points_mean_diff:+.1f}"
                ),
                priority=1,
            )
        )

    # --- trigger_timing ---
    cascade_b = comparison.with_change.cascade_event_counts
    confusion = cascade_b.get("tactical_confusion", 0.0)
    if confusion > _HIGH_CASCADE_EVENT_THRESHOLD:
        candidates.append(
            ScenarioSuggestion(
                category="trigger_timing",
                suggestion=(
                    "Try triggering the change at a different match week. "
                    "An earlier trigger gives more adaptation time; a later "
                    "trigger reduces the number of affected fixtures."
                ),
                reason=(
                    f"Early tactical confusion is prominent "
                    f"({confusion:.1f} events/run), suggesting the "
                    f"adaptation window is a significant factor."
                ),
                signal_source=f"tactical_confusion={confusion:.1f}/run",
                priority=2,
            )
        )

    # --- focus_player ---
    impacts = explanation.player_impacts
    if len(impacts) >= 2:
        top = impacts[0]
        second = impacts[1]
        if second.impact_score > 0 and (
            top.impact_score / second.impact_score >= _DOMINANT_IMPACT_RATIO
        ):
            candidates.append(
                ScenarioSuggestion(
                    category="focus_player",
                    suggestion=(
                        f"Explore a scenario involving {top.player_name} — "
                        f"for example, their absence (injury) or transfer. "
                        f"This would clarify whether the overall outcome "
                        f"depends heavily on this player."
                    ),
                    reason=(
                        f"{top.player_name} has a significantly higher "
                        f"impact score ({top.impact_score:.3f}) than the "
                        f"next player ({second.player_name}: "
                        f"{second.impact_score:.3f})."
                    ),
                    signal_source=(
                        f"impact_ratio="
                        f"{top.impact_score / second.impact_score:.2f}"
                    ),
                    priority=3,
                )
            )

    # Deduplicate by category (keep lowest priority per category).
    seen: set[str] = set()
    deduped: list[ScenarioSuggestion] = []
    # Sort by priority first, then category order for stability.
    candidates.sort(
        key=lambda s: (s.priority, _CATEGORY_ORDER.get(s.category, 99))
    )
    for s in candidates:
        if s.category not in seen:
            seen.add(s.category)
            deduped.append(s)

    return tuple(deduped[:3])


def render_suggestions_markdown(
    suggestions: tuple[ScenarioSuggestion, ...],
) -> str:
    """Render suggestions as a Markdown section.

    Returns empty string if no suggestions.
    """
    if not suggestions:
        return ""

    lines = ["## Next Steps", ""]
    for i, s in enumerate(suggestions, 1):
        lines.append(
            f"{i}. **{s.suggestion}**  "
        )
        lines.append(f"   *{s.reason}*")
        lines.append("")

    return "\n".join(lines)
