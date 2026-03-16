"""Report planner: decides what to show, in what order, at what detail level.

Takes a completed StructuredExplanation and produces a ReportPlan that
controls section ordering, player prioritization, causal chain expansion,
and limitation visibility. Does NOT create new assertions or analysis.

Pipeline position:
    StructuredExplanation (complete)
      -> plan_report()
      -> ReportPlan
      -> adapter uses ReportPlan to build ReportInput with DisplayHints

Design principles:
    - Rule-based only (no LLM). Reproducible for the same input.
    - Reads from StructuredExplanation exclusively. No external data sources.
    - Editing responsibility only: decides display order and detail level,
      never invents content.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from iffootball.simulation.structured_explanation import (
    DifferenceHighlight,
    StructuredExplanation,
    ValidationSignal,
)


# ---------------------------------------------------------------------------
# Display context
# ---------------------------------------------------------------------------


class DisplayContext(str, Enum):
    """Controls the level of detail shown in the report."""

    COMPACT = "compact"  # Summary + top 1-2 players + warning limitations
    STANDARD = "standard"  # Current output equivalent
    ANALYST = "analyst"  # Full expansion + info-severity limitations


# ---------------------------------------------------------------------------
# Section planning
# ---------------------------------------------------------------------------


class SectionType(str, Enum):
    """Report section identifiers."""

    SUMMARY = "summary"
    KEY_DIFFERENCES = "key_differences"
    CAUSAL_CHAIN = "causal_chain"
    PLAYER_IMPACT = "player_impact"
    LIMITATIONS = "limitations"
    WHAT_TO_WATCH = "what_to_watch"


class DetailLevel(str, Enum):
    """How much detail to show in a section."""

    BRIEF = "brief"
    NORMAL = "normal"
    FULL = "full"


@dataclass(frozen=True)
class SectionPlan:
    """Display plan for one report section.

    Attributes:
        section_type: Which section this plan describes.
        include:      Whether to include this section in the report.
        detail_level: How much detail to show.
    """

    section_type: SectionType
    include: bool
    detail_level: DetailLevel


# ---------------------------------------------------------------------------
# Summary priority
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SummaryPriority:
    """Controls what appears in the Summary section.

    Summary follows a 4-element structure:
        1. Trigger — what happened
        2. Outcome — direction and main number
        3. Trade-off — cost or side-effect (0-1 sentence)
        4. Takeaway — overall conclusion (0-1 sentence)

    Attributes:
        lead_metric:       Primary metric (usually "total_points_mean").
        lead_player:       Most impacted player name, or None.
        lead_direction:    Overall direction of the lead metric change.
        secondary_metrics: Metric names available for mention (not all go in Summary).
        tradeoff_metric:   Negative-direction metric to use for trade-off sentence.
                           None if no negative highlight exists.
        max_sentences:     Maximum sentences allowed in Summary.
    """

    lead_metric: str
    lead_player: str | None
    lead_direction: Literal["positive", "negative", "marginal"]
    secondary_metrics: tuple[str, ...]
    tradeoff_metric: str | None
    max_sentences: int


# ---------------------------------------------------------------------------
# Limitation placement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LimitationPlacement:
    """Controls limitation display visibility.

    Attributes:
        show_system:   Whether to display system-level limitations.
        show_scenario: Whether to display scenario-specific limitations.
        include_info:  Whether to include severity="info" items.
                       False means only severity="warning" items are shown.
    """

    show_system: bool
    show_scenario: bool
    include_info: bool


# ---------------------------------------------------------------------------
# Display hints (DTO for ReportInput)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DisplayHints:
    """Display instructions passed through ReportInput to report generation.

    This DTO preserves planner decisions through the adapter layer so
    that report generation (LLM or data-only) can respect them.

    Attributes:
        section_order:         Ordered section type values to include.
        expanded_step_ids:     CausalStep IDs to show in full.
        collapsed_step_ids:    CausalStep IDs to collapse/summarize.
        featured_players:      Player names in display order.
        show_limitations_info: Whether to include info-severity limitations.
        summary_max_sentences: Maximum sentences in Summary.
        summary_tradeoff_metric: Metric to use for trade-off sentence (or None).
    """

    section_order: tuple[str, ...]
    expanded_step_ids: frozenset[str]
    collapsed_step_ids: frozenset[str]
    featured_players: tuple[str, ...]
    show_limitations_info: bool
    summary_max_sentences: int = 4
    summary_tradeoff_metric: str | None = None


# ---------------------------------------------------------------------------
# Report plan (top-level output)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReportPlan:
    """Complete report plan produced by plan_report().

    Contains all display decisions needed to render the report.
    Can be used by the adapter to build ReportInput with DisplayHints,
    and by UI components for layout control.

    Attributes:
        sections:             Ordered section plans.
        summary_priority:     What to lead with in the summary.
        player_display_order: Player names in display order (by impact_score).
        expanded_step_ids:    CausalStep IDs to show expanded.
        collapsed_step_ids:   CausalStep IDs to show collapsed.
        limitation_placement: Limitation visibility settings.
        validation_signals:   Early observation points for hypothesis checking.
    """

    sections: tuple[SectionPlan, ...]
    summary_priority: SummaryPriority
    player_display_order: tuple[str, ...]
    expanded_step_ids: frozenset[str]
    collapsed_step_ids: frozenset[str]
    limitation_placement: LimitationPlacement
    validation_signals: tuple[ValidationSignal, ...] = ()

    def to_display_hints(self) -> DisplayHints:
        """Convert to DisplayHints DTO for ReportInput."""
        return DisplayHints(
            section_order=tuple(
                s.section_type.value for s in self.sections if s.include
            ),
            expanded_step_ids=self.expanded_step_ids,
            collapsed_step_ids=self.collapsed_step_ids,
            featured_players=self.player_display_order,
            show_limitations_info=self.limitation_placement.include_info,
            summary_max_sentences=self.summary_priority.max_sentences,
            summary_tradeoff_metric=self.summary_priority.tradeoff_metric,
        )


# ---------------------------------------------------------------------------
# Section templates per display context
# ---------------------------------------------------------------------------

_DEPTH_EXPAND_THRESHOLD = 2

# Event polarity: whether an increase in this metric is bad for the team.
# True = increase is bad (e.g. form_drop, tactical_confusion).
# False = increase is good (e.g. adaptation_progress).
# Missing = unknown, treat as neutral.
_INCREASE_IS_BAD: dict[str, bool] = {
    "form_drop": True,
    "tactical_confusion": True,
    "trust_decline": True,
    "squad_unrest": True,
    "playing_time_change": False,  # Neutral, not clearly bad.
    "adaptation_progress": False,
    "manager_tactical_shift": False,  # Neutral.
    "manager_dismissal": True,
    "total_points_mean": False,
}

_SECTION_TEMPLATES: dict[DisplayContext, list[SectionPlan]] = {
    DisplayContext.COMPACT: [
        SectionPlan(SectionType.SUMMARY, include=True, detail_level=DetailLevel.BRIEF),
        SectionPlan(
            SectionType.KEY_DIFFERENCES, include=True, detail_level=DetailLevel.BRIEF
        ),
        SectionPlan(
            SectionType.CAUSAL_CHAIN, include=False, detail_level=DetailLevel.BRIEF
        ),
        SectionPlan(
            SectionType.PLAYER_IMPACT, include=True, detail_level=DetailLevel.BRIEF
        ),
        SectionPlan(
            SectionType.LIMITATIONS, include=True, detail_level=DetailLevel.BRIEF
        ),
        SectionPlan(
            SectionType.WHAT_TO_WATCH, include=False, detail_level=DetailLevel.BRIEF
        ),
    ],
    DisplayContext.STANDARD: [
        SectionPlan(
            SectionType.SUMMARY, include=True, detail_level=DetailLevel.NORMAL
        ),
        SectionPlan(
            SectionType.KEY_DIFFERENCES, include=True, detail_level=DetailLevel.NORMAL
        ),
        SectionPlan(
            SectionType.CAUSAL_CHAIN, include=True, detail_level=DetailLevel.NORMAL
        ),
        SectionPlan(
            SectionType.PLAYER_IMPACT, include=True, detail_level=DetailLevel.NORMAL
        ),
        SectionPlan(
            SectionType.LIMITATIONS, include=True, detail_level=DetailLevel.NORMAL
        ),
        SectionPlan(
            SectionType.WHAT_TO_WATCH, include=True, detail_level=DetailLevel.NORMAL
        ),
    ],
    DisplayContext.ANALYST: [
        SectionPlan(SectionType.SUMMARY, include=True, detail_level=DetailLevel.FULL),
        SectionPlan(
            SectionType.KEY_DIFFERENCES, include=True, detail_level=DetailLevel.FULL
        ),
        SectionPlan(
            SectionType.CAUSAL_CHAIN, include=True, detail_level=DetailLevel.FULL
        ),
        SectionPlan(
            SectionType.PLAYER_IMPACT, include=True, detail_level=DetailLevel.FULL
        ),
        SectionPlan(
            SectionType.LIMITATIONS, include=True, detail_level=DetailLevel.FULL
        ),
        SectionPlan(
            SectionType.WHAT_TO_WATCH, include=True, detail_level=DetailLevel.FULL
        ),
    ],
}

# Maximum featured players per context.
_MAX_PLAYERS: dict[DisplayContext, int] = {
    DisplayContext.COMPACT: 1,
    DisplayContext.STANDARD: 3,
    DisplayContext.ANALYST: 999,  # All players
}

# Maximum secondary metrics in summary per context.
_MAX_SECONDARY_METRICS: dict[DisplayContext, int] = {
    DisplayContext.COMPACT: 1,
    DisplayContext.STANDARD: 2,
    DisplayContext.ANALYST: 5,
}

# Maximum sentences in Summary per context.
_MAX_SUMMARY_SENTENCES: dict[DisplayContext, int] = {
    DisplayContext.COMPACT: 2,   # Trigger + Outcome only
    DisplayContext.STANDARD: 4,  # Trigger + Outcome + Trade-off + Takeaway
    DisplayContext.ANALYST: 5,   # Standard + optional extra
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plan_report(
    explanation: StructuredExplanation,
    display_context: DisplayContext = DisplayContext.STANDARD,
) -> ReportPlan:
    """Build a ReportPlan from a completed StructuredExplanation.

    All decisions are rule-based and reproducible for the same input.

    Args:
        explanation:     Completed StructuredExplanation (statements filled).
        display_context: Controls detail level (compact/standard/analyst).

    Returns:
        ReportPlan with all display decisions.
    """
    sections = tuple(_SECTION_TEMPLATES[display_context])

    summary_priority = _build_summary_priority(explanation, display_context)
    player_order = _build_player_order(explanation, display_context)
    expanded, collapsed = _classify_steps(explanation, display_context)
    limitation_placement = _build_limitation_placement(display_context)

    # Generate validation signals if the section is included.
    signals: tuple[ValidationSignal, ...] = ()
    show_signals = any(
        s.section_type == SectionType.WHAT_TO_WATCH and s.include
        for s in sections
    )
    if show_signals:
        from iffootball.simulation.validation_signals import (
            generate_validation_signals,
        )

        signals = generate_validation_signals(explanation)

    return ReportPlan(
        sections=sections,
        summary_priority=summary_priority,
        player_display_order=player_order,
        expanded_step_ids=expanded,
        collapsed_step_ids=collapsed,
        limitation_placement=limitation_placement,
        validation_signals=signals,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_negative_change(metric_name: str, diff: float) -> bool:
    """Determine if a metric change is bad for the team.

    Uses the polarity table: if increase_is_bad and diff > 0, it's negative.
    If increase_is_bad is False and diff < 0, it's also negative (good thing decreased).
    Unknown metrics are treated as non-negative.
    """
    polarity = _INCREASE_IS_BAD.get(metric_name)
    if polarity is None:
        return False
    if polarity and diff > 0.01:
        return True  # Bad thing increased.
    if not polarity and diff < -0.01:
        return True  # Good thing decreased.
    return False


def _select_tradeoff_metric(
    highlights: tuple[DifferenceHighlight, ...],
    lead_metric: str,
) -> str | None:
    """Select the most significant negative-connotation highlight for trade-off.

    Picks the non-lead highlight with the largest absolute diff whose
    change is bad for the team (per polarity table). Returns None if
    no negative-connotation highlight exists.
    """
    candidates: list[tuple[float, str]] = []
    for hl in highlights:
        if hl.metric_name == lead_metric:
            continue
        if _is_negative_change(hl.metric_name, hl.diff):
            candidates.append((abs(hl.diff), hl.metric_name))

    if not candidates:
        return None
    # Sort by absolute diff descending, pick top.
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _build_summary_priority(
    explanation: StructuredExplanation,
    context: DisplayContext,
) -> SummaryPriority:
    """Determine what to highlight in the summary section."""
    # Lead metric: first highlight (always total_points_mean by convention).
    lead_metric = "total_points_mean"
    lead_diff = 0.0
    if explanation.highlights:
        lead_metric = explanation.highlights[0].metric_name
        lead_diff = explanation.highlights[0].diff

    # Direction from lead metric diff.
    if lead_diff > 0.5:
        lead_direction: Literal["positive", "negative", "marginal"] = "positive"
    elif lead_diff < -0.5:
        lead_direction = "negative"
    else:
        lead_direction = "marginal"

    # Lead player: highest impact_score.
    lead_player: str | None = None
    if explanation.player_impacts:
        lead_player = explanation.player_impacts[0].player_name

    # Secondary metrics: remaining highlights sorted by absolute diff.
    max_secondary = _MAX_SECONDARY_METRICS[context]
    secondary = sorted(
        explanation.highlights[1:],
        key=lambda h: abs(h.diff),
        reverse=True,
    )
    secondary_metrics = tuple(h.metric_name for h in secondary[:max_secondary])

    # Trade-off: pick the most significant negative-connotation highlight.
    # A "trade-off" is a metric whose change is bad for the team.
    # This requires knowing whether an increase or decrease in each metric
    # is negative. Use a polarity table for known event types.
    tradeoff_metric = _select_tradeoff_metric(explanation.highlights, lead_metric)

    # Max sentences by context.
    max_sentences = _MAX_SUMMARY_SENTENCES[context]

    return SummaryPriority(
        lead_metric=lead_metric,
        lead_player=lead_player,
        lead_direction=lead_direction,
        secondary_metrics=secondary_metrics,
        tradeoff_metric=tradeoff_metric,
        max_sentences=max_sentences,
    )


def _build_player_order(
    explanation: StructuredExplanation,
    context: DisplayContext,
) -> tuple[str, ...]:
    """Determine player display order, limited by context."""
    max_players = _MAX_PLAYERS[context]
    return tuple(
        pi.player_name for pi in explanation.player_impacts[:max_players]
    )


def _classify_steps(
    explanation: StructuredExplanation,
    context: DisplayContext,
) -> tuple[frozenset[str], frozenset[str]]:
    """Classify causal steps into expanded vs collapsed sets.

    Rules:
        - analyst: all expanded
        - standard: depth <= 2 expanded, depth >= 3 collapsed
        - compact: all collapsed (causal chain section hidden anyway)
    """
    if context == DisplayContext.ANALYST:
        all_ids = frozenset(s.step_id for s in explanation.causal_chain)
        return all_ids, frozenset()

    if context == DisplayContext.COMPACT:
        all_ids = frozenset(s.step_id for s in explanation.causal_chain)
        return frozenset(), all_ids

    # Standard: split by depth.
    expanded = frozenset(
        s.step_id
        for s in explanation.causal_chain
        if s.depth <= _DEPTH_EXPAND_THRESHOLD
    )
    collapsed = frozenset(
        s.step_id
        for s in explanation.causal_chain
        if s.depth > _DEPTH_EXPAND_THRESHOLD
    )
    return expanded, collapsed


def _build_limitation_placement(
    context: DisplayContext,
) -> LimitationPlacement:
    """Determine limitation visibility for the display context."""
    return LimitationPlacement(
        show_system=True,
        show_scenario=True,
        include_info=context == DisplayContext.ANALYST,
    )
