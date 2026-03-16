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

from iffootball.simulation.structured_explanation import StructuredExplanation


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
    """Controls what appears at the top of the summary.

    Attributes:
        lead_metric:       Primary metric to highlight (metric_name from highlights).
        lead_player:       Most impacted player name, or None if no impacts.
        lead_direction:    Overall direction of the lead metric change.
        secondary_metrics: Additional metric_names to mention after the lead.
    """

    lead_metric: str
    lead_player: str | None
    lead_direction: Literal["positive", "negative", "marginal"]
    secondary_metrics: tuple[str, ...]


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
        section_order:      Ordered section type values to include.
        expanded_step_ids:  CausalStep IDs to show in full.
        collapsed_step_ids: CausalStep IDs to collapse/summarize.
        featured_players:   Player names in display order.
        show_limitations_info: Whether to include info-severity limitations.
    """

    section_order: tuple[str, ...]
    expanded_step_ids: frozenset[str]
    collapsed_step_ids: frozenset[str]
    featured_players: tuple[str, ...]
    show_limitations_info: bool


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
    """

    sections: tuple[SectionPlan, ...]
    summary_priority: SummaryPriority
    player_display_order: tuple[str, ...]
    expanded_step_ids: frozenset[str]
    collapsed_step_ids: frozenset[str]
    limitation_placement: LimitationPlacement

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
        )


# ---------------------------------------------------------------------------
# Section templates per display context
# ---------------------------------------------------------------------------

_DEPTH_EXPAND_THRESHOLD = 2

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

    return ReportPlan(
        sections=sections,
        summary_priority=summary_priority,
        player_display_order=player_order,
        expanded_step_ids=expanded,
        collapsed_step_ids=collapsed,
        limitation_placement=limitation_placement,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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

    return SummaryPriority(
        lead_metric=lead_metric,
        lead_player=lead_player,
        lead_direction=lead_direction,
        secondary_metrics=secondary_metrics,
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
