"""Thin adapter: StructuredExplanation -> ReportInput.

Bridges the new structured explanation schema to the existing
report_generation pipeline. This allows gradual migration without
modifying the current generate_report() or its prompt.

This adapter is temporary — once report_generation is updated to
consume StructuredExplanation directly, this module can be removed.

Semantic boundaries preserved:
    - ReportInput.limitations = system-level constraints (DEFAULT_LIMITATIONS).
      Not scenario-specific confidence_notes. Caller must provide these.
    - ReportInput.action_explanations = turning point actions from
      ActionExplanationEntry, not CausalStep data. The adapter leaves
      this empty because CausalStep does not map cleanly to the
      existing tp_type/action/explanation contract.
"""

from __future__ import annotations

from iffootball.llm.report_generation import (
    PlayerImpactEntry,
    ReportInput,
)
from iffootball.simulation.report_planner import ReportPlan
from iffootball.simulation.structured_explanation import StructuredExplanation


def structured_to_report_input(
    explanation: StructuredExplanation,
    *,
    plan: ReportPlan | None = None,
    limitations: list[str] | None = None,
    n_runs: int = 1,
    lang: str = "en",
) -> ReportInput:
    """Convert a StructuredExplanation to a ReportInput.

    Extracts the data needed by the existing report generation pipeline
    from the structured explanation.

    When a ReportPlan is provided, its decisions are preserved as
    DisplayHints in the returned ReportInput. The plan also controls:
        - Player ordering and count (featured_players).
        - Limitation visibility (include_info filter).

    Semantic notes:
        - action_explanations is always empty. CausalStep data does not
          map to the existing ActionExplanationEntry contract (tp_type /
          sampled action / explanation). The existing prompt handles
          empty action_explanations gracefully.
        - limitations merges system limitations (always shown) and
          scenario limitations with severity="warning". System
          limitations come from the StructuredExplanation, not
          DEFAULT_LIMITATIONS. Caller can override via the limitations
          parameter.

    Args:
        explanation: Completed StructuredExplanation.
        plan:        ReportPlan from plan_report(). If provided,
                     DisplayHints are included and player/limitation
                     filtering is applied.
        limitations: Override limitations list. If None, extracts
                     from StructuredExplanation.limitations.
        n_runs:      Number of simulation runs (not stored in
                     StructuredExplanation; caller must provide).
        lang:        Language for limitation messages ("en" or "ja").

    Returns:
        ReportInput compatible with generate_report().
    """
    scenario = explanation.scenario
    detail = scenario.detail

    # Build trigger description from structured fields.
    if scenario.trigger_type == "manager_change":
        trigger_desc = (
            f"Manager change: {detail.get('outgoing_manager', '?')} -> "
            f"{detail.get('incoming_manager', '?')}"
        )
    elif scenario.trigger_type == "player_transfer_in":
        trigger_desc = (
            f"Transfer in: {detail.get('player_name', '?')} "
            f"({detail.get('expected_role', '?')})"
        )
    else:
        trigger_desc = f"Trigger: {scenario.trigger_type}"

    # Extract points and cascade highlights.
    points_a = 0.0
    points_b = 0.0
    points_diff = 0.0
    cascade_diff: dict[str, float] = {}

    for hl in explanation.highlights:
        if hl.metric_name == "total_points_mean":
            points_a = hl.value_a
            points_b = hl.value_b
            points_diff = hl.diff
        else:
            cascade_diff[hl.metric_name] = hl.diff

    # Build player impact entries, respecting plan order if provided.
    player_impacts = _build_player_impacts(explanation, plan)

    # Limitations: respect plan's include_info setting.
    resolved_limitations = _resolve_limitations(explanation, plan, limitations, lang)

    # Display hints from plan.
    display_hints = plan.to_display_hints() if plan is not None else None

    return ReportInput(
        trigger_description=trigger_desc,
        points_mean_a=points_a,
        points_mean_b=points_b,
        points_mean_diff=points_diff,
        cascade_count_diff=cascade_diff,
        n_runs=n_runs,
        player_impacts=player_impacts,
        action_explanations=[],  # CausalStep does not map to this contract.
        limitations=resolved_limitations,
        display_hints=display_hints,
    )


def _build_player_impacts(
    explanation: StructuredExplanation,
    plan: ReportPlan | None,
) -> list[PlayerImpactEntry]:
    """Build player impact entries, filtered and ordered by plan."""
    if plan is not None:
        ordered_names = plan.player_display_order
    else:
        ordered_names = tuple(pi.player_name for pi in explanation.player_impacts)

    # Index by name for O(1) lookup.
    impacts_by_name = {pi.player_name: pi for pi in explanation.player_impacts}

    entries: list[PlayerImpactEntry] = []
    for name in ordered_names:
        pi = impacts_by_name.get(name)
        if pi is None:
            continue
        diffs = {c.axis: c.diff for c in pi.changes}
        entries.append(
            PlayerImpactEntry(
                player_name=pi.player_name,
                impact_score=pi.impact_score,
                form_diff=diffs.get("form", 0.0),
                fatigue_diff=diffs.get("fatigue", 0.0),
                understanding_diff=diffs.get("understanding", 0.0),
                trust_diff=diffs.get("trust", 0.0),
            )
        )

    return entries


def _resolve_limitations(
    explanation: StructuredExplanation,
    plan: ReportPlan | None,
    override: list[str] | None,
    lang: str,
) -> list[str]:
    """Resolve limitations list, respecting plan's include_info setting."""
    if override is not None:
        return override

    include_info = plan.limitation_placement.include_info if plan is not None else False
    msg_attr = "message_ja" if lang == "ja" else "message_en"

    items: list[str] = [
        getattr(item, msg_attr) for item in explanation.limitations.system
    ]

    for item in explanation.limitations.scenario:
        if item.severity == "warning" or (include_info and item.severity == "info"):
            items.append(getattr(item, msg_attr))

    return items
