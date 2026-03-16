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
    CausalStepEntry,
    EvidenceEntry,
    HighlightEntry,
    PlayerAxisChange as ReportAxisChange,
    PlayerImpactDetailEntry,
    PlayerImpactEntry,
    PlayerImpactMeta,
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
    resolved_limitations = _resolve_limitations(explanation, plan, limitations)

    # Display hints from plan.
    display_hints = plan.to_display_hints() if plan is not None else None

    # Structured label-carrying fields.
    highlights = _build_highlights(explanation)
    causal_steps = _build_causal_steps(explanation, plan)
    shared_resets = _detect_shared_resets(explanation, plan)
    player_details = _build_player_details(
        explanation, plan, tuple(shared_resets.keys())
    )
    player_meta = (
        PlayerImpactMeta(shared_resets=shared_resets) if shared_resets else None
    )

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
        highlights=highlights,
        causal_steps=causal_steps,
        player_impact_details=player_details,
        player_impact_meta=player_meta,
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
) -> list[str]:
    """Resolve limitations list, respecting plan's include_info setting."""
    if override is not None:
        return override

    include_info = plan.limitation_placement.include_info if plan is not None else False
    msg_attr = "message_en"

    items: list[str] = [
        getattr(item, msg_attr) for item in explanation.limitations.system
    ]

    for item in explanation.limitations.scenario:
        if item.severity == "warning" or (include_info and item.severity == "info"):
            items.append(getattr(item, msg_attr))

    return items


# ---------------------------------------------------------------------------
# Structured label-carrying builders (v2)
# ---------------------------------------------------------------------------


def _diff_direction(diff: float) -> str:
    """Return direction string from a numeric diff."""
    if diff > 0.001:
        return "increased"
    if diff < -0.001:
        return "decreased"
    return "unchanged"


def _highlight_unit(metric_name: str) -> str:
    """Infer unit from metric name."""
    if metric_name == "total_points_mean":
        return "points_mean"
    return "events_per_run"


def _build_highlights(
    explanation: StructuredExplanation,
) -> list[HighlightEntry]:
    """Build HighlightEntry list from StructuredExplanation highlights."""
    entries: list[HighlightEntry] = []
    for hl in explanation.highlights:
        label = hl.interpretations[0].label if hl.interpretations else "data"
        statement = hl.interpretations[0].statement if hl.interpretations else ""
        entries.append(
            HighlightEntry(
                metric_name=hl.metric_name,
                diff=round(hl.diff, 3),
                label=label,
                statement=statement,
                unit=_highlight_unit(hl.metric_name),
                direction=_diff_direction(hl.diff),
            )
        )
    return entries


_LABEL_RANK = {"data": 0, "analysis": 1, "hypothesis": 2}


def _infer_paragraph_label(evidence_labels: tuple[str, ...]) -> str:
    """Infer the paragraph label for a causal step.

    cause/effect text connects data points, so it is never plain "data".
    The paragraph label is the maximum of "analysis" and the highest
    evidence label.

    Rules:
        - If any evidence is "hypothesis" -> "hypothesis"
        - If any evidence is "analysis" -> "analysis"
        - If all evidence is "data" -> still "analysis" (cause/effect is causal)
        - No evidence -> "analysis"
    """
    if not evidence_labels:
        return "analysis"
    max_rank = max(_LABEL_RANK.get(lbl, 0) for lbl in evidence_labels)
    # Floor at "analysis" because cause/effect is always causal reasoning.
    floor_rank = _LABEL_RANK["analysis"]
    effective_rank = max(max_rank, floor_rank)
    for label, rank in _LABEL_RANK.items():
        if rank == effective_rank:
            return label
    return "analysis"


def _build_causal_steps(
    explanation: StructuredExplanation,
    plan: ReportPlan | None,
) -> list[CausalStepEntry]:
    """Build CausalStepEntry list from StructuredExplanation causal chain.

    When a plan is provided, only steps in expanded_step_ids are included.
    When no plan, all steps are included.

    paragraph_label is inferred from evidence labels but always >= "analysis"
    because cause/effect text connects data points (never plain "data").
    """
    expanded = plan.expanded_step_ids if plan is not None else None

    entries: list[CausalStepEntry] = []
    for step in explanation.causal_chain:
        if expanded is not None and step.step_id not in expanded:
            continue
        evidence_labels = tuple(ev.label for ev in step.evidence)
        paragraph_label = _infer_paragraph_label(evidence_labels)
        evidence_entries = tuple(
            EvidenceEntry(
                statement=ev.statement,
                label=ev.label,
                source=ev.source,
            )
            for ev in step.evidence
        )
        entries.append(
            CausalStepEntry(
                step_id=step.step_id,
                cause=step.cause,
                effect=step.effect,
                affected_agent=step.affected_agent,
                event_type=step.event_type,
                depth=step.depth,
                paragraph_label=paragraph_label,
                evidence_labels=evidence_labels,
                evidence=evidence_entries,
            )
        )
    return entries


_MAX_PLAYER_AXES = 2


def _detect_shared_resets(
    explanation: StructuredExplanation,
    plan: ReportPlan | None,
) -> dict[str, float]:
    """Detect axes where all featured players share the same diff value.

    Returns a dict mapping axis name to the common diff value.
    Only axes with a single non-trivial value across all featured players
    are included. Returns empty dict if no shared resets detected.
    """
    if plan is not None:
        featured = set(plan.player_display_order)
    else:
        featured = {pi.player_name for pi in explanation.player_impacts}

    if not featured:
        return {}

    # Collect per-axis diff values across featured players.
    axis_values: dict[str, set[float]] = {}
    for pi in explanation.player_impacts:
        if pi.player_name not in featured:
            continue
        for c in pi.changes:
            rounded = round(c.diff, 4)
            axis_values.setdefault(c.axis, set()).add(rounded)

    shared: dict[str, float] = {}
    for axis, values in axis_values.items():
        if len(values) == 1:
            val = next(iter(values))
            if abs(val) > 0.001:  # Non-trivial shared value.
                shared[axis] = val

    return shared


def _build_player_details(
    explanation: StructuredExplanation,
    plan: ReportPlan | None,
    shared_reset_axes: tuple[str, ...],
) -> list[PlayerImpactDetailEntry]:
    """Build PlayerImpactDetailEntry list with per-axis labels and statements.

    Shared reset axes are excluded from individual player changes.
    At most _MAX_PLAYER_AXES (2) most significant axes are retained,
    sorted by absolute diff descending.
    """
    if plan is not None:
        ordered_names = plan.player_display_order
    else:
        ordered_names = tuple(pi.player_name for pi in explanation.player_impacts)

    shared_set = set(shared_reset_axes)
    impacts_by_name = {pi.player_name: pi for pi in explanation.player_impacts}

    entries: list[PlayerImpactDetailEntry] = []
    for name in ordered_names:
        pi = impacts_by_name.get(name)
        if pi is None:
            continue
        # Filter out shared reset axes and sort by significance.
        candidates: list[ReportAxisChange] = []
        for c in pi.changes:
            if c.axis in shared_set:
                continue
            candidates.append(
                ReportAxisChange(
                    axis=c.axis,
                    diff=round(c.diff, 4),
                    label=c.interpretation.label,
                    statement=c.interpretation.statement,
                )
            )
        # Keep top N by absolute diff.
        candidates.sort(key=lambda x: abs(x.diff), reverse=True)
        top_changes = tuple(candidates[:_MAX_PLAYER_AXES])

        entries.append(
            PlayerImpactDetailEntry(
                player_name=pi.player_name,
                impact_score=round(pi.impact_score, 4),
                changes=top_changes,
            )
        )
    return entries
