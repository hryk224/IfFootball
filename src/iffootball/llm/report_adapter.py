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
    DEFAULT_LIMITATIONS,
    PlayerImpactEntry,
    ReportInput,
)
from iffootball.simulation.structured_explanation import StructuredExplanation


def structured_to_report_input(
    explanation: StructuredExplanation,
    *,
    limitations: list[str] | None = None,
    n_runs: int = 1,
    lang: str = "en",
) -> ReportInput:
    """Convert a StructuredExplanation to a ReportInput.

    Extracts the data needed by the existing report generation pipeline
    from the structured explanation.

    Semantic notes:
        - action_explanations is always empty. CausalStep data does not
          map to the existing ActionExplanationEntry contract (tp_type /
          sampled action / explanation). The existing prompt handles
          empty action_explanations gracefully.
        - limitations uses DEFAULT_LIMITATIONS (system constraints),
          not confidence_notes (scenario-specific uncertainty). Caller
          can override via the limitations parameter.

    Args:
        explanation: Completed StructuredExplanation.
        limitations: Override limitations list. If None, uses
                     DEFAULT_LIMITATIONS for the given language.
        n_runs:      Number of simulation runs (not stored in
                     StructuredExplanation; caller must provide).
        lang:        Language for default limitations ("en" or "ja").

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

    # Build player impact entries.
    player_impacts: list[PlayerImpactEntry] = []
    for pi in explanation.player_impacts:
        diffs = {c.axis: c.diff for c in pi.changes}
        player_impacts.append(
            PlayerImpactEntry(
                player_name=pi.player_name,
                impact_score=pi.impact_score,
                form_diff=diffs.get("form", 0.0),
                fatigue_diff=diffs.get("fatigue", 0.0),
                understanding_diff=diffs.get("understanding", 0.0),
                trust_diff=diffs.get("trust", 0.0),
            )
        )

    # Limitations: use system-level defaults, not confidence_notes.
    resolved_limitations = (
        limitations
        if limitations is not None
        else list(DEFAULT_LIMITATIONS.get(lang, DEFAULT_LIMITATIONS["en"]))
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
    )
