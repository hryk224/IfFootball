"""Build StructuredExplanation skeleton from simulation outputs.

Constructs the full StructuredExplanation with all structural fields
populated and statement fields left empty. The skeleton is a valid
intermediate state ready for LLM statement completion.

Pipeline position:
    ComparisonResult + CascadeEvents + PlayerImpacts + ChangeTrigger
      -> build_skeleton()
      -> StructuredExplanation (statements empty)
      -> LLM fills statements
      -> StructuredExplanation (complete)

Note on llm_knowledge source:
    The skeleton builder currently only populates "simulation_output" and
    "rule_based_model" sources. The "llm_knowledge" source is intentionally
    deferred — it will be populated in Step 3-4 when knowledge_query results
    are integrated into the LLM statement completion layer.
"""

from __future__ import annotations

from collections import defaultdict

from iffootball.agents.trigger import (
    ChangeTrigger,
    ManagerChangeTrigger,
    TransferInTrigger,
)
from iffootball.simulation.cascade_tracker import CascadeEvent
from iffootball.simulation.comparison import ComparisonResult
from iffootball.simulation.structured_explanation import (
    SYSTEM_LIMITATIONS,
    CausalStep,
    DifferenceHighlight,
    EvidenceItem,
    EvidenceSource,
    LimitationsDisclosure,
    PlayerImpactChange,
    PlayerImpactSummary,
    ScenarioDescriptor,
    StructuredExplanation,
    generate_scenario_limitations,
    infer_label,
)
from iffootball.visualization.player_impact import PlayerImpact


# ---------------------------------------------------------------------------
# Scenario descriptor
# ---------------------------------------------------------------------------


def _build_scenario(trigger: ChangeTrigger, team_name: str) -> ScenarioDescriptor:
    """Build ScenarioDescriptor from a ChangeTrigger."""
    if isinstance(trigger, ManagerChangeTrigger):
        return ScenarioDescriptor(
            trigger_type="manager_change",
            team_name=team_name,
            detail={
                "outgoing_manager": trigger.outgoing_manager_name,
                "incoming_manager": trigger.incoming_manager_name,
            },
        )
    if isinstance(trigger, TransferInTrigger):
        return ScenarioDescriptor(
            trigger_type="player_transfer_in",
            team_name=team_name,
            detail={
                "player_name": trigger.player_name,
                "expected_role": trigger.expected_role,
            },
        )
    raise TypeError(
        f"Unsupported trigger type: {type(trigger).__name__}. "
        f"Add a handler in _build_scenario() before using this trigger."
    )


# ---------------------------------------------------------------------------
# Difference highlights
# ---------------------------------------------------------------------------


def _build_highlights(
    comparison: ComparisonResult,
) -> tuple[DifferenceHighlight, ...]:
    """Build DifferenceHighlights from ComparisonResult delta.

    Includes points difference and cascade event count differences.
    Statements are left empty for LLM completion.
    """
    highlights: list[DifferenceHighlight] = []

    # Points difference (always included).
    highlights.append(
        DifferenceHighlight(
            metric_name="total_points_mean",
            value_a=round(comparison.no_change.total_points_mean, 2),
            value_b=round(comparison.with_change.total_points_mean, 2),
            diff=round(comparison.delta.points_mean_diff, 2),
            interpretations=(
                EvidenceItem(
                    statement="",
                    label=infer_label("simulation_output", depth=1),
                    source="simulation_output",
                ),
            ),
        )
    )

    # Cascade event count differences (sorted by absolute diff, descending).
    sorted_events = sorted(
        comparison.delta.cascade_count_diff.items(),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )
    for event_type, diff in sorted_events:
        if abs(diff) < 0.01:
            continue
        val_a = comparison.no_change.cascade_event_counts.get(event_type, 0.0)
        val_b = comparison.with_change.cascade_event_counts.get(event_type, 0.0)
        highlights.append(
            DifferenceHighlight(
                metric_name=event_type,
                value_a=round(val_a, 3),
                value_b=round(val_b, 3),
                diff=round(diff, 3),
                interpretations=(
                    EvidenceItem(
                        statement="",
                        label=infer_label("simulation_output", depth=1),
                        source="simulation_output",
                    ),
                ),
            )
        )

    return tuple(highlights)


# ---------------------------------------------------------------------------
# Causal chain
# ---------------------------------------------------------------------------


def _deduplicate_events(
    events: list[CascadeEvent],
) -> list[CascadeEvent]:
    """Deduplicate cascade events preserving causal provenance.

    Key: (event_type, affected_agent, depth, cause_chain) — ensures
    events from different causal paths remain separately traceable.
    Keeps the event with the highest magnitude for each unique key.
    This aggregates across N simulation runs where the same causal
    pattern may appear repeatedly.
    """
    best: dict[tuple[str, str, int, tuple[str, ...]], CascadeEvent] = {}
    for ev in events:
        key = (ev.event_type, ev.affected_agent, ev.depth, ev.cause_chain)
        if key not in best or ev.magnitude > best[key].magnitude:
            best[key] = ev
    # Sort by depth (ascending), then magnitude (descending).
    return sorted(best.values(), key=lambda e: (e.depth, -e.magnitude))


def _build_causal_chain(
    comparison: ComparisonResult,
) -> tuple[CausalStep, ...]:
    """Build CausalSteps from cascade events across all Branch B runs.

    Collects cascade events from all with_change runs, deduplicates,
    and converts to CausalSteps with empty statements.
    """
    all_events: list[CascadeEvent] = []
    for run_result in comparison.with_change.run_results:
        all_events.extend(run_result.cascade_events)

    deduped = _deduplicate_events(all_events)

    steps: list[CausalStep] = []
    for i, ev in enumerate(deduped):
        step_id = f"cs-{i + 1:03d}"
        source: EvidenceSource = (
            "simulation_output" if ev.depth <= 1 else "rule_based_model"
        )
        label = infer_label(source, ev.depth)

        steps.append(
            CausalStep(
                step_id=step_id,
                cause="",  # LLM fills
                effect="",  # LLM fills
                affected_agent=ev.affected_agent,
                event_type=ev.event_type,
                evidence=(
                    EvidenceItem(
                        statement="",
                        label=label,
                        source=source,
                    ),
                ),
                depth=ev.depth,
            )
        )

    return tuple(steps)


# ---------------------------------------------------------------------------
# Player impacts
# ---------------------------------------------------------------------------

# Axis mapping from PlayerImpact attribute pairs to axis names.
_AXIS_ATTRS: tuple[
    tuple[str, str, str, str],
    ...,
] = (
    ("form", "mean_form_a", "mean_form_b", "form"),
    ("fatigue", "mean_fatigue_a", "mean_fatigue_b", "fatigue"),
    ("understanding", "mean_understanding_a", "mean_understanding_b", "understanding"),
    ("trust", "mean_trust_a", "mean_trust_b", "trust"),
)


def _build_player_impacts(
    impacts: list[PlayerImpact],
    causal_chain: tuple[CausalStep, ...],
) -> tuple[PlayerImpactSummary, ...]:
    """Build PlayerImpactSummaries from ranked impacts.

    Links players to causal steps by affected_agent name matching (v1).
    """
    # Build step lookup by affected_agent.
    agent_steps: dict[str, list[str]] = defaultdict(list)
    for step in causal_chain:
        agent_steps[step.affected_agent].append(step.step_id)

    summaries: list[PlayerImpactSummary] = []
    for p in impacts:
        changes: list[PlayerImpactChange] = []
        for axis_name, attr_a, attr_b, axis_literal in _AXIS_ATTRS:
            val_a = getattr(p, attr_a)
            val_b = getattr(p, attr_b)
            diff = round(val_b - val_a, 4)
            changes.append(
                PlayerImpactChange(
                    axis=axis_literal,  # type: ignore[arg-type]
                    diff=diff,
                    interpretation=EvidenceItem(
                        statement="",
                        label=infer_label("simulation_output", depth=1),
                        source="simulation_output",
                    ),
                )
            )

        related_ids = tuple(agent_steps.get(p.player_name, []))

        summaries.append(
            PlayerImpactSummary(
                player_name=p.player_name,
                impact_score=round(p.impact_score, 4),
                changes=tuple(changes),
                related_step_ids=related_ids,
                sample_tier=p.sample_tier,
            )
        )

    return tuple(summaries)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_skeleton(
    comparison: ComparisonResult,
    trigger: ChangeTrigger,
    team_name: str,
    impacts: list[PlayerImpact],
) -> StructuredExplanation:
    """Build a StructuredExplanation skeleton from simulation outputs.

    All structural fields (step_id, affected_agent, event_type, depth,
    metric values, diffs, labels, sources) are populated by code.
    Statement fields are left as empty strings for LLM completion.

    Args:
        comparison:  A/B comparison result.
        trigger:     The change trigger that was applied.
        team_name:   Name of the team being analyzed.
        impacts:     Ranked player impacts from rank_player_impact().

    Returns:
        StructuredExplanation with empty statements, ready for LLM fill.
    """
    scenario = _build_scenario(trigger, team_name)
    highlights = _build_highlights(comparison)
    causal_chain = _build_causal_chain(comparison)
    player_impacts = _build_player_impacts(impacts, causal_chain)
    scenario_limitations = generate_scenario_limitations(causal_chain)
    limitations = LimitationsDisclosure(
        system=SYSTEM_LIMITATIONS,
        scenario=scenario_limitations,
    )

    return StructuredExplanation(
        scenario=scenario,
        highlights=highlights,
        causal_chain=causal_chain,
        player_impacts=player_impacts,
        limitations=limitations,
    )
