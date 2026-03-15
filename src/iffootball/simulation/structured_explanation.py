"""Structured explanation schema for causal scenario analysis.

Defines the intermediate data structure between simulation results and
report generation. Separates "what to say" from "how to present it".

Design principles:
    - Structure is built by code; LLM only fills statement text.
    - label and source are assigned by code, not LLM.
    - An unfilled skeleton (empty statements) is a valid intermediate state.
    - This module owns the schema and validation, not prose generation.

Future extension notes:
    - confidence_notes: v1 uses plain strings. Future versions may use a
      structured type (reason_type, related_step_ids, message).
    - related_step_ids in PlayerImpactSummary: v1 uses name-based matching
      against affected_agent. Future versions may use richer linking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Trigger detail key definitions per trigger type
# ---------------------------------------------------------------------------

TRIGGER_DETAIL_KEYS: dict[str, tuple[str, ...]] = {
    "manager_change": ("outgoing_manager", "incoming_manager"),
    "player_transfer_in": ("player_name", "expected_role"),
}

# ---------------------------------------------------------------------------
# Label inference
# ---------------------------------------------------------------------------

EvidenceLabel = Literal["data", "analysis", "hypothesis"]
EvidenceSource = Literal["simulation_output", "rule_based_model", "llm_knowledge"]


def infer_label(source: EvidenceSource, depth: int) -> EvidenceLabel:
    """Infer evidence label from source type and causal depth.

    Rules:
        simulation_output   -> always "data"
        rule_based_model    -> "analysis" if depth <= 2, else "hypothesis"
        llm_knowledge       -> always "hypothesis"

    Args:
        source: Origin of the evidence.
        depth:  Causal chain depth (1 = direct, higher = more indirect).

    Returns:
        Inferred label string.
    """
    if source == "simulation_output":
        return "data"
    if source == "rule_based_model":
        return "analysis" if depth <= 2 else "hypothesis"
    # llm_knowledge or unknown
    return "hypothesis"


# ---------------------------------------------------------------------------
# Schema dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioDescriptor:
    """Structured trigger description. No prose, only fields.

    Attributes:
        trigger_type: Matches TriggerType.value (e.g. "manager_change").
        team_name:    Team affected by the trigger.
        detail:       Trigger-specific key-value pairs.
                      Required keys are defined in TRIGGER_DETAIL_KEYS.
    """

    trigger_type: str
    team_name: str
    detail: dict[str, str]

    def __post_init__(self) -> None:
        required = TRIGGER_DETAIL_KEYS.get(self.trigger_type, ())
        missing = [k for k in required if k not in self.detail]
        if missing:
            raise ValueError(
                f"ScenarioDescriptor for {self.trigger_type!r} "
                f"missing required detail keys: {missing}"
            )


@dataclass(frozen=True)
class EvidenceItem:
    """A single piece of evidence with provenance tracking.

    Attributes:
        statement: Natural language description. Empty string in skeleton.
        label:     "data" / "analysis" / "hypothesis". Set by code.
        source:    Origin of the evidence. Set by code.
    """

    statement: str
    label: EvidenceLabel
    source: EvidenceSource


@dataclass(frozen=True)
class DifferenceHighlight:
    """A single metric difference with interpretations.

    Attributes:
        metric_name:     Metric identifier (e.g. "total_points_mean").
        value_a:         Branch A value.
        value_b:         Branch B value.
        diff:            B - A difference.
        interpretations: One or more labelled evidence items explaining
                         the meaning of this difference.
    """

    metric_name: str
    value_a: float
    value_b: float
    diff: float
    interpretations: tuple[EvidenceItem, ...]


@dataclass(frozen=True)
class CausalStep:
    """One step in the causal chain with full traceability.

    Attributes:
        step_id:        Unique identifier for cross-referencing (e.g. "cs-001").
        cause:          Natural language cause description. Empty in skeleton.
        effect:         Natural language effect description. Empty in skeleton.
        affected_agent: Player or manager name (from CascadeEvent).
        event_type:     Event taxonomy type (from VALID_EVENT_TYPES).
        evidence:       Supporting evidence items with labels and sources.
        depth:          Causal chain depth (1 = direct trigger effect).
    """

    step_id: str
    cause: str
    effect: str
    affected_agent: str
    event_type: str
    evidence: tuple[EvidenceItem, ...]
    depth: int


@dataclass(frozen=True)
class PlayerImpactChange:
    """Impact on a single dynamic state axis for one player.

    Attributes:
        axis:           Which dynamic state changed.
        diff:           Branch B - A difference (positive = increase).
        interpretation: Labelled evidence explaining the change.
    """

    axis: Literal["form", "fatigue", "understanding", "trust"]
    diff: float
    interpretation: EvidenceItem


@dataclass(frozen=True)
class PlayerImpactSummary:
    """Aggregated impact summary for one player.

    Attributes:
        player_name:      Display name.
        impact_score:     Mean absolute dynamic-state difference.
        changes:          Per-axis impact with interpretations.
        related_step_ids: CausalStep IDs related to this player.
                          v1: linked by affected_agent name matching.
    """

    player_name: str
    impact_score: float
    changes: tuple[PlayerImpactChange, ...]
    related_step_ids: tuple[str, ...]


# ---------------------------------------------------------------------------
# Confidence note generation helpers
# ---------------------------------------------------------------------------

# Conditions that trigger confidence notes (code-driven, not LLM).
_DEEP_CHAIN_THRESHOLD = 3
_HIGH_LLM_RATIO_THRESHOLD = 0.5


def generate_confidence_note_drafts(
    causal_chain: tuple[CausalStep, ...],
) -> list[str]:
    """Generate draft confidence notes from causal chain properties.

    Produces notes for:
        - Deep causal chains (depth >= threshold).
        - High ratio of rule_based_model or llm_knowledge evidence.

    Args:
        causal_chain: The causal steps to analyze.

    Returns:
        List of confidence note strings. May be empty.
    """
    notes: list[str] = []

    # Deep chain warning.
    max_depth = max((s.depth for s in causal_chain), default=0)
    if max_depth >= _DEEP_CHAIN_THRESHOLD:
        notes.append(
            f"Causal chain reaches depth {max_depth}; "
            f"effects beyond depth 2 carry increasing uncertainty."
        )

    # Evidence source distribution.
    total = 0
    non_simulation = 0
    for step in causal_chain:
        for ev in step.evidence:
            total += 1
            if ev.source != "simulation_output":
                non_simulation += 1

    if total > 0 and (non_simulation / total) >= _HIGH_LLM_RATIO_THRESHOLD:
        pct = round(non_simulation / total * 100)
        notes.append(
            f"{pct}% of evidence relies on rule-based models or "
            f"LLM knowledge rather than direct simulation output."
        )

    return notes


# ---------------------------------------------------------------------------
# Top-level container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructuredExplanation:
    """Complete structured explanation for a scenario comparison.

    This is the contract between simulation analysis and report generation.
    The structure is built by code (build_skeleton); LLM fills statements.

    Attributes:
        scenario:         Structured trigger description.
        highlights:       Key metric differences with interpretations.
        causal_chain:     Ordered causal steps from trigger to effects.
        player_impacts:   Per-player impact summaries.
        confidence_notes: Trust-relevant notes about result reliability.
                          v1: plain strings generated by code heuristics.
                          Future: structured ConfidenceNote type.
    """

    scenario: ScenarioDescriptor
    highlights: tuple[DifferenceHighlight, ...]
    causal_chain: tuple[CausalStep, ...]
    player_impacts: tuple[PlayerImpactSummary, ...]
    confidence_notes: tuple[str, ...]
