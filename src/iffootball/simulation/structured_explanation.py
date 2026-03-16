"""Structured explanation schema for causal scenario analysis.

Defines the intermediate data structure between simulation results and
report generation. Separates "what to say" from "how to present it".

Design principles:
    - Structure is built by code; LLM only fills statement text.
    - label and source are assigned by code, not LLM.
    - An unfilled skeleton (empty statements) is a valid intermediate state.
    - This module owns the schema and validation, not prose generation.

Future extension notes:
    - related_step_ids in PlayerImpactSummary: v1 uses name-based matching
      against affected_agent. Future versions may use richer linking.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
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
# Limitations disclosure
# ---------------------------------------------------------------------------


class LimitationCategory(str, Enum):
    """Classification of limitation types."""

    MODEL_BOUNDARY = "model_boundary"
    DATA_SCOPE = "data_scope"
    ESTIMATION_DEPENDENCY = "estimation"
    CHAIN_DEPTH = "chain_depth"
    UNOBSERVED_FACTOR = "unobserved"


LimitationSeverity = Literal["info", "warning"]


@dataclass(frozen=True)
class LimitationItem:
    """A single limitation with classification and severity.

    Attributes:
        category:         Type of limitation.
        message_en:       English description.
        message_ja:       Japanese description.
        severity:         "warning" = always show, "info" = detail view only.
        related_step_ids: CausalStep IDs this limitation applies to.
                          Empty = system-wide or scenario-wide.
    """

    category: LimitationCategory
    message_en: str
    message_ja: str
    severity: LimitationSeverity
    related_step_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class LimitationsDisclosure:
    """Two-layer limitation disclosure.

    Attributes:
        system:   Fixed model-level constraints. Same for all scenarios.
        scenario: Scenario-specific uncertainty. Generated from analysis.
    """

    system: tuple[LimitationItem, ...]
    scenario: tuple[LimitationItem, ...]


# ---------------------------------------------------------------------------
# System limitations (authoritative structured definitions)
# ---------------------------------------------------------------------------

SYSTEM_LIMITATIONS: tuple[LimitationItem, ...] = (
    LimitationItem(
        category=LimitationCategory.MODEL_BOUNDARY,
        message_en=(
            "Match outcomes use a Poisson model with xG-based expected goals; "
            "in-match events (shots, passes) are not simulated."
        ),
        message_ja=(
            "試合結果は xG ベースの Poisson モデルで決定されます。"
            "試合内イベント（シュート、パス）はシミュレートされません。"
        ),
        severity="warning",
    ),
    LimitationItem(
        category=LimitationCategory.ESTIMATION_DEPENDENCY,
        message_en=(
            "Tactical metrics (PPDA, possession, progressive passes) for the "
            "incoming manager are estimates, not simulation outputs."
        ),
        message_ja=(
            "後任監督の戦術指標（PPDA、ポゼッション、プログレッシブパス）は "
            "推定値であり、シミュレーション出力ではありません。"
        ),
        severity="warning",
    ),
    LimitationItem(
        category=LimitationCategory.MODEL_BOUNDARY,
        message_en=(
            "Player technical attributes are fixed throughout the simulation; "
            "only dynamic state (form, fatigue, trust, understanding) changes."
        ),
        message_ja=(
            "選手の技術属性はシミュレーション中固定です。"
            "変化するのは動的状態（フォーム、疲労、信頼度、戦術理解度）のみです。"
        ),
        severity="info",
    ),
    LimitationItem(
        category=LimitationCategory.ESTIMATION_DEPENDENCY,
        message_en=(
            "The action distribution at turning points is rule-based; "
            "LLM-based action selection is not yet implemented."
        ),
        message_ja=(
            "ターニングポイントでの行動分布はルールベースです。"
            "LLM ベースの行動選択はまだ実装されていません。"
        ),
        severity="info",
    ),
    LimitationItem(
        category=LimitationCategory.MODEL_BOUNDARY,
        message_en=(
            "xGA/90 is a fixed baseline; the current model does not simulate "
            "defensive impact of manager changes."
        ),
        message_ja=(
            "xGA/90 は固定ベースラインです。現在のモデルは "
            "監督交代による守備への影響をシミュレートしません。"
        ),
        severity="warning",
    ),
)


# ---------------------------------------------------------------------------
# Scenario limitation generation
# ---------------------------------------------------------------------------

_DEEP_CHAIN_THRESHOLD = 3
_HIGH_NON_SIMULATION_RATIO = 0.5


def generate_scenario_limitations(
    causal_chain: tuple[CausalStep, ...],
) -> tuple[LimitationItem, ...]:
    """Generate scenario-specific limitations from causal chain analysis.

    v1 conditions:
        - Causal chain depth >= 3 -> chain_depth warning.
        - Non-simulation evidence ratio >= 50% -> estimation warning.

    Args:
        causal_chain: The causal steps to analyze.

    Returns:
        Tuple of scenario-specific LimitationItems. May be empty.
    """
    items: list[LimitationItem] = []

    # Deep chain warning.
    max_depth = max((s.depth for s in causal_chain), default=0)
    deep_step_ids = tuple(
        s.step_id for s in causal_chain if s.depth >= _DEEP_CHAIN_THRESHOLD
    )
    if max_depth >= _DEEP_CHAIN_THRESHOLD:
        items.append(
            LimitationItem(
                category=LimitationCategory.CHAIN_DEPTH,
                message_en=(
                    f"Causal chain reaches depth {max_depth}; "
                    f"effects beyond depth 2 carry increasing uncertainty."
                ),
                message_ja=(
                    f"因果連鎖が深さ {max_depth} に達しています。"
                    f"深さ 2 を超える効果は不確実性が増大します。"
                ),
                severity="warning",
                related_step_ids=deep_step_ids,
            )
        )

    # Evidence source distribution.
    total = 0
    non_simulation = 0
    for step in causal_chain:
        for ev in step.evidence:
            total += 1
            if ev.source != "simulation_output":
                non_simulation += 1

    if total > 0 and (non_simulation / total) >= _HIGH_NON_SIMULATION_RATIO:
        pct = round(non_simulation / total * 100)
        items.append(
            LimitationItem(
                category=LimitationCategory.ESTIMATION_DEPENDENCY,
                message_en=(
                    f"{pct}% of evidence relies on rule-based models or "
                    f"LLM knowledge rather than direct simulation output."
                ),
                message_ja=(
                    f"根拠の {pct}% がルールベースモデルまたは "
                    f"LLM 知識に依存しており、シミュレーション直接出力ではありません。"
                ),
                severity="warning",
            )
        )

    return tuple(items)


# ---------------------------------------------------------------------------
# Top-level container
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Validation signals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationSignal:
    """An observation point for post-hoc hypothesis checking.

    Tells the reader what to watch in the first few matches after the
    trigger to verify whether the simulation's causal chain holds.

    Not a recommendation — a testable observation.

    Attributes:
        metric:          What to observe (StatsBomb metric or event type).
        observation_window: When to look (e.g. "first 3 matches").
        metric_direction:   Direction of the metric change this signal expects
                            ("increase" / "decrease" / "stable").
        hypothesis_support: What this direction means for the hypothesis
                            ("supports" / "contradicts"). When the metric moves
                            in metric_direction, the hypothesis is supported.
        reason:          Why this metric matters for this scenario.
        related_step_id: CausalStep this signal validates (if any).
        confidence:      Confidence level, determined by causal depth:
                         "high" = depth 1 step, "medium" = depth 2 or
                         player impact, "low" = highlight only.
    """

    metric: str
    observation_window: str
    metric_direction: str  # "increase" / "decrease" / "stable"
    hypothesis_support: str  # "supports" / "contradicts"
    reason: str
    related_step_id: str | None
    confidence: str  # "high" / "medium" / "low"


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
        limitations:      Two-layer limitation disclosure
                          (system + scenario-specific).
    """

    scenario: ScenarioDescriptor
    highlights: tuple[DifferenceHighlight, ...]
    causal_chain: tuple[CausalStep, ...]
    player_impacts: tuple[PlayerImpactSummary, ...]
    limitations: LimitationsDisclosure
