"""LLM-based statement completion for StructuredExplanation skeletons.

Takes a skeleton (empty statements) and fills text fields via LLM.
The LLM only generates natural-language descriptions; all structural
fields (step_id, label, source, depth, etc.) are preserved from the
skeleton unchanged.

Pipeline position:
    build_skeleton() -> complete_skeleton() -> StructuredExplanation (filled)
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from iffootball.llm.client import LLMClient
from iffootball.simulation.structured_explanation import (
    CausalStep,
    DifferenceHighlight,
    EvidenceItem,
    PlayerImpactChange,
    PlayerImpactSummary,
    StructuredExplanation,
)

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_PROMPT_DIR = Path(__file__).parents[2] / "prompts"
_PROMPT_PATH = _PROMPT_DIR / "structured_explanation_v1.md"


def _load_system_prompt(path: Path | None = None) -> str:
    """Load system prompt from file."""
    resolved = path or _PROMPT_PATH
    return resolved.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _skeleton_to_json(skeleton: StructuredExplanation) -> str:
    """Serialize skeleton to JSON for the LLM user message."""
    raw = asdict(skeleton)
    return json.dumps(raw, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Deserialization (merge LLM response back into skeleton)
# ---------------------------------------------------------------------------


def _parse_llm_response(raw: str) -> dict[str, Any] | None:
    """Parse LLM JSON response, stripping code fences if present."""
    text = raw.strip()
    # Strip markdown code block wrapper if present.
    if text.startswith("```"):
        first_newline = text.index("\n")
        text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[: -len("```")]
        text = text.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _merge_evidence(
    skeleton_ev: EvidenceItem,
    filled: dict[str, Any],
) -> EvidenceItem:
    """Merge LLM-filled statement into skeleton evidence, preserving label/source."""
    return EvidenceItem(
        statement=filled.get("statement", skeleton_ev.statement),
        label=skeleton_ev.label,
        source=skeleton_ev.source,
    )


def _merge_highlight(
    skeleton_hl: DifferenceHighlight,
    filled: dict[str, Any],
) -> DifferenceHighlight:
    """Merge filled interpretations into skeleton highlight."""
    filled_interps = filled.get("interpretations", [])
    merged_interps: list[EvidenceItem] = []
    for i, sev in enumerate(skeleton_hl.interpretations):
        if i < len(filled_interps):
            merged_interps.append(_merge_evidence(sev, filled_interps[i]))
        else:
            merged_interps.append(sev)
    return DifferenceHighlight(
        metric_name=skeleton_hl.metric_name,
        value_a=skeleton_hl.value_a,
        value_b=skeleton_hl.value_b,
        diff=skeleton_hl.diff,
        interpretations=tuple(merged_interps),
    )


def _merge_causal_step(
    skeleton_step: CausalStep,
    filled: dict[str, Any],
) -> CausalStep:
    """Merge filled cause/effect/evidence into skeleton step."""
    filled_evidence = filled.get("evidence", [])
    merged_evidence: list[EvidenceItem] = []
    for i, sev in enumerate(skeleton_step.evidence):
        if i < len(filled_evidence):
            merged_evidence.append(_merge_evidence(sev, filled_evidence[i]))
        else:
            merged_evidence.append(sev)

    return CausalStep(
        step_id=skeleton_step.step_id,
        cause=filled.get("cause", skeleton_step.cause),
        effect=filled.get("effect", skeleton_step.effect),
        affected_agent=skeleton_step.affected_agent,
        event_type=skeleton_step.event_type,
        evidence=tuple(merged_evidence),
        depth=skeleton_step.depth,
    )


def _merge_player_change(
    skeleton_change: PlayerImpactChange,
    filled: dict[str, Any],
) -> PlayerImpactChange:
    """Merge filled interpretation into skeleton player change."""
    filled_interp = filled.get("interpretation", {})
    return PlayerImpactChange(
        axis=skeleton_change.axis,
        diff=skeleton_change.diff,
        interpretation=EvidenceItem(
            statement=filled_interp.get(
                "statement", skeleton_change.interpretation.statement
            ),
            label=skeleton_change.interpretation.label,
            source=skeleton_change.interpretation.source,
        ),
    )


def _merge_player_impact(
    skeleton_pi: PlayerImpactSummary,
    filled: dict[str, Any],
) -> PlayerImpactSummary:
    """Merge filled changes into skeleton player impact."""
    filled_changes = filled.get("changes", [])
    merged: list[PlayerImpactChange] = []
    for i, sc in enumerate(skeleton_pi.changes):
        if i < len(filled_changes):
            merged.append(_merge_player_change(sc, filled_changes[i]))
        else:
            merged.append(sc)
    return PlayerImpactSummary(
        player_name=skeleton_pi.player_name,
        impact_score=skeleton_pi.impact_score,
        changes=tuple(merged),
        related_step_ids=skeleton_pi.related_step_ids,
        sample_tier=skeleton_pi.sample_tier,
    )


def _merge_response(
    skeleton: StructuredExplanation,
    filled_data: dict[str, Any],
) -> StructuredExplanation:
    """Merge LLM-filled data back into the skeleton, preserving all structural fields."""
    # Highlights — merge by metric_name to tolerate reordering.
    filled_highlights = filled_data.get("highlights", [])
    filled_by_metric: dict[str, dict[str, Any]] = {}
    for fh in filled_highlights:
        mn = fh.get("metric_name", "")
        if mn:
            filled_by_metric[mn] = fh
    merged_highlights: list[DifferenceHighlight] = []
    for shl in skeleton.highlights:
        matched = filled_by_metric.get(shl.metric_name)
        if matched is not None:
            merged_highlights.append(_merge_highlight(shl, matched))
        else:
            merged_highlights.append(shl)

    # Causal chain — merge by step_id to tolerate reordering.
    filled_chain = filled_data.get("causal_chain", [])
    filled_by_step_id: dict[str, dict[str, Any]] = {}
    for fc in filled_chain:
        sid = fc.get("step_id", "")
        if sid:
            filled_by_step_id[sid] = fc
    merged_chain: list[CausalStep] = []
    for ss in skeleton.causal_chain:
        matched = filled_by_step_id.get(ss.step_id)
        if matched is not None:
            merged_chain.append(_merge_causal_step(ss, matched))
        else:
            merged_chain.append(ss)

    # Player impacts — merge by player_name to tolerate reordering.
    filled_players = filled_data.get("player_impacts", [])
    filled_by_name: dict[str, dict[str, Any]] = {}
    for fp in filled_players:
        pn = fp.get("player_name", "")
        if pn:
            filled_by_name[pn] = fp
    merged_players: list[PlayerImpactSummary] = []
    for sp in skeleton.player_impacts:
        matched = filled_by_name.get(sp.player_name)
        if matched is not None:
            merged_players.append(_merge_player_impact(sp, matched))
        else:
            merged_players.append(sp)

    # Limitations — preserved from skeleton entirely.
    # LLM cannot modify system or scenario limitations.

    return StructuredExplanation(
        scenario=skeleton.scenario,
        highlights=tuple(merged_highlights),
        causal_chain=tuple(merged_chain),
        player_impacts=tuple(merged_players),
        limitations=skeleton.limitations,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def complete_skeleton(
    client: LLMClient,
    skeleton: StructuredExplanation,
    *,
    system_prompt: str | None = None,
) -> StructuredExplanation:
    """Fill empty statement fields in a StructuredExplanation via LLM.

    Sends the skeleton as JSON to the LLM, which returns the same
    structure with statement fields filled. Structural fields (step_id,
    label, source, depth, etc.) are always preserved from the skeleton,
    never from the LLM response.

    Args:
        client:        LLMClient implementation.
        skeleton:      StructuredExplanation with empty statements.
        system_prompt: Override the loaded system prompt (tests only).

    Returns:
        StructuredExplanation with statements filled by LLM.
        Falls back to the original skeleton if the LLM response is
        unparseable.
    """
    prompt = system_prompt or _load_system_prompt()
    payload = _skeleton_to_json(skeleton)

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": payload},
    ]

    raw = client.complete(messages)

    if not raw or not raw.strip():
        return skeleton

    parsed = _parse_llm_response(raw)
    if parsed is None:
        return skeleton

    return _merge_response(skeleton, parsed)
