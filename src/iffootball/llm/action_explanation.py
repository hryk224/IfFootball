"""LLM-based action explanation for turning point responses.

Generates a short explanation of why a player took a specific action
(adapt / resist / transfer) at a turning point. The explanation includes
a fact/analysis/hypothesis label for downstream report generation.

Pipeline:
  1. Load system prompt from prompts/action_explanation_v1.md.
     Raises FileNotFoundError if missing — prompt files are required.
  2. Serialize player state, TP context, and action as JSON into user message.
  3. Send [system, user] to LLMClient.complete().
  4. Parse and validate JSON response.
  5. On invalid JSON or missing fields, return a safe default result.

Prompt injection mitigation:
  External inputs (player_name, manager_name) are embedded only in the
  user message as JSON values, never in the system prompt. The system
  prompt is loaded from a static file and never interpolated.

Calling convention:
  This function is called in post-processing (not inside the N-run
  simulation loop). It is invoked once per unique TP-type × action
  combination observed across simulation results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from iffootball.agents.player import PlayerAgent
from iffootball.llm.client import LLMClient
from iffootball.simulation.turning_point import ActionDistribution, SimContext

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_DEFAULT_PROMPT_PATH = (
    Path(__file__).parents[3] / "prompts" / "action_explanation_v1.md"
)


def _load_system_prompt(path: Path | None = None) -> str:
    """Load system prompt from file. Raises FileNotFoundError if missing."""
    resolved = path if path is not None else _DEFAULT_PROMPT_PATH
    return resolved.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_LABELS = frozenset({"fact", "analysis", "hypothesis"})
_DEFAULT_LABEL = "hypothesis"
_DEFAULT_EXPLANATION = "Unable to generate structured explanation."


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionExplanationResult:
    """LLM-generated explanation for a turning point action.

    Attributes:
        explanation:     1-2 sentence explanation with data references.
        label:           Classification: "fact", "analysis", or "hypothesis".
        confidence_note: Optional note on uncertainty or data limitations.
    """

    explanation: str
    label: str
    confidence_note: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def explain_action(
    client: LLMClient,
    player: PlayerAgent,
    context: SimContext,
    sampled_action: str,
    distribution: ActionDistribution,
    turning_points: list[str],
    *,
    system_prompt: str | None = None,
) -> ActionExplanationResult:
    """Generate an LLM explanation for a player's turning point action.

    Args:
        client:          LLMClient implementation.
        player:          Player at the time of the turning point.
        context:         Simulation context (week, manager, points).
        sampled_action:  The action that was sampled ("adapt"/"resist"/"transfer").
        distribution:    The ActionDistribution that produced the action.
        turning_points:  List of TP types that fired (e.g. ["bench_streak"]).
        system_prompt:   Override the loaded system prompt (tests only).

    Returns:
        ActionExplanationResult with explanation, label, and confidence_note.
        Falls back to safe defaults on parse failure.
    """
    prompt = (
        system_prompt if system_prompt is not None else _load_system_prompt()
    )

    user_payload = _build_user_payload(
        player, context, sampled_action, distribution, turning_points
    )

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]

    raw = client.complete(messages)
    return _parse_response(raw)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_user_payload(
    player: PlayerAgent,
    context: SimContext,
    sampled_action: str,
    distribution: ActionDistribution,
    turning_points: list[str],
) -> dict[str, Any]:
    """Build the JSON payload for the user message."""
    return {
        "player": {
            "name": player.player_name,
            "position": player.position_name,
            "current_form": round(player.current_form, 3),
            "fatigue": round(player.fatigue, 3),
            "tactical_understanding": round(player.tactical_understanding, 3),
            "manager_trust": round(player.manager_trust, 3),
            "bench_streak": player.bench_streak,
        },
        "turning_points": turning_points,
        "sampled_action": sampled_action,
        "action_distribution": {
            k: round(v, 3) for k, v in distribution.choices.items()
        },
        "context": {
            "current_week": context.current_week,
            "matches_since_appointment": context.matches_since_appointment,
            "manager_name": context.manager.manager_name,
            "recent_points": list(context.recent_points),
        },
        "source_types": {
            "form_fatigue_trust": "simulation_output",
            "tactical_understanding": "simulation_output",
            "action_distribution": "rule_based_model",
        },
    }


def _parse_response(raw: str) -> ActionExplanationResult:
    """Parse and validate LLM JSON response."""
    data = _parse_json(raw)
    if data is None:
        return ActionExplanationResult(
            explanation=_DEFAULT_EXPLANATION,
            label=_DEFAULT_LABEL,
            confidence_note="",
        )

    explanation = data.get("explanation", "")
    if not isinstance(explanation, str) or not explanation.strip():
        explanation = _DEFAULT_EXPLANATION

    label = data.get("label", "")
    if not isinstance(label, str) or label not in _VALID_LABELS:
        label = _DEFAULT_LABEL

    confidence_note = data.get("confidence_note", "")
    if not isinstance(confidence_note, str):
        confidence_note = ""

    return ActionExplanationResult(
        explanation=explanation.strip(),
        label=label,
        confidence_note=confidence_note.strip(),
    )


def _parse_json(raw: str) -> dict[str, Any] | None:
    """Return parsed dict from a JSON string, or None on failure."""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
        return None
    except json.JSONDecodeError:
        return None
