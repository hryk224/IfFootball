"""LLM-based comparison report generation.

Generates a structured Markdown report from simulation comparison results,
action explanations, and player impact data. The report includes labelled
sections ([fact] / [analysis] / [hypothesis]) and a limitations section
for known system constraints.

Pipeline:
  1. Caller assembles a ReportInput from existing outputs.
  2. ReportInput is serialized to JSON as the user message.
  3. LLM generates a Markdown report following the section structure
     defined in prompts/report_generation_v1.md.
  4. The raw Markdown string is returned.

Calling convention:
  Called once per comparison, after simulation runs, action explanations,
  and player impact ranking are complete. Not called inside the N-run loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from iffootball.llm.client import LLMClient

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_DEFAULT_PROMPT_PATH = (
    Path(__file__).parents[3] / "prompts" / "report_generation_v1.md"
)


def _load_system_prompt(path: Path | None = None) -> str:
    """Load system prompt from file. Raises FileNotFoundError if missing."""
    resolved = path if path is not None else _DEFAULT_PROMPT_PATH
    return resolved.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Required section headings in the generated report.
REQUIRED_SECTIONS: tuple[str, ...] = (
    "## Summary",
    "## Key Differences",
    "## Causal Chain",
    "## Player Impact",
    "## Limitations",
)

# Default limitations describing known simulation constraints.
DEFAULT_LIMITATIONS: tuple[str, ...] = (
    "Match outcomes use a Poisson model with xG-based expected goals; "
    "in-match events (shots, passes) are not simulated.",
    "Tactical metrics (PPDA, possession, progressive passes) for the "
    "incoming manager are estimates, not simulation outputs.",
    "Player technical attributes are fixed throughout the simulation; "
    "only dynamic state (form, fatigue, trust, understanding) changes.",
    "The action distribution at turning points is rule-based (Phase 1); "
    "LLM-based action selection is not yet implemented.",
    "xGA/90 is a fixed baseline; the current model does not simulate "
    "defensive impact of manager changes.",
)

# Default number of top impacted players to include.
DEFAULT_TOP_PLAYERS = 3

# Fallback report when LLM output is empty or unusable.
_FALLBACK_REPORT = (
    "## Summary\n\n"
    "Unable to generate structured report.\n\n"
    "## Key Differences\n\n"
    "No data available.\n\n"
    "## Causal Chain\n\n"
    "No data available.\n\n"
    "## Player Impact\n\n"
    "No data available.\n\n"
    "## Limitations\n\n"
    "Report generation failed. Results may be incomplete."
)


# ---------------------------------------------------------------------------
# Input types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionExplanationEntry:
    """A turning point action explanation with context.

    Wraps ActionExplanationResult with the TP type and action that
    produced it, so the report can present causal chains.

    Attributes:
        tp_type:         Turning point type (e.g. "bench_streak").
        action:          Sampled action ("adapt" / "resist" / "transfer").
        explanation:     LLM-generated explanation text.
        label:           "fact" / "analysis" / "hypothesis".
        confidence_note: Optional uncertainty note.
    """

    tp_type: str
    action: str
    explanation: str
    label: str
    confidence_note: str


@dataclass(frozen=True)
class PlayerImpactEntry:
    """Simplified player impact for report input.

    Attributes:
        player_name:       Display name.
        impact_score:      Mean absolute state difference.
        form_diff:         Branch B - A mean form.
        fatigue_diff:      Branch B - A mean fatigue.
        understanding_diff: Branch B - A mean tactical understanding.
        trust_diff:        Branch B - A mean manager trust.
    """

    player_name: str
    impact_score: float
    form_diff: float
    fatigue_diff: float
    understanding_diff: float
    trust_diff: float


@dataclass(frozen=True)
class ReportInput:
    """Complete input for report generation.

    Assembles all data needed to generate a comparison report.
    Caller is responsible for constructing this from ComparisonResult,
    rank_player_impact(), and explain_action() outputs.

    Attributes:
        trigger_description:  Human-readable trigger description.
        points_mean_a:        Branch A mean total points.
        points_mean_b:        Branch B mean total points.
        points_mean_diff:     B - A mean points difference.
        cascade_count_diff:   Event type -> mean frequency diff (B - A).
        n_runs:               Number of simulation runs.
        player_impacts:       Top impacted players (fixed count).
        action_explanations:  TP action explanations with context.
        limitations:          Known simulation constraints.
    """

    trigger_description: str
    points_mean_a: float
    points_mean_b: float
    points_mean_diff: float
    cascade_count_diff: dict[str, float]
    n_runs: int
    player_impacts: list[PlayerImpactEntry]
    action_explanations: list[ActionExplanationEntry]
    limitations: list[str]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_report(
    client: LLMClient,
    report_input: ReportInput,
    *,
    system_prompt: str | None = None,
) -> str:
    """Generate a structured Markdown comparison report via LLM.

    Args:
        client:        LLMClient implementation.
        report_input:  Assembled report input data.
        system_prompt: Override the loaded system prompt (tests only).

    Returns:
        Markdown string with all required sections. Falls back to a
        structured fallback report if the LLM output is empty or
        missing required section headings.
    """
    prompt = (
        system_prompt if system_prompt is not None else _load_system_prompt()
    )

    payload = _build_payload(report_input)

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

    raw = client.complete(messages)

    if not raw or not raw.strip():
        return _FALLBACK_REPORT

    report = raw.strip()

    if not _has_required_sections(report):
        return _FALLBACK_REPORT

    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_payload(report_input: ReportInput) -> dict[str, Any]:
    """Build the JSON payload for the user message."""
    return {
        "trigger_description": report_input.trigger_description,
        "points_mean_a": round(report_input.points_mean_a, 2),
        "points_mean_b": round(report_input.points_mean_b, 2),
        "points_mean_diff": round(report_input.points_mean_diff, 2),
        "cascade_count_diff": {
            k: round(v, 3)
            for k, v in report_input.cascade_count_diff.items()
        },
        "n_runs": report_input.n_runs,
        "player_impacts": [
            {
                "player_name": p.player_name,
                "impact_score": round(p.impact_score, 3),
                "form_diff": round(p.form_diff, 3),
                "fatigue_diff": round(p.fatigue_diff, 3),
                "understanding_diff": round(p.understanding_diff, 3),
                "trust_diff": round(p.trust_diff, 3),
            }
            for p in report_input.player_impacts
        ],
        "action_explanations": [
            {
                "tp_type": a.tp_type,
                "action": a.action,
                "explanation": a.explanation,
                "label": a.label,
                "confidence_note": a.confidence_note,
            }
            for a in report_input.action_explanations
        ],
        "limitations": report_input.limitations,
    }


def _has_required_sections(report: str) -> bool:
    """Check that the report contains all required section headings."""
    for heading in REQUIRED_SECTIONS:
        if heading not in report:
            return False
    return True
