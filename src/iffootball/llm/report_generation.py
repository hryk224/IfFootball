"""LLM-based comparison report generation.

Generates a structured Markdown report from simulation comparison results,
action explanations, and player impact data. The report includes labelled
sections ([data] / [analysis] / [hypothesis]) and a limitations section
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
from iffootball.simulation.report_planner import DisplayHints

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_PROMPT_DIR = Path(__file__).parents[3] / "prompts"

_PROMPT_PATHS: dict[str, Path] = {
    "en": _PROMPT_DIR / "report_generation_v1.md",
    "ja": _PROMPT_DIR / "report_generation_ja_v1.md",
}


def _load_system_prompt(path: Path | None = None, lang: str = "en") -> str:
    """Load system prompt from file. Raises FileNotFoundError if missing."""
    if path is not None:
        resolved = path
    else:
        resolved = _PROMPT_PATHS.get(lang, _PROMPT_PATHS["en"])
    return resolved.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Section type -> heading per language.
_SECTION_HEADINGS: dict[str, dict[str, str]] = {
    "en": {
        "summary": "## Summary",
        "key_differences": "## Key Differences",
        "causal_chain": "## Causal Chain",
        "player_impact": "## Player Impact",
        "limitations": "## Limitations",
    },
    "ja": {
        "summary": "## サマリー",
        "key_differences": "## 主な差分",
        "causal_chain": "## 因果連鎖",
        "player_impact": "## 選手への影響",
        "limitations": "## 制約事項",
    },
}

# Default section order (all sections).
_DEFAULT_SECTION_ORDER: tuple[str, ...] = (
    "summary",
    "key_differences",
    "causal_chain",
    "player_impact",
    "limitations",
)

# Required section headings per language (backward compat).
REQUIRED_SECTIONS: dict[str, tuple[str, ...]] = {
    lang: tuple(headings.values())
    for lang, headings in _SECTION_HEADINGS.items()
}

# Default limitations describing known simulation constraints.
DEFAULT_LIMITATIONS: dict[str, tuple[str, ...]] = {
    "en": (
        "Match outcomes use a Poisson model with xG-based expected goals; "
        "in-match events (shots, passes) are not simulated.",
        "Tactical metrics (PPDA, possession, progressive passes) for the "
        "incoming manager are estimates, not simulation outputs.",
        "Player technical attributes are fixed throughout the simulation; "
        "only dynamic state (form, fatigue, trust, understanding) changes.",
        "The action distribution at turning points is rule-based; "
        "LLM-based action selection is not yet implemented.",
        "xGA/90 is a fixed baseline; the current model does not simulate "
        "defensive impact of manager changes.",
    ),
    "ja": (
        "試合結果は xG ベースの Poisson モデルで決定されます。"
        "試合内イベント（シュート、パス）はシミュレートされません。",
        "後任監督の戦術指標（PPDA、ポゼッション、プログレッシブパス）は "
        "推定値であり、シミュレーション出力ではありません。",
        "選手の技術属性はシミュレーション中固定です。"
        "変化するのは動的状態（フォーム、疲労、信頼度、戦術理解度）のみです。",
        "ターニングポイントでの行動分布はルールベースです。"
        "LLM ベースの行動選択はまだ実装されていません。",
        "xGA/90 は固定ベースラインです。現在のモデルは "
        "監督交代による守備への影響をシミュレートしません。",
    ),
}

# Default number of top impacted players to include.
DEFAULT_TOP_PLAYERS = 3

# Fallback body text per section per language.
_FALLBACK_BODY: dict[str, dict[str, str]] = {
    "en": {
        "summary": "Unable to generate structured report.",
        "key_differences": "No data available.",
        "causal_chain": "No data available.",
        "player_impact": "No data available.",
        "limitations": "Report generation failed. Results may be incomplete.",
    },
    "ja": {
        "summary": "構造化レポートを生成できませんでした。",
        "key_differences": "データがありません。",
        "causal_chain": "データがありません。",
        "player_impact": "データがありません。",
        "limitations": "レポート生成に失敗しました。結果が不完全な可能性があります。",
    },
}


def _build_fallback(
    lang: str = "en",
    section_order: tuple[str, ...] | None = None,
) -> str:
    """Build a fallback report respecting the section order."""
    order = section_order or _DEFAULT_SECTION_ORDER
    headings = _SECTION_HEADINGS.get(lang, _SECTION_HEADINGS["en"])
    bodies = _FALLBACK_BODY.get(lang, _FALLBACK_BODY["en"])
    parts: list[str] = []
    for section in order:
        heading = headings.get(section)
        body = bodies.get(section, "No data available.")
        if heading:
            parts.append(f"{heading}\n\n{body}")
    return "\n\n".join(parts)


# Backward compat aliases.
_FALLBACK_REPORTS: dict[str, str] = {
    lang: _build_fallback(lang) for lang in _SECTION_HEADINGS
}
_FALLBACK_REPORT = _FALLBACK_REPORTS["en"]


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
        label:           "data" / "analysis" / "hypothesis".
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
    display_hints: DisplayHints | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_report(
    client: LLMClient,
    report_input: ReportInput,
    *,
    system_prompt: str | None = None,
    lang: str = "en",
) -> str:
    """Generate a structured Markdown comparison report via LLM.

    Args:
        client:        LLMClient implementation.
        report_input:  Assembled report input data.
        system_prompt: Override the loaded system prompt (tests only).
        lang:          Output language ("en" or "ja").

    Returns:
        Markdown string with all required sections. Falls back to a
        structured fallback report if the LLM output is empty or
        missing required section headings.
    """
    # Determine active sections from display_hints or default.
    section_order: tuple[str, ...] | None = None
    if report_input.display_hints is not None:
        section_order = report_input.display_hints.section_order

    fallback = _build_fallback(lang=lang, section_order=section_order)

    prompt = (
        system_prompt
        if system_prompt is not None
        else _load_system_prompt(lang=lang)
    )

    payload = _build_payload(report_input)

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

    raw = client.complete(messages)

    if not raw or not raw.strip():
        return fallback

    report = raw.strip()

    if not _has_required_sections(report, lang=lang, section_order=section_order):
        return fallback

    # Strip any quality notes or meta-commentary the LLM may have added.
    report = _strip_quality_notes(report)

    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_payload(report_input: ReportInput) -> dict[str, Any]:
    """Build the JSON payload for the user message."""
    payload: dict[str, Any] = {
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
    if report_input.display_hints is not None:
        payload["display_hints"] = {
            "section_order": list(report_input.display_hints.section_order),
            "expanded_step_ids": sorted(report_input.display_hints.expanded_step_ids),
            "collapsed_step_ids": sorted(report_input.display_hints.collapsed_step_ids),
            "featured_players": list(report_input.display_hints.featured_players),
            "show_limitations_info": report_input.display_hints.show_limitations_info,
        }
    return payload


def _strip_quality_notes(report: str) -> str:
    """Remove quality notes or meta-commentary from report end."""
    # Remove paragraphs starting with quality/confidence markers.
    lines = report.rstrip().split("\n")
    while lines:
        last = lines[-1].strip()
        if last.startswith("*Quality note:") or last.startswith("*品質注記:"):
            lines.pop()
            # Also remove preceding --- separator if present.
            if lines and lines[-1].strip() == "---":
                lines.pop()
            continue
        if not last:
            lines.pop()
            continue
        break
    return "\n".join(lines)


def _has_required_sections(
    report: str,
    lang: str = "en",
    section_order: tuple[str, ...] | None = None,
) -> bool:
    """Check that the report contains expected section headings.

    When section_order is provided (from DisplayHints), only those
    sections are required. Otherwise all default sections are required.
    """
    headings = _SECTION_HEADINGS.get(lang, _SECTION_HEADINGS["en"])
    if section_order is not None:
        required = [headings[s] for s in section_order if s in headings]
    else:
        required = list(headings.values())

    for heading in required:
        if heading not in report:
            return False
    return True


import re  # noqa: E402


def _validate_player_facts(report: str, report_input: ReportInput) -> bool:
    """Check for likely player value mix-ups in the report.

    Focuses on form_diff and trust_diff (high individual variance).
    Excludes understanding_diff (often shared across all players).
    Returns False if a likely misattribution is detected.
    """
    for p in report_input.player_impacts:
        if p.player_name not in report:
            continue

        # Extract text near this player's name (up to next ## or end).
        player_pattern = re.escape(p.player_name) + r"[^#]*?(?=\n##|\Z)"
        match = re.search(player_pattern, report, re.DOTALL)
        if not match:
            continue

        section_text = match.group(0)
        numbers_in_text = {
            abs(float(n)) for n in re.findall(r"-?\d+\.\d+", section_text)
        }

        # This player's differentiating values (form + trust only).
        player_values = {
            abs(round(p.form_diff, 2)),
            abs(round(p.trust_diff, 2)),
        }

        # Other players' form + trust values.
        other_values: set[float] = set()
        for other in report_input.player_impacts:
            if other.player_name == p.player_name:
                continue
            other_values |= {
                abs(round(other.form_diff, 2)),
                abs(round(other.trust_diff, 2)),
            }

        # Only flag values unique to other players.
        unique_other = other_values - player_values
        if numbers_in_text & unique_other:
            return False

    return True


# Japanese dangling sentence endings.
_JA_DANGLING_ENDINGS = re.compile(
    r"(が|けれど|しかし|一方で|ただし|ものの)\s*$", re.MULTILINE
)


def _has_dangling_sentences(report: str) -> bool:
    """Check for incomplete Japanese sentences ending with conjunctions."""
    return bool(_JA_DANGLING_ENDINGS.search(report))
