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


# ---------------------------------------------------------------------------
# Structured label-carrying DTOs (v2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HighlightEntry:
    """A metric difference with code-assigned label and unit.

    Attributes:
        metric_name: Metric identifier (e.g. "total_points_mean").
        diff:        B - A difference (absolute value).
        label:       Code-assigned evidence label.
        statement:   Natural language description from StructuredExplanation.
        unit:        "points_mean" / "events_per_run".
        direction:   "increased" / "decreased" / "unchanged".
    """

    metric_name: str
    diff: float
    label: str
    statement: str
    unit: str
    direction: str


@dataclass(frozen=True)
class EvidenceEntry:
    """A single evidence item with label, source, and statement.

    Attributes:
        statement: Natural language description.
        label:     Code-assigned evidence label.
        source:    Origin of the evidence.
    """

    statement: str
    label: str
    source: str


@dataclass(frozen=True)
class CausalStepEntry:
    """A causal step with code-assigned labels and provenance.

    Attributes:
        step_id:         Unique step identifier for cross-referencing.
        cause:           Natural language cause (from LLM completion).
        effect:          Natural language effect (from LLM completion).
        affected_agent:  Player or manager name.
        event_type:      Event taxonomy type.
        depth:           Causal chain depth (1 = direct).
        paragraph_label: Label for the cause/effect paragraph as a whole.
                         Always >= "analysis" because cause/effect text
                         connects data points (never plain "data").
        evidence_labels: All evidence labels for this step.
        evidence:        Supporting evidence with labels, sources, and
                         statements for full provenance.
    """

    step_id: str
    cause: str
    effect: str
    affected_agent: str
    event_type: str
    depth: int
    paragraph_label: str
    evidence_labels: tuple[str, ...]
    evidence: tuple[EvidenceEntry, ...]


@dataclass(frozen=True)
class PlayerAxisChange:
    """Impact on a single dynamic state axis with label and statement.

    Attributes:
        axis:      "form" / "fatigue" / "understanding" / "trust".
        diff:      Branch B - A difference.
        label:     Code-assigned evidence label.
        statement: Natural language description from StructuredExplanation.
    """

    axis: str
    diff: float
    label: str
    statement: str


@dataclass(frozen=True)
class PlayerImpactDetailEntry:
    """Structured player impact with per-axis labels and statements.

    Changes are pre-filtered by the adapter:
    - Shared reset axes are removed (reported via PlayerImpactMeta).
    - At most 2 most significant axes are retained.

    Attributes:
        player_name:  Display name.
        impact_score: Mean absolute dynamic-state difference.
        changes:      Per-axis impact with labels and statements (max 2).
    """

    player_name: str
    impact_score: float
    changes: tuple[PlayerAxisChange, ...]


@dataclass(frozen=True)
class PlayerImpactMeta:
    """Metadata about shared patterns across all players.

    Attributes:
        shared_resets: Mapping from axis name to the common diff value.
                       Axes where all featured players share the same
                       non-trivial diff. These should be mentioned once
                       at the start of the Player Impact section, not
                       repeated per player.
    """

    shared_resets: dict[str, float]


@dataclass(frozen=True)
class ReportInput:
    """Complete input for report generation.

    Assembles all data needed to generate a comparison report.
    Caller is responsible for constructing this from ComparisonResult,
    rank_player_impact(), and explain_action() outputs.

    Attributes:
        trigger_description:    Human-readable trigger description.
        points_mean_a:          Branch A mean total points.
        points_mean_b:          Branch B mean total points.
        points_mean_diff:       B - A mean points difference.
        cascade_count_diff:     Event type -> mean frequency diff (B - A).
        n_runs:                 Number of simulation runs.
        player_impacts:         Top impacted players (legacy flat format).
        action_explanations:    TP action explanations with context.
        limitations:            Known simulation constraints.
        display_hints:          Planner display instructions.
        highlights:             Structured label-carrying highlights.
        causal_steps:           Structured label-carrying causal steps.
        player_impact_details:  Structured label-carrying player impacts.
        player_impact_meta:     Shared reset info (e.g. understanding reset).
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
    highlights: list[HighlightEntry] | None = None
    causal_steps: list[CausalStepEntry] | None = None
    player_impact_details: list[PlayerImpactDetailEntry] | None = None
    player_impact_meta: PlayerImpactMeta | None = None


# ---------------------------------------------------------------------------
# Validation debug info
# ---------------------------------------------------------------------------


@dataclass
class ValidationDebug:
    """Debug information from report validation.

    Captures validator results and failure details from both the initial
    generation and the retry attempt. Designed for diagnostic output,
    not for production control flow.

    Attributes:
        initial_issues:    Issues from the first generation attempt.
        retry_issues:      Issues from the retry attempt (empty if no retry).
        used_fallback:     Whether the final output is a fallback report.
        used_retry:        Whether the retry output was accepted.
        section_label_detail: Per-section label counts from sentence-level check.
    """

    initial_issues: list[str]
    retry_issues: list[str]
    used_fallback: bool
    used_retry: bool
    section_label_detail: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, object]:
        """Serialize to JSON-safe dict."""
        return {
            "initial_issues": self.initial_issues,
            "retry_issues": self.retry_issues,
            "used_fallback": self.used_fallback,
            "used_retry": self.used_retry,
            "section_label_detail": self.section_label_detail,
        }


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
    report, _ = generate_report_with_debug(
        client, report_input, system_prompt=system_prompt, lang=lang,
    )
    return report


def generate_report_with_debug(
    client: LLMClient,
    report_input: ReportInput,
    *,
    system_prompt: str | None = None,
    lang: str = "en",
) -> tuple[str, ValidationDebug]:
    """Generate a report and return validation debug info.

    Same as generate_report() but additionally returns a ValidationDebug
    object with validator results, failure details, and per-section
    label counts. Use this from diagnostic scripts to understand why
    reports fall back.

    Args:
        client:        LLMClient implementation.
        report_input:  Assembled report input data.
        system_prompt: Override the loaded system prompt (tests only).
        lang:          Output language ("en" or "ja").

    Returns:
        Tuple of (report_markdown, validation_debug).
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

    debug = ValidationDebug(
        initial_issues=[],
        retry_issues=[],
        used_fallback=False,
        used_retry=False,
        section_label_detail={},
    )

    raw = client.complete(messages)

    if not raw or not raw.strip():
        debug.used_fallback = True
        debug.initial_issues = ["empty_response"]
        return fallback, debug

    report = raw.strip()

    if not _has_required_sections(report, lang=lang, section_order=section_order):
        debug.used_fallback = True
        debug.initial_issues = ["missing_required_sections"]
        return fallback, debug

    # Post-process: strip quality notes and normalize expressions.
    report = _postprocess(report, lang)

    # Validate output quality. Retry once on failure.
    valid, issues = _is_valid_report(report, report_input, lang)
    debug.initial_issues = issues
    debug.section_label_detail = _section_label_detail(report, lang)

    if not valid:
        retry_report = _retry_once(
            client, messages, issues, lang,
            section_detail=debug.section_label_detail,
        )
        if retry_report is not None:
            retry_report = _postprocess(retry_report, lang)
            retry_valid, retry_issues = _is_valid_report(
                retry_report, report_input, lang,
            )
            debug.retry_issues = retry_issues
            if retry_valid:
                debug.used_retry = True
                debug.section_label_detail = _section_label_detail(
                    retry_report, lang,
                )
                return retry_report, debug
        # Retry also failed — fall back.
        debug.used_fallback = True
        return fallback, debug

    return report, debug


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
    if report_input.highlights is not None:
        payload["highlights"] = [
            {
                "metric_name": h.metric_name,
                "diff": round(h.diff, 3),
                "label": h.label,
                "statement": h.statement,
                "unit": h.unit,
                "direction": h.direction,
            }
            for h in report_input.highlights
        ]
    if report_input.causal_steps is not None:
        payload["causal_steps"] = [
            {
                "step_id": s.step_id,
                "cause": s.cause,
                "effect": s.effect,
                "affected_agent": s.affected_agent,
                "event_type": s.event_type,
                "depth": s.depth,
                "paragraph_label": s.paragraph_label,
                "evidence_labels": list(s.evidence_labels),
                "evidence": [
                    {
                        "statement": e.statement,
                        "label": e.label,
                        "source": e.source,
                    }
                    for e in s.evidence
                ],
            }
            for s in report_input.causal_steps
        ]
    if report_input.player_impact_details is not None:
        payload["player_impact_details"] = [
            {
                "player_name": p.player_name,
                "impact_score": round(p.impact_score, 3),
                "changes": [
                    {
                        "axis": c.axis,
                        "diff": round(c.diff, 4),
                        "label": c.label,
                        "statement": c.statement,
                    }
                    for c in p.changes
                ],
            }
            for p in report_input.player_impact_details
        ]
    if report_input.player_impact_meta is not None:
        meta = report_input.player_impact_meta
        payload["player_impact_meta"] = {
            "shared_resets": {
                axis: round(val, 4) for axis, val in meta.shared_resets.items()
            },
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

# Known Japanese typos from LLM output -> correct form.
_JA_TYPO_FIXES: dict[str, str] = {
    "戦頼度": "戦術理解度",
    "戦頼": "信頼",
    "理頼度": "信頼度",
}


def _fix_ja_typos(report: str) -> str:
    """Fix known Japanese typos that LLMs produce."""
    for wrong, correct in _JA_TYPO_FIXES.items():
        report = report.replace(wrong, correct)
    return report


# ---------------------------------------------------------------------------
# Sign/direction normalization
# ---------------------------------------------------------------------------

# EN: "decreased by -35.3" -> "decreased by 35.3"
#     "increased by -24.0" -> "decreased by 24.0"
_EN_DIRECTION_NEG = re.compile(
    r"\b((?:increased|decreased|dropped|risen|grew|fell|improved|worsened|changed))"
    r"(\s+by\s+)-(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

_EN_DIRECTION_FLIP: dict[str, str] = {
    "increased": "decreased",
    "decreased": "increased",
    "dropped": "risen",
    "risen": "dropped",
    "grew": "fell",
    "fell": "grew",
    "improved": "worsened",
    "worsened": "improved",
    "changed": "changed",
}


def _en_direction_fix(m: re.Match[str]) -> str:
    """Fix a single EN direction+negative match."""
    word = m.group(1)
    separator = m.group(2)
    number = m.group(3)
    lower = word.lower()
    if lower in ("decreased", "dropped", "fell", "worsened"):
        # "decreased by -X" -> "decreased by X" (remove double negative)
        return f"{word}{separator}{number}"
    flipped = _EN_DIRECTION_FLIP.get(lower, "changed")
    # Preserve original casing pattern (capitalize if original was).
    if word[0].isupper():
        flipped = flipped.capitalize()
    return f"{flipped}{separator}{number}"


def _normalize_signed_deltas_en(report: str) -> str:
    """Normalize English sign/direction double expressions.

    Fixes:
        "decreased by -35.3" -> "decreased by 35.3"
        "increased by -24.0" -> "decreased by 24.0"
    """
    return _EN_DIRECTION_NEG.sub(_en_direction_fix, report)


# JA: "-35.3 減少" -> "35.3 減少"
#     "-24.0 増加" -> "24.0 減少"
_JA_DIRECTION_NEG = re.compile(
    r"-(\d+(?:\.\d+)?)\s*(減少|増加|低下|上昇|悪化|改善|変化)"
)

_JA_DIRECTION_FLIP: dict[str, str] = {
    "減少": "減少",  # "-X 減少" -> "X 減少" (remove double negative)
    "低下": "低下",  # "-X 低下" -> "X 低下"
    "悪化": "悪化",  # "-X 悪化" -> "X 悪化"
    "増加": "減少",  # "-X 増加" -> "X 減少" (flip direction)
    "上昇": "低下",  # "-X 上昇" -> "X 低下"
    "改善": "悪化",  # "-X 改善" -> "X 悪化"
    "変化": "変化",  # "-X 変化" -> "X 変化"
}


def _normalize_signed_deltas_ja(report: str) -> str:
    """Normalize Japanese sign/direction double expressions.

    Fixes:
        "-35.3 減少" -> "35.3 減少"
        "-24.0 増加" -> "24.0 減少"
        "-35.3減少"  -> "35.3 減少"
    """

    def _ja_fix(m: re.Match[str]) -> str:
        number = m.group(1)
        direction = m.group(2)
        corrected = _JA_DIRECTION_FLIP.get(direction, "変化")
        return f"{number} {corrected}"

    return _JA_DIRECTION_NEG.sub(_ja_fix, report)


# ---------------------------------------------------------------------------
# Post-process wrapper
# ---------------------------------------------------------------------------


def _postprocess(report: str, lang: str) -> str:
    """Apply all post-processing fixes to a report."""
    report = _strip_quality_notes(report)
    if lang == "ja":
        report = _fix_ja_typos(report)
        report = _normalize_signed_deltas_ja(report)
    else:
        report = _normalize_signed_deltas_en(report)
    return report


# ---------------------------------------------------------------------------
# Output validators
# ---------------------------------------------------------------------------

_LABELS_EN = {"[data]", "[analysis]", "[hypothesis]"}
_LABELS_JA = {"[データ]", "[分析]", "[仮説]"}

# Sections that require per-sentence labels.
_LABELLED_SECTIONS = {"summary", "key_differences", "causal_chain", "player_impact"}

# Internal field names that should never appear in output text.
_INTERNAL_TOKENS = (
    "unit:",
    "direction:",
    "label:",
    "metric_name",
    "step_id",
    "causal_steps",
    "action_explanations",
    "display_hints",
    "paragraph_label",
    "evidence_labels",
    "player_impact_details",
    "player_impact_meta",
    "shared_resets",
)


def _extract_labelled_sections(report: str, lang: str) -> str:
    """Extract text from sections that should have per-sentence labels."""
    headings = _SECTION_HEADINGS.get(lang, _SECTION_HEADINGS["en"])
    limitations_heading = headings.get("limitations", "## Limitations")

    # Collect text from labelled sections only (exclude Limitations).
    labelled_headings = [
        headings[s] for s in _LABELLED_SECTIONS if s in headings
    ]

    parts: list[str] = []
    lines = report.split("\n")
    in_labelled = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            if stripped == limitations_heading:
                in_labelled = False
            elif any(stripped == h for h in labelled_headings):
                in_labelled = True
            else:
                in_labelled = False
            continue
        if in_labelled and stripped:
            parts.append(stripped)

    return "\n".join(parts)


def _has_sentence_level_labels(report: str, lang: str) -> bool:
    """Check that substantive sentences end with a label.

    Splits text into label-terminated units and checks that each
    substantive segment ends with a recognised label. Bullet items
    (lines starting with - or *) are treated as individual units.

    The expected pattern is: "Sentence text. [label]"
    Split happens *after* the closing bracket of each label.
    """
    labels = _LABELS_JA if lang == "ja" else _LABELS_EN
    text = _extract_labelled_sections(report, lang)
    if not text.strip():
        return True  # No labelled sections to check.

    # Split into label-terminated units.
    # Pattern: split after "]" followed by whitespace or newline.
    if lang == "ja":
        label_pat = r"\[(?:データ|分析|仮説)\]"
    else:
        label_pat = r"\[(?:data|analysis|hypothesis)\]"

    # Split after label closings.
    units = re.split(f"({label_pat})", text)

    # Reassemble pairs: [text, label, text, label, ...] -> ["text [label]", ...]
    segments: list[str] = []
    i = 0
    while i < len(units):
        part = units[i].strip()
        if i + 1 < len(units) and re.fullmatch(label_pat, units[i + 1]):
            segments.append(part + " " + units[i + 1])
            i += 2
        else:
            if part:
                segments.append(part)
            i += 1

    # For each segment, count how many sentences it contains and
    # whether it ends with a label.
    # "Sentence A. [data]" = 1 sentence, 1 label -> OK
    # "Sentence A. Sentence B. [data]" = 2 sentences, 1 label -> 1 missing
    # "Sentence A. Sentence B." = 2 sentences, 0 labels -> 2 missing
    if lang == "ja":
        sent_end = re.compile(r"。")
    else:
        # Match sentence-ending punctuation: a period/!/? followed by
        # a space+uppercase or end of string. Excludes decimals like 2.1.
        sent_end = re.compile(r"[.!?](?=\s+[A-Z]|\s*$)")

    labelled = 0
    unlabelled = 0

    for seg in segments:
        for line in seg.split("\n"):
            line = line.strip()
            if line.startswith(("- ", "* ")):
                line = line[2:].strip()
            if not line or len(line) < 5:
                continue

            has_label = any(line.endswith(lbl) for lbl in labels)

            # Count sentence-ending punctuation to estimate sentence count.
            # Exclude punctuation inside labels like "[data]".
            text_before_label = line
            for lbl in labels:
                if line.endswith(lbl):
                    text_before_label = line[: -len(lbl)].rstrip()
                    break

            n_sentences = len(sent_end.findall(text_before_label))
            if n_sentences == 0:
                n_sentences = 1  # At least one unit per line/bullet.

            if has_label:
                # The label covers the last sentence. Earlier ones are unlabelled.
                labelled += 1
                unlabelled += max(0, n_sentences - 1)
            else:
                unlabelled += n_sentences

    total = labelled + unlabelled
    if total == 0:
        return True

    # At most 1 unlabelled sentence is allowed (shared reset intro).
    return unlabelled <= 1


def _section_label_detail(report: str, lang: str) -> dict[str, dict[str, int]]:
    """Return per-section labelled/unlabelled sentence counts.

    Useful for diagnosing which section causes sentence-level label
    failures. Returns a dict: section_type -> {"labelled": N, "unlabelled": M}.
    """
    labels = _LABELS_JA if lang == "ja" else _LABELS_EN
    headings = _SECTION_HEADINGS.get(lang, _SECTION_HEADINGS["en"])
    limitations_heading = headings.get("limitations", "## Limitations")
    labelled_headings = {
        headings[s]: s for s in _LABELLED_SECTIONS if s in headings
    }

    if lang == "ja":
        label_pat = r"\[(?:データ|分析|仮説)\]"
        sent_end = re.compile(r"。")
    else:
        label_pat = r"\[(?:data|analysis|hypothesis)\]"
        sent_end = re.compile(r"[.!?](?=\s+[A-Z]|\s*$)")

    # Parse report into section blocks.
    lines = report.split("\n")
    current_section: str | None = None
    section_texts: dict[str, list[str]] = {}

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            if stripped == limitations_heading:
                current_section = None
            elif stripped in labelled_headings:
                current_section = labelled_headings[stripped]
                section_texts.setdefault(current_section, [])
            else:
                current_section = None
            continue
        if current_section and stripped:
            section_texts.setdefault(current_section, [])
            section_texts[current_section].append(stripped)

    result: dict[str, dict[str, int]] = {}
    for section_type, text_lines in section_texts.items():
        text = "\n".join(text_lines)
        units = re.split(f"({label_pat})", text)
        segments: list[str] = []
        i = 0
        while i < len(units):
            part = units[i].strip()
            if i + 1 < len(units) and re.fullmatch(label_pat, units[i + 1]):
                segments.append(part + " " + units[i + 1])
                i += 2
            else:
                if part:
                    segments.append(part)
                i += 1

        sec_labelled = 0
        sec_unlabelled = 0
        for seg in segments:
            for seg_line in seg.split("\n"):
                seg_line = seg_line.strip()
                if seg_line.startswith(("- ", "* ")):
                    seg_line = seg_line[2:].strip()
                if not seg_line or len(seg_line) < 5:
                    continue
                has_label = any(seg_line.endswith(lbl) for lbl in labels)
                text_before = seg_line
                for lbl in labels:
                    if seg_line.endswith(lbl):
                        text_before = seg_line[: -len(lbl)].rstrip()
                        break
                n_sent = len(sent_end.findall(text_before))
                if n_sent == 0:
                    n_sent = 1
                if has_label:
                    sec_labelled += 1
                    sec_unlabelled += max(0, n_sent - 1)
                else:
                    sec_unlabelled += n_sent

        result[section_type] = {
            "labelled": sec_labelled,
            "unlabelled": sec_unlabelled,
        }

    return result


# Patterns that combine multiple claims in one sentence.
_EN_MULTI_CLAIM = re.compile(
    r"\b(?:while|indicating|suggesting|which\s+(?:led|caused|resulted|suggests))"
    r"(?:\s|,)",
    re.IGNORECASE,
)
_JA_MULTI_CLAIM = re.compile(
    r"(?:一方で|し、|ことから|ため、|とともに|に加えて)"
)


def _extract_player_impact_text(report: str, lang: str) -> str:
    """Extract the Player Impact section text."""
    headings = _SECTION_HEADINGS.get(lang, _SECTION_HEADINGS["en"])
    pi_heading = headings.get("player_impact", "## Player Impact")
    if pi_heading not in report:
        return ""
    start = report.index(pi_heading) + len(pi_heading)
    rest = report[start:]
    next_heading = rest.find("\n## ")
    return rest[:next_heading] if next_heading != -1 else rest


def _has_no_multi_claim_sentences(report: str, lang: str) -> bool:
    """Check that Player Impact sentences don't combine multiple claims.

    Detects connecting words like "while", "indicating", "suggesting"
    (EN) or "一方で", "ことから" (JA) within labelled sentences in the
    Player Impact section. These patterns typically join data and
    interpretation in a single sentence, which violates the 1-claim-
    per-sentence rule.
    """
    pi_text = _extract_player_impact_text(report, lang)
    if not pi_text.strip():
        return True

    pattern = _JA_MULTI_CLAIM if lang == "ja" else _EN_MULTI_CLAIM

    # Check each non-empty line for combining patterns.
    violations = 0
    for line in pi_text.split("\n"):
        line = line.strip()
        if not line or len(line) < 10:
            continue
        if pattern.search(line):
            violations += 1

    # Allow at most 0 violations.
    return violations == 0


def _has_no_internal_metadata_leak(report: str) -> bool:
    """Check that no internal field names appear in report text."""
    lower = report.lower()
    for token in _INTERNAL_TOKENS:
        if token.lower() in lower:
            return False
    return True


def _has_valid_key_differences_format(report: str, lang: str) -> bool:
    """Check that Key Differences section is reader-facing, not raw key-value."""
    headings = _SECTION_HEADINGS.get(lang, _SECTION_HEADINGS["en"])
    kd_heading = headings.get("key_differences", "## Key Differences")

    if kd_heading not in report:
        return True  # Section not present (e.g. compact mode).

    # Extract Key Differences section text.
    start = report.index(kd_heading) + len(kd_heading)
    rest = report[start:]
    next_heading = rest.find("\n## ")
    section_text = rest[:next_heading] if next_heading != -1 else rest

    # Fail if raw key-value patterns appear.
    if re.search(r"\b(unit|direction|label)\s*:", section_text):
        return False

    # JA: check for unreadable number sequences in bullets.
    if lang == "ja":
        for line in section_text.split("\n"):
            line = line.strip()
            if not line.startswith(("- ", "* ")):
                continue
            numbers = re.findall(r"\d+(?:\.\d+)?", line)
            if len(numbers) >= 3:
                # 3+ numbers in a bullet: check for explanatory context.
                context_words = ("→", "（", "から", "ポイント", "回", "平均", "差")
                if not any(w in line for w in context_words):
                    return False

    return True


def _has_no_shared_reset_repetition(
    report: str, report_input: ReportInput, lang: str,
) -> bool:
    """Check that shared reset axes are not repeated per player.

    Allows one mention at the section start but fails if mentioned
    in individual player paragraphs.
    """
    meta = report_input.player_impact_meta
    if meta is None or not meta.shared_resets:
        return True

    headings = _SECTION_HEADINGS.get(lang, _SECTION_HEADINGS["en"])
    pi_heading = headings.get("player_impact", "## Player Impact")

    if pi_heading not in report:
        return True

    # Extract Player Impact section.
    start = report.index(pi_heading) + len(pi_heading)
    rest = report[start:]
    next_heading = rest.find("\n## ")
    section_text = rest[:next_heading] if next_heading != -1 else rest

    # Split into intro paragraph and per-player paragraphs.
    # The first paragraph (before any player name) is the intro.
    player_names = [p.player_name for p in report_input.player_impacts]
    if report_input.player_impact_details:
        player_names = [p.player_name for p in report_input.player_impact_details]

    # Keywords for shared reset axes.
    shared_keywords: list[str] = []
    for axis in meta.shared_resets:
        if axis == "understanding":
            shared_keywords.extend(["understanding", "戦術理解度", "理解度"])
        elif axis == "fatigue":
            shared_keywords.extend(["fatigue", "疲労"])
        elif axis == "form":
            shared_keywords.extend(["form", "フォーム"])
        elif axis == "trust":
            shared_keywords.extend(["trust", "信頼"])

    if not shared_keywords:
        return True

    # Find per-player text blocks.
    lines = section_text.split("\n")
    current_player: str | None = None
    player_blocks: dict[str, list[str]] = {}
    intro_done = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Check if this line starts a player section.
        matched_player = None
        for pn in player_names:
            if pn in stripped:
                matched_player = pn
                intro_done = True
                break
        if matched_player:
            current_player = matched_player
            player_blocks.setdefault(current_player, [])
            player_blocks[current_player].append(stripped)
        elif current_player and intro_done:
            player_blocks.setdefault(current_player, [])
            player_blocks[current_player].append(stripped)

    # Count shared keyword mentions per player.
    repetitions = 0
    for _player, block_lines in player_blocks.items():
        block_text = " ".join(block_lines)
        for kw in shared_keywords:
            if kw.lower() in block_text.lower():
                repetitions += 1
                break  # One hit per player is enough.

    # Fail if more than 1 player mentions shared keywords
    # (1 player mentioning it could be the intro overlap).
    return repetitions <= 1


def _has_expected_hypothesis_labels(
    report: str, report_input: ReportInput, lang: str,
) -> bool:
    """Check that hypothesis labels appear when expected.

    If causal_steps include paragraph_label="hypothesis", the output
    must contain at least one hypothesis label.

    Also checks for speculative language without hypothesis labels.
    """
    hyp_label = "[仮説]" if lang == "ja" else "[hypothesis]"
    has_hyp_label = hyp_label in report

    # Check 1: If input has hypothesis steps, output should too.
    if report_input.causal_steps:
        needs_hyp = any(
            s.paragraph_label == "hypothesis" for s in report_input.causal_steps
        )
        if needs_hyp and not has_hyp_label:
            return False

    # Check 2: Speculative language without any hypothesis label.
    if lang == "ja":
        speculative = re.search(
            r"(可能性がある|かもしれない|だろう|であろう|推測)", report
        )
    else:
        speculative = re.search(
            r"\b(likely|may|might|could|possibly|speculative)\b", report,
            re.IGNORECASE,
        )
    if speculative and not has_hyp_label:
        return False

    return True


# Event name mappings for direction checking.
_EVENT_NAMES_EN: dict[str, str] = {
    "adaptation_progress": "adaptation progress",
    "tactical_confusion": "tactical confusion",
    "form_drop": "form drop",
    "trust_decline": "trust decline",
    "squad_unrest": "squad unrest",
    "playing_time_change": "playing time change",
    "total_points_mean": "points",
}
_EVENT_NAMES_JA: dict[str, str] = {
    "adaptation_progress": "適応の進行",
    "tactical_confusion": "戦術的混乱",
    "form_drop": "フォーム低下",
    "trust_decline": "信頼低下",
    "squad_unrest": "チーム内不安",
    "playing_time_change": "出場機会の変化",
    "total_points_mean": "勝ち点",
}

_DIRECTION_WORDS_EN: dict[str, tuple[str, ...]] = {
    "increased": ("increased", "rose", "grew", "improved", "higher"),
    "decreased": ("decreased", "dropped", "fell", "declined", "reduced", "lower"),
}
_DIRECTION_WORDS_JA: dict[str, tuple[str, ...]] = {
    "increased": ("増加", "上昇", "改善", "向上"),
    "decreased": ("減少", "低下", "悪化"),
}


def _has_consistent_summary_directions(
    report: str,
    report_input: ReportInput,
    lang: str,
) -> bool:
    """Check that direction words in Summary match highlights.direction.

    For each highlight with a known event name, checks that if the Summary
    mentions that event, the direction word matches. Contradictions like
    "tactical confusion increased" when direction is "decreased" cause failure.
    """
    if not report_input.highlights:
        return True

    headings = _SECTION_HEADINGS.get(lang, _SECTION_HEADINGS["en"])
    summary_heading = headings.get("summary", "## Summary")
    if summary_heading not in report:
        return True

    # Extract Summary text.
    start = report.index(summary_heading) + len(summary_heading)
    rest = report[start:]
    next_heading = rest.find("\n## ")
    summary_text = (rest[:next_heading] if next_heading != -1 else rest).lower()

    event_names = _EVENT_NAMES_JA if lang == "ja" else _EVENT_NAMES_EN
    dir_words = _DIRECTION_WORDS_JA if lang == "ja" else _DIRECTION_WORDS_EN
    opposite = {"increased": "decreased", "decreased": "increased"}

    for hl in report_input.highlights:
        event_display = event_names.get(hl.metric_name, hl.metric_name.replace("_", " "))
        if event_display.lower() not in summary_text:
            continue  # Event not mentioned in summary — OK.

        expected_dir = hl.direction
        if expected_dir == "unchanged":
            continue

        wrong_dir = opposite.get(expected_dir)
        if wrong_dir is None:
            continue

        # Check all occurrences of the event name in the summary.
        # For each occurrence, extract the containing sentence and check
        # for wrong-direction words anywhere in that sentence (before or
        # after the event name). This catches both "confusion increased"
        # and "increased confusion" patterns.
        wrong_words = dir_words.get(wrong_dir, ())
        search_text = event_display.lower()
        search_start = 0
        while True:
            event_pos = summary_text.find(search_text, search_start)
            if event_pos == -1:
                break

            # Find the sentence boundaries around the event mention.
            if lang == "ja":
                sent_delim = "。"
            else:
                sent_delim = "."
            sentence_start = summary_text.rfind(sent_delim, 0, event_pos)
            sentence_start = sentence_start + 1 if sentence_start != -1 else 0
            sentence_end = summary_text.find(sent_delim, event_pos)
            if sentence_end == -1:
                sentence_end = len(summary_text)
            sentence = summary_text[sentence_start:sentence_end + 1]

            for word in wrong_words:
                if word in sentence:
                    return False

            search_start = event_pos + len(search_text)

    return True


def _is_valid_report(
    report: str,
    report_input: ReportInput,
    lang: str,
) -> tuple[bool, list[str]]:
    """Run all validators on a generated report.

    Returns (is_valid, list_of_issues). Issues are short strings
    suitable for inclusion in a retry prompt.
    """
    issues: list[str] = []

    if not _has_sentence_level_labels(report, lang):
        issues.append("missing sentence-level labels")

    if not _has_no_internal_metadata_leak(report):
        issues.append("internal metadata leaked")

    if not _has_valid_key_differences_format(report, lang):
        issues.append("invalid key differences format")

    if not _has_no_shared_reset_repetition(report, report_input, lang):
        issues.append("shared reset repeated per player")

    if not _has_expected_hypothesis_labels(report, report_input, lang):
        issues.append("hypothesis wording without hypothesis label")

    if not _has_no_multi_claim_sentences(report, lang):
        issues.append("multi-claim sentences in player impact")

    if not _has_consistent_summary_directions(report, report_input, lang):
        issues.append("summary direction contradicts input data")

    return (len(issues) == 0, issues)


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

_RETRY_INSTRUCTION_EN = (
    "Revise the report. Fix only these issues: {issues}. "
    "Keep the same facts and section structure."
)
_RETRY_INSTRUCTION_JA = (
    "レポートを修正してください。以下の問題のみ修正してください: {issues}。"
    "同じ事実とセクション構造を維持してください。"
)

# Section-specific label fix instructions.
_SECTION_LABEL_FIX_EN: dict[str, str] = {
    "summary": (
        "In Summary, ensure every sentence ends with its own label."
    ),
    "key_differences": (
        "In Key Differences, ensure every bullet ends with its own label."
    ),
    "causal_chain": (
        "In Causal Chain, ensure every sentence ends with its own label."
    ),
    "player_impact": (
        "In Player Impact, split multi-sentence paragraphs into separately "
        "labelled sentences. Each sentence must end with exactly one label. "
        "Do not combine data and interpretation in one sentence."
    ),
}

_SECTION_LABEL_FIX_JA: dict[str, str] = {
    "summary": (
        "「サマリー」で各文が文末ラベルを持つようにしてください。"
    ),
    "key_differences": (
        "「主な差分」で各項目が文末ラベルを持つようにしてください。"
    ),
    "causal_chain": (
        "「因果連鎖」で各文が文末ラベルを持つようにしてください。"
    ),
    "player_impact": (
        "「選手への影響」セクションの文末ラベルを修正してください。"
        "複数文を1つのラベルでまとめず、各文を独立してラベル付けしてください。"
        "数値と解釈を1文に混ぜないでください。"
    ),
}


def _build_retry_instruction(
    issues: list[str],
    section_detail: dict[str, dict[str, int]],
    lang: str,
) -> str:
    """Build a retry instruction with section-specific guidance."""
    template = _RETRY_INSTRUCTION_JA if lang == "ja" else _RETRY_INSTRUCTION_EN
    base = template.format(issues="; ".join(issues))

    # Add section-specific fixes for label issues.
    if "missing sentence-level labels" in issues:
        section_fixes = (
            _SECTION_LABEL_FIX_JA if lang == "ja" else _SECTION_LABEL_FIX_EN
        )
        fix_parts: list[str] = []
        for section, counts in section_detail.items():
            if counts.get("unlabelled", 0) > 0 and section in section_fixes:
                fix_parts.append(section_fixes[section])
        if fix_parts:
            base += " " + " ".join(fix_parts)

    # Add multi-claim fix for Player Impact.
    if "multi-claim sentences in player impact" in issues:
        if lang == "ja":
            base += (
                " 「選手への影響」で複数主張を1文にまとめないでください。"
                "「一方で」「〜し、」「〜ことから」を使って結合せず、"
                "各主張を別々の文にしてください。"
            )
        else:
            base += (
                " In Player Impact, do not combine multiple claims in one "
                "sentence. Split sentences joined by 'while', 'indicating', "
                "'suggesting', or 'which' into separate labelled sentences."
            )

    # Add summary direction fix.
    if "summary direction contradicts input data" in issues:
        if lang == "ja":
            base += (
                " サマリーで増減の方向語が入力データと矛盾しています。"
                "入力の direction フィールドに従ってください。"
            )
        else:
            base += (
                " In Summary, the direction words contradict the input data. "
                "Check each event: use 'increased' only when direction is "
                "'increased', and 'decreased' only when direction is 'decreased'."
            )

    return base


def _retry_once(
    client: "LLMClient",
    messages: list[dict[str, str]],
    issues: list[str],
    lang: str,
    section_detail: dict[str, dict[str, int]] | None = None,
) -> str | None:
    """Attempt one retry with issue feedback appended."""
    instruction = _build_retry_instruction(
        issues, section_detail or {}, lang,
    )

    retry_messages = list(messages) + [
        {"role": "user", "content": instruction},
    ]

    raw = client.complete(retry_messages)
    if not raw or not raw.strip():
        return None
    return raw.strip()


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
