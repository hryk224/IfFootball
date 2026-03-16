"""Generate early validation signals from StructuredExplanation.

Produces a short list of observation points that help verify whether
the simulation's causal chain holds in the first few real matches
after the trigger event.

Design principles:
    - Rule-based only (no LLM). Same input = same signals.
    - Signals are observations, not recommendations.
    - Each signal links back to a CausalStep or PlayerImpact.
    - Confidence is determined by causal depth, not subjective judgment.
"""

from __future__ import annotations

from iffootball.simulation.structured_explanation import (
    StructuredExplanation,
    ValidationSignal,
)


# ---------------------------------------------------------------------------
# Event type -> observable metric mapping
# ---------------------------------------------------------------------------

# Maps simulation event types to StatsBomb-observable metrics.
# metric_when_increase: what direction the metric goes when this event fires.
_EVENT_OBSERVABLES: dict[str, dict[str, str]] = {
    "tactical_confusion": {
        "metric": "PPDA",
        "metric_direction": "increase",  # PPDA goes up = less pressing = confusion
        "reason": "Tactical confusion should manifest as higher PPDA "
                  "(less effective pressing) in early matches.",
        "window": "first 3 matches",
    },
    "adaptation_progress": {
        "metric": "Progressive Passes per 90",
        "metric_direction": "increase",
        "reason": "Adaptation progress should correlate with more progressive "
                  "passing as players adjust to the new system.",
        "window": "first 5 matches",
    },
    "form_drop": {
        "metric": "xG per 90",
        "metric_direction": "decrease",
        "reason": "Form drops should be visible as reduced attacking output.",
        "window": "first 3 matches",
    },
    "trust_decline": {
        "metric": "minutes played",
        "metric_direction": "decrease",
        "reason": "Trust decline should lead to reduced playing time "
                  "for the affected player.",
        "window": "first 5 matches",
    },
}

# Player impact axis -> observable metric mapping.
_AXIS_OBSERVABLES: dict[str, dict[str, str]] = {
    "form": {
        "metric": "xG involvement per 90",
        "reason": "Form changes should be visible in overall attacking "
                  "contribution metrics.",
        "window": "first 5 matches",
    },
    "trust": {
        "metric": "minutes played",
        "reason": "Trust changes directly affect selection priority "
                  "and playing time.",
        "window": "first 3 matches",
    },
}

# Confidence determination by causal depth.
_DEPTH_CONFIDENCE: dict[int, str] = {
    1: "high",
    2: "medium",
}
_DEFAULT_CONFIDENCE = "low"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_validation_signals(
    explanation: StructuredExplanation,
    *,
    max_signals: int = 3,
) -> tuple[ValidationSignal, ...]:
    """Generate early validation signals from a StructuredExplanation.

    Produces at most max_signals observation points, prioritized by
    confidence (depth 1 first, then depth 2, then highlights).

    Args:
        explanation: Completed StructuredExplanation.
        max_signals: Maximum number of signals to return.

    Returns:
        Tuple of ValidationSignals ordered by confidence (high first).
    """
    candidates: list[tuple[int, ValidationSignal]] = []

    # 1. Signals from causal chain steps.
    for step in explanation.causal_chain:
        observable = _EVENT_OBSERVABLES.get(step.event_type)
        if observable is None:
            continue

        confidence = _DEPTH_CONFIDENCE.get(step.depth, _DEFAULT_CONFIDENCE)
        # Priority: lower depth = higher priority.
        priority = step.depth

        signal = ValidationSignal(
            metric=observable["metric"],
            observation_window=observable["window"],
            metric_direction=observable["metric_direction"],
            hypothesis_support="supports",
            reason=observable["reason"],
            related_step_id=step.step_id,
            confidence=confidence,
        )
        candidates.append((priority, signal))

    # 2. Signals from player impacts (if not already covered by causal steps).
    covered_metrics = {c[1].metric for c in candidates}
    for pi in explanation.player_impacts:
        for change in pi.changes:
            if abs(change.diff) < 0.05:
                continue  # Skip negligible changes.
            axis_obs = _AXIS_OBSERVABLES.get(change.axis)
            if axis_obs is None:
                continue
            if axis_obs["metric"] in covered_metrics:
                continue

            direction = "increase" if change.diff > 0 else "decrease"
            signal = ValidationSignal(
                metric=f"{pi.player_name}: {axis_obs['metric']}",
                observation_window=axis_obs["window"],
                metric_direction=direction,
                hypothesis_support="supports",
                reason=f"{pi.player_name}'s {change.axis} changed by "
                       f"{change.diff:+.2f}. {axis_obs['reason']}",
                related_step_id=pi.related_step_ids[0] if pi.related_step_ids else None,
                confidence="medium",
            )
            candidates.append((3, signal))  # Lower priority than causal steps.
            covered_metrics.add(axis_obs["metric"])

    # Sort by priority (depth), take top max_signals.
    candidates.sort(key=lambda x: x[0])
    return tuple(c[1] for c in candidates[:max_signals])


# ---------------------------------------------------------------------------
# Text rendering (no LLM)
# ---------------------------------------------------------------------------


def render_signals_markdown(signals: tuple[ValidationSignal, ...]) -> str:
    """Render validation signals as a Markdown section.

    Produces a standalone "## What to Watch" section that can be
    appended to the report.
    """
    if not signals:
        return ""

    lines = ["## What to Watch", ""]
    for signal in signals:
        confidence_icon = {
            "high": "●",
            "medium": "◐",
            "low": "○",
        }.get(signal.confidence, "○")

        lines.append(
            f"- {confidence_icon} **{signal.metric}** — "
            f"Watch for {signal.metric_direction} "
            f"in the {signal.observation_window}. "
            f"{signal.reason} "
            f"({signal.confidence} confidence)"
        )

    return "\n".join(lines)
