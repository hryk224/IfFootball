"""Preview report-generation output (English only).

Uses a fixed sample ReportInput and the configured LLM provider to
generate Markdown reports for prompt-quality review.

Supports two modes:
  - Legacy (default): builds ReportInput directly from fixed data.
  - Planner (--use-planner): builds a StructuredExplanation, runs
    plan_report(), and goes through the full adapter pipeline.

Usage:
    uv run python scripts/preview_report_generation.py
    uv run python scripts/preview_report_generation.py --stdout
    uv run python scripts/preview_report_generation.py --use-planner --display-context compact
    uv run python scripts/preview_report_generation.py --use-planner --repeat 3 --save-payload
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from iffootball.llm.providers import create_client
from iffootball.llm.report_adapter import structured_to_report_input
from iffootball.llm.report_generation import (
    DEFAULT_LIMITATIONS,
    ActionExplanationEntry,
    PlayerImpactEntry,
    ReportInput,
    generate_report_with_debug,
    _build_payload,
)
from iffootball.simulation.report_planner import (
    DisplayContext,
    plan_report,
)
from iffootball.simulation.structured_explanation import (
    SYSTEM_LIMITATIONS,
    CausalStep,
    DifferenceHighlight,
    EvidenceItem,
    LimitationCategory,
    LimitationItem,
    LimitationsDisclosure,
    PlayerImpactChange,
    PlayerImpactSummary,
    ScenarioDescriptor,
    StructuredExplanation,
)

_DEFAULT_OUTPUT_DIR = Path(__file__).parents[1] / "output" / "report_preview"


# ---------------------------------------------------------------------------
# Legacy ReportInput (no planner)
# ---------------------------------------------------------------------------


def _make_report_input() -> ReportInput:
    return ReportInput(
        trigger_description=(
            "Manager change: Louis van Gaal -> José Mario Felix dos Santos Mourinho "
            "at week 29"
        ),
        points_mean_a=12.2,
        points_mean_b=14.3,
        points_mean_diff=2.1,
        cascade_count_diff={
            "adaptation_progress": 24.0,
            "tactical_confusion": -35.3,
            "form_drop": 8.4,
        },
        n_runs=20,
        player_impacts=[
            PlayerImpactEntry(
                player_name="Juan Mata",
                impact_score=0.412,
                form_diff=-0.11,
                fatigue_diff=-0.04,
                understanding_diff=-0.25,
                trust_diff=-0.08,
            ),
            PlayerImpactEntry(
                player_name="Ander Herrera",
                impact_score=0.355,
                form_diff=0.03,
                fatigue_diff=0.08,
                understanding_diff=-0.25,
                trust_diff=0.12,
            ),
            PlayerImpactEntry(
                player_name="Morgan Schneiderlin",
                impact_score=0.331,
                form_diff=-0.13,
                fatigue_diff=0.01,
                understanding_diff=-0.25,
                trust_diff=-0.04,
            ),
        ],
        action_explanations=[
            ActionExplanationEntry(
                tp_type="bench_streak",
                action="resist",
                explanation=(
                    "Repeated benchings reduced trust and led the player to resist "
                    "the new manager's approach."
                ),
                label="analysis",
                confidence_note="Based on rule-based action distribution",
            ),
            ActionExplanationEntry(
                tp_type="low_understanding",
                action="adapt",
                explanation=(
                    "Despite initial confusion, the player tried to adjust to the "
                    "new tactical demands."
                ),
                label="analysis",
                confidence_note="Based on rule-based action distribution",
            ),
        ],
        limitations=list(DEFAULT_LIMITATIONS["en"]),
    )


# ---------------------------------------------------------------------------
# Fixed StructuredExplanation for planner mode
# ---------------------------------------------------------------------------


def _make_structured_explanation() -> StructuredExplanation:
    """Build a fixed StructuredExplanation matching the legacy sample data."""
    scenario = ScenarioDescriptor(
        trigger_type="manager_change",
        team_name="Manchester United",
        detail={
            "outgoing_manager": "Louis van Gaal",
            "incoming_manager": "José Mario Felix dos Santos Mourinho",
        },
    )

    highlights = (
        DifferenceHighlight(
            metric_name="total_points_mean",
            value_a=12.2,
            value_b=14.3,
            diff=2.1,
            interpretations=(
                EvidenceItem(
                    statement="Mean total points increased by 2.1 across 20 runs.",
                    label="data",
                    source="simulation_output",
                ),
            ),
        ),
        DifferenceHighlight(
            metric_name="adaptation_progress",
            value_a=0.0,
            value_b=24.0,
            diff=24.0,
            interpretations=(
                EvidenceItem(
                    statement="Adaptation progress events appeared 24.0 times per run on average.",
                    label="data",
                    source="simulation_output",
                ),
            ),
        ),
        DifferenceHighlight(
            metric_name="tactical_confusion",
            value_a=35.3,
            value_b=0.0,
            diff=-35.3,
            interpretations=(
                EvidenceItem(
                    statement="Tactical confusion decreased by 35.3 per run.",
                    label="data",
                    source="simulation_output",
                ),
            ),
        ),
        DifferenceHighlight(
            metric_name="form_drop",
            value_a=0.0,
            value_b=8.4,
            diff=8.4,
            interpretations=(
                EvidenceItem(
                    statement="Form drop events increased by 8.4 per run.",
                    label="data",
                    source="simulation_output",
                ),
            ),
        ),
    )

    causal_chain = (
        CausalStep(
            step_id="cs-001",
            cause="Manager change triggered a tactical reset.",
            effect="All players experienced a tactical understanding drop.",
            affected_agent="Juan Mata",
            event_type="tactical_confusion",
            evidence=(
                EvidenceItem(
                    statement="Understanding dropped by 0.25 for all players.",
                    label="data",
                    source="simulation_output",
                ),
            ),
            depth=1,
        ),
        CausalStep(
            step_id="cs-002",
            cause="Tactical confusion caused early-phase instability.",
            effect="Form dropped for players with low adaptability.",
            affected_agent="Juan Mata",
            event_type="form_drop",
            evidence=(
                EvidenceItem(
                    statement="Juan Mata form_diff = -0.11.",
                    label="data",
                    source="simulation_output",
                ),
            ),
            depth=2,
        ),
        CausalStep(
            step_id="cs-003",
            cause="Manager trust shifted toward pressing-oriented players.",
            effect="Ander Herrera gained trust and form.",
            affected_agent="Ander Herrera",
            event_type="adaptation_progress",
            evidence=(
                EvidenceItem(
                    statement="Ander Herrera trust_diff = +0.12.",
                    label="analysis",
                    source="rule_based_model",
                ),
            ),
            depth=2,
        ),
        CausalStep(
            step_id="cs-004",
            cause="Continued form drop led to reduced selection priority.",
            effect="Morgan Schneiderlin lost playing time.",
            affected_agent="Morgan Schneiderlin",
            event_type="form_drop",
            evidence=(
                EvidenceItem(
                    statement="Schneiderlin form_diff = -0.13, trust_diff = -0.04.",
                    label="hypothesis",
                    source="rule_based_model",
                ),
            ),
            depth=3,
        ),
    )

    player_impacts = (
        PlayerImpactSummary(
            player_name="Juan Mata",
            impact_score=0.412,
            changes=(
                PlayerImpactChange(
                    axis="form",
                    diff=-0.11,
                    interpretation=EvidenceItem(
                        statement="Form declined under new tactical demands.",
                        label="data",
                        source="simulation_output",
                    ),
                ),
                PlayerImpactChange(
                    axis="fatigue",
                    diff=-0.04,
                    interpretation=EvidenceItem(
                        statement="Slight fatigue decrease.",
                        label="data",
                        source="simulation_output",
                    ),
                ),
                PlayerImpactChange(
                    axis="understanding",
                    diff=-0.25,
                    interpretation=EvidenceItem(
                        statement="Tactical understanding reset due to manager change.",
                        label="data",
                        source="simulation_output",
                    ),
                ),
                PlayerImpactChange(
                    axis="trust",
                    diff=-0.08,
                    interpretation=EvidenceItem(
                        statement="Trust declined under new manager.",
                        label="data",
                        source="simulation_output",
                    ),
                ),
            ),
            related_step_ids=("cs-001", "cs-002"),
        ),
        PlayerImpactSummary(
            player_name="Ander Herrera",
            impact_score=0.355,
            changes=(
                PlayerImpactChange(
                    axis="form",
                    diff=0.03,
                    interpretation=EvidenceItem(
                        statement="Form slightly improved.",
                        label="data",
                        source="simulation_output",
                    ),
                ),
                PlayerImpactChange(
                    axis="fatigue",
                    diff=0.08,
                    interpretation=EvidenceItem(
                        statement="Fatigue increased with more playing time.",
                        label="data",
                        source="simulation_output",
                    ),
                ),
                PlayerImpactChange(
                    axis="understanding",
                    diff=-0.25,
                    interpretation=EvidenceItem(
                        statement="Tactical understanding reset.",
                        label="data",
                        source="simulation_output",
                    ),
                ),
                PlayerImpactChange(
                    axis="trust",
                    diff=0.12,
                    interpretation=EvidenceItem(
                        statement="Trust increased as pressing-oriented player.",
                        label="data",
                        source="simulation_output",
                    ),
                ),
            ),
            related_step_ids=("cs-003",),
        ),
        PlayerImpactSummary(
            player_name="Morgan Schneiderlin",
            impact_score=0.331,
            changes=(
                PlayerImpactChange(
                    axis="form",
                    diff=-0.13,
                    interpretation=EvidenceItem(
                        statement="Form dropped significantly.",
                        label="data",
                        source="simulation_output",
                    ),
                ),
                PlayerImpactChange(
                    axis="fatigue",
                    diff=0.01,
                    interpretation=EvidenceItem(
                        statement="Negligible fatigue change.",
                        label="data",
                        source="simulation_output",
                    ),
                ),
                PlayerImpactChange(
                    axis="understanding",
                    diff=-0.25,
                    interpretation=EvidenceItem(
                        statement="Tactical understanding reset.",
                        label="data",
                        source="simulation_output",
                    ),
                ),
                PlayerImpactChange(
                    axis="trust",
                    diff=-0.04,
                    interpretation=EvidenceItem(
                        statement="Trust slightly declined.",
                        label="data",
                        source="simulation_output",
                    ),
                ),
            ),
            related_step_ids=("cs-004",),
        ),
    )

    limitations = LimitationsDisclosure(
        system=SYSTEM_LIMITATIONS,
        scenario=(
            LimitationItem(
                category=LimitationCategory.CHAIN_DEPTH,
                message_en=(
                    "Causal chain reaches depth 3; effects beyond depth 2 "
                    "carry increasing uncertainty."
                ),
                message_ja=(
                    "因果連鎖が深さ 3 に達しています。"
                    "深さ 2 を超える効果は不確実性が増大します。"
                ),
                severity="warning",
                related_step_ids=("cs-004",),
            ),
            LimitationItem(
                category=LimitationCategory.ESTIMATION_DEPENDENCY,
                message_en=(
                    "Some evidence relies on rule-based models rather than "
                    "direct simulation output."
                ),
                message_ja=(
                    "一部の根拠がルールベースモデルに依存しており、"
                    "シミュレーション直接出力ではありません。"
                ),
                severity="info",
            ),
        ),
    )

    return StructuredExplanation(
        scenario=scenario,
        highlights=highlights,
        causal_chain=causal_chain,
        player_impacts=player_impacts,
        limitations=limitations,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DISPLAY_CONTEXTS = {
    "compact": DisplayContext.COMPACT,
    "standard": DisplayContext.STANDARD,
    "analyst": DisplayContext.ANALYST,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview LLM report-generation output."
    )
    # Language is EN-only (canonical output).
    # --lang argument removed; report is always in English.
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print the generated report(s) to stdout as well as saving files.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "gemini", "groq"],
        default=None,
        help="Explicit provider override. Defaults to env resolution.",
    )
    parser.add_argument(
        "--use-planner",
        action="store_true",
        help="Use ReportPlanner pipeline instead of legacy direct ReportInput.",
    )
    parser.add_argument(
        "--display-context",
        choices=list(_DISPLAY_CONTEXTS.keys()),
        default="standard",
        help="Display context for planner mode (default: standard).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Generate N reports under the same conditions (default: 1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (default: output/report_preview/).",
    )
    parser.add_argument(
        "--save-payload",
        action="store_true",
        help="Save the JSON payload sent to the LLM alongside each report.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _plan_to_dict(plan: object) -> dict:
    """Serialize a ReportPlan to a JSON-safe dict.

    Outputs a human-readable summary at the top, followed by the full
    plan details. The summary contains the fields most useful for
    quick review of planner decisions.
    """
    raw = asdict(plan)  # type: ignore[arg-type]
    # Convert frozenset to sorted list for JSON.
    if "expanded_step_ids" in raw:
        raw["expanded_step_ids"] = sorted(raw["expanded_step_ids"])
    if "collapsed_step_ids" in raw:
        raw["collapsed_step_ids"] = sorted(raw["collapsed_step_ids"])

    # Build a summary block at the top for quick review.
    lp = raw.get("limitation_placement", {})
    summary = {
        "section_order": [
            s["section_type"]
            for s in raw.get("sections", [])
            if s.get("include")
        ],
        "featured_players": list(raw.get("player_display_order", [])),
        "expanded_step_ids": raw.get("expanded_step_ids", []),
        "collapsed_step_ids": raw.get("collapsed_step_ids", []),
        "show_limitations_info": lp.get("include_info", False),
    }

    return {"_summary": summary, **raw}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    client = create_client(provider=args.provider)
    if client is None:
        raise SystemExit(
            "No LLM provider is available. Set API key env vars or .env first."
        )

    output_dir: Path = args.output_dir or _DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    langs = ["en"]
    context = _DISPLAY_CONTEXTS[args.display_context]
    repeat: int = max(1, args.repeat)

    mode = "planner" if args.use_planner else "legacy"
    print(f"Mode: {mode}")
    if args.use_planner:
        print(f"Display context: {args.display_context}")
    print(f"Repeat: {repeat}")
    print(f"Output: {output_dir}")
    print()

    for lang in langs:
        for run_idx in range(1, repeat + 1):
            suffix = f"_{run_idx:03d}" if repeat > 1 else ""
            ctx_label = f"_{args.display_context}" if args.use_planner else ""

            if args.use_planner:
                report_input, plan = _build_planner_input(context)
            else:
                report_input = _make_report_input()
                plan = None

            # Generate report with debug info.
            report, debug = generate_report_with_debug(
                client, report_input,
            )

            # Save report.
            report_path = output_dir / f"report_{lang}{ctx_label}{suffix}.md"
            report_path.write_text(report, encoding="utf-8")
            status = "PASS"
            if debug.used_fallback:
                status = "FALLBACK"
            elif debug.used_retry:
                status = "RETRY"
            print(f"[{lang}] report -> {report_path}  ({status})")

            # Always save debug info.
            debug_path = output_dir / f"debug_{lang}{ctx_label}{suffix}.json"
            debug_path.write_text(
                json.dumps(debug.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[{lang}] debug  -> {debug_path}")
            if debug.initial_issues:
                print(f"        initial issues: {debug.initial_issues}")
            if debug.retry_issues:
                print(f"        retry issues:   {debug.retry_issues}")

            # Save payload.
            if args.save_payload:
                payload = _build_payload(report_input)
                payload_path = output_dir / f"payload_{lang}{ctx_label}{suffix}.json"
                payload_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"[{lang}] payload -> {payload_path}")

            # Save plan.
            if plan is not None:
                plan_path = output_dir / f"plan_{lang}{ctx_label}{suffix}.json"
                plan_path.write_text(
                    json.dumps(_plan_to_dict(plan), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"[{lang}] plan -> {plan_path}")

            # Stdout output.
            if args.stdout:
                print()
                print("=" * 80)
                print(f"REPORT ({lang}{ctx_label}{suffix}) [{status}]")
                print("=" * 80)
                print(report)
                print()


def _build_planner_input(
    context: DisplayContext,
) -> tuple[ReportInput, object]:
    """Build ReportInput via the planner pipeline."""
    explanation = _make_structured_explanation()
    plan = plan_report(explanation, context)
    report_input = structured_to_report_input(
        explanation,
        plan=plan,
        n_runs=20,
    )
    return report_input, plan


if __name__ == "__main__":
    main()
