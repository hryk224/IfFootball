"""Preview report-generation output in English and Japanese.

Uses a fixed sample ReportInput and the configured LLM provider to
generate Markdown reports for prompt-quality review.

Usage:
    uv run python scripts/preview_report_generation.py
    uv run python scripts/preview_report_generation.py --lang ja
    uv run python scripts/preview_report_generation.py --stdout
"""

from __future__ import annotations

import argparse
from pathlib import Path

from iffootball.llm.providers import create_client
from iffootball.llm.report_generation import (
    DEFAULT_LIMITATIONS,
    ActionExplanationEntry,
    PlayerImpactEntry,
    ReportInput,
    generate_report,
)

OUTPUT_DIR = Path(__file__).parents[1] / "output" / "report_preview"


def _make_report_input(lang: str) -> ReportInput:
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
        limitations=list(DEFAULT_LIMITATIONS.get(lang, DEFAULT_LIMITATIONS["en"])),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview LLM report-generation output."
    )
    parser.add_argument(
        "--lang",
        choices=["en", "ja", "both"],
        default="both",
        help="Output language to generate.",
    )
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    client = create_client(provider=args.provider)
    if client is None:
        raise SystemExit(
            "No LLM provider is available. Set API key env vars or .env first."
        )

    langs = ["en", "ja"] if args.lang == "both" else [args.lang]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for lang in langs:
        report_input = _make_report_input(lang)
        report = generate_report(client, report_input, lang=lang)
        output_path = OUTPUT_DIR / f"report_{lang}.md"
        output_path.write_text(report, encoding="utf-8")

        print(f"[{lang}] saved to {output_path}")
        if args.stdout:
            print()
            print("=" * 80)
            print(f"REPORT ({lang})")
            print("=" * 80)
            print(report)
            print()


if __name__ == "__main__":
    main()
