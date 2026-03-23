"""Run all demo presets under identical conditions for M9 review.

Executes each DemoPreset through the full pipeline (initialize ->
comparison -> player impact) and writes per-preset summaries plus a
cross-preset overview.

Usage:
    uv run python scripts/preview_presets.py
    uv run python scripts/preview_presets.py --n-runs 5 --seed 99
    uv run python scripts/preview_presets.py --with-report

Output:
    output/preset_preview/
        01_van_gaal_to_mourinho/
            summary.txt
            comparison.json
            report.md          (only with --with-report and LLM configured)
        02_van_gaal_to_klopp/
            ...
        overview.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import asdict
from pathlib import Path

from iffootball.agents.player import SampleTier
from iffootball.agents.trigger import ManagerChangeTrigger
from iffootball.collectors.statsbomb import StatsBombOpenDataCollector
from iffootball.config import SimulationRules
from iffootball.incoming_profile import resolve_incoming_profile
from iffootball.pipeline import InitializationResult, initialize
from iffootball.presets import COMPETITION_ID, DEMO_PRESETS, SEASON_ID, DemoPreset
from iffootball.storage.db import Database
from iffootball.simulation.comparison import RNG_POLICY, run_comparison
from iffootball.simulation.turning_point import RuleBasedHandler
from iffootball.visualization.player_impact import rank_player_impact

_RULES_DIR = Path(__file__).parents[1] / "config" / "simulation_rules"
_DEFAULT_OUTPUT_DIR = Path(__file__).parents[1] / "output" / "preset_preview"
_CACHE_DIR = Path(__file__).parents[1] / "data" / "demo_cache"
_DEFAULT_N_RUNS = 20
_DEFAULT_SEED = 42
_TOP_PLAYERS = 5


# ---------------------------------------------------------------------------
# Per-preset execution
# ---------------------------------------------------------------------------


def _cache_path(preset: DemoPreset) -> Path:
    """Return the per-preset cache DB path."""
    safe_name = preset.team_name.lower().replace(" ", "_")
    return _CACHE_DIR / f"{safe_name}_w{preset.trigger_week}.db"


def _load_from_cache(preset: DemoPreset) -> InitializationResult | None:
    """Try to load initialization data from demo cache DB."""
    path = _cache_path(preset)
    if not path.exists():
        return None
    try:
        db = Database(path)
    except Exception:
        return None
    try:
        agents = db.load_player_agents(COMPETITION_ID, SEASON_ID)
        if not agents:
            return None
        baseline = db.load_team_baseline(
            preset.team_name, COMPETITION_ID, SEASON_ID
        )
        if baseline is None:
            return None
        manager = db.load_manager_agent(
            preset.manager_name, preset.team_name, COMPETITION_ID, SEASON_ID
        )
        if manager is None:
            return None
        fixtures = db.load_fixture_list(
            preset.team_name, COMPETITION_ID, SEASON_ID,
        )
        if fixtures is None:
            return None
        opponents = db.load_opponent_strengths(
            COMPETITION_ID, SEASON_ID
        )
        if not opponents:
            return None
        league = db.load_league_context(COMPETITION_ID, SEASON_ID)
        if league is None:
            return None
        return InitializationResult(
            player_agents=agents,
            team_baseline=baseline,
            manager_agent=manager,
            fixture_list=fixtures,
            opponent_strengths=opponents,
            league_context=league,
        )
    except Exception:
        return None
    finally:
        db.close()


def _save_to_cache(preset: DemoPreset, init: InitializationResult) -> None:
    """Save initialization data to demo cache DB."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(preset)
    # Remove stale cache with old schema.
    if path.exists():
        path.unlink()
    db = Database(path)
    try:
        db.save_player_agents(init.player_agents, COMPETITION_ID, SEASON_ID)
        db.save_team_baseline(init.team_baseline)
        db.save_manager_agent(init.manager_agent)
        db.save_fixture_list(init.fixture_list, COMPETITION_ID, SEASON_ID)
        db.save_opponent_strengths(
            init.opponent_strengths, COMPETITION_ID, SEASON_ID,
        )
        db.save_league_context(init.league_context)
    finally:
        db.close()


def _run_preset(
    preset: DemoPreset,
    rules: SimulationRules,
    n_runs: int,
    seed: int,
) -> dict:
    """Run a single preset and return summary data."""
    print(f"  Initializing {preset.team_name} ({preset.manager_name})...")

    # Try cache first.
    init_result = _load_from_cache(preset)
    if init_result is not None:
        print("  (loaded from cache)")
    else:
        collector = StatsBombOpenDataCollector()
        init_result = initialize(
            collector=collector,
            competition_id=COMPETITION_ID,
            season_id=SEASON_ID,
            team_name=preset.team_name,
            manager_name=preset.manager_name,
            trigger_week=preset.trigger_week,
            league_name=f"Competition {COMPETITION_ID}",
        )
        _save_to_cache(preset, init_result)
        print("  (saved to cache)")

    # Resolve incoming manager profile.
    incoming_profile = resolve_incoming_profile(
        name=preset.incoming_manager_name,
        competition_id=COMPETITION_ID,
        season_id=SEASON_ID,
    )

    trigger = ManagerChangeTrigger(
        outgoing_manager_name=preset.manager_name,
        incoming_manager_name=preset.incoming_manager_name,
        transition_type="mid_season",
        applied_at=preset.trigger_week,
        incoming_profile=incoming_profile,
    )

    handler = RuleBasedHandler(rules)

    print(f"  Running comparison (n_runs={n_runs}, seed={seed})...")
    comparison = run_comparison(
        team=init_result.team_baseline,
        squad=init_result.player_agents,
        manager=init_result.manager_agent,
        fixture_list=init_result.fixture_list,
        opponent_strengths=init_result.opponent_strengths,
        rules=rules,
        handler=handler,
        trigger=trigger,
        n_runs=n_runs,
        rng_seed=seed,
    )

    impacts = rank_player_impact(comparison, top_n=_TOP_PLAYERS)

    # Count squad tiers.
    full_count = sum(
        1 for p in init_result.player_agents if p.sample_tier == SampleTier.FULL
    )
    partial_count = sum(
        1 for p in init_result.player_agents if p.sample_tier == SampleTier.PARTIAL
    )

    return {
        "preset": preset,
        "comparison": comparison,
        "impacts": impacts,
        "init_result": init_result,
        "squad_full": full_count,
        "squad_partial": partial_count,
    }


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------


def _write_summary(data: dict, output_dir: Path) -> None:
    """Write summary.txt for a single preset."""
    preset: DemoPreset = data["preset"]
    comp = data["comparison"]
    impacts = data["impacts"]

    lines = [
        f"Preset: {preset.label} (week {preset.trigger_week})",
        f"Points: {comp.no_change.total_points_mean:.1f} -> "
        f"{comp.with_change.total_points_mean:.1f} "
        f"({comp.delta.points_mean_diff:+.1f})",
        f"Squad size: {data['squad_full']} (full) + {data['squad_partial']} (partial)",
        "",
        "Featured players:",
    ]
    for i, p in enumerate(impacts, 1):
        lines.append(
            f"  {i}. {p.player_name:25s} impact={p.impact_score:.3f}  "
            f"form_diff={p.mean_form_b - p.mean_form_a:+.3f}  "
            f"tier={p.sample_tier.value}"
        )

    lines.append("")
    lines.append("Cascade events (B, mean per run):")
    for et in sorted(comp.with_change.cascade_event_counts):
        count = comp.with_change.cascade_event_counts[et]
        lines.append(f"  {et}: {count:.1f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_comparison_json(data: dict, output_dir: Path) -> None:
    """Write comparison.json with key metrics."""
    comp = data["comparison"]
    payload = {
        "points_mean_a": comp.no_change.total_points_mean,
        "points_mean_b": comp.with_change.total_points_mean,
        "points_mean_diff": comp.delta.points_mean_diff,
        "points_std_a": comp.no_change.total_points_std,
        "points_std_b": comp.with_change.total_points_std,
        "cascade_event_counts_a": comp.no_change.cascade_event_counts,
        "cascade_event_counts_b": comp.with_change.cascade_event_counts,
        "cascade_count_diff": comp.delta.cascade_count_diff,
        "n_runs": comp.no_change.n_runs,
        "rng_policy": RNG_POLICY,
        "squad_full": data["squad_full"],
        "squad_partial": data["squad_partial"],
        "featured_players": [
            {
                "player_name": p.player_name,
                "impact_score": round(p.impact_score, 4),
                "form_diff": round(p.mean_form_b - p.mean_form_a, 4),
                "fatigue_diff": round(p.mean_fatigue_b - p.mean_fatigue_a, 4),
                "understanding_diff": round(
                    p.mean_understanding_b - p.mean_understanding_a, 4
                ),
                "trust_diff": round(p.mean_trust_b - p.mean_trust_a, 4),
                "sample_tier": p.sample_tier.value,
            }
            for p in data["impacts"]
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_overview(all_data: list[dict], output_dir: Path, seed: int, n_runs: int) -> None:
    """Write overview.txt with cross-preset comparison."""
    lines = [
        f"M9 Preset Preview — seed={seed}, n_runs={n_runs}, rng_policy={RNG_POLICY}",
        "",
        f"{'Preset':<30s} {'Pts A':>6s} {'Pts B':>6s} {'Diff':>6s} "
        f"{'Top Player':<25s} {'Impact':>7s} {'Squad':>12s}",
        "-" * 100,
    ]
    for data in all_data:
        preset: DemoPreset = data["preset"]
        comp = data["comparison"]
        top = data["impacts"][0] if data["impacts"] else None
        top_name = top.player_name if top else "N/A"
        top_impact = f"{top.impact_score:.3f}" if top else "N/A"
        squad_str = f"{data['squad_full']}+{data['squad_partial']}p"
        lines.append(
            f"{preset.label:<30s} {comp.no_change.total_points_mean:>6.1f} "
            f"{comp.with_change.total_points_mean:>6.1f} "
            f"{comp.delta.points_mean_diff:>+6.1f} "
            f"{top_name:<25s} {top_impact:>7s} {squad_str:>12s}"
        )

    lines.append("")
    lines.append("Cascade events (B, mean per run):")
    # Collect all event types across presets.
    all_types: set[str] = set()
    for data in all_data:
        all_types.update(data["comparison"].with_change.cascade_event_counts)

    header = f"  {'Event':<30s}"
    for data in all_data:
        header += f" {data['preset'].slug[:20]:>20s}"
    lines.append(header)

    for et in sorted(all_types):
        row = f"  {et:<30s}"
        for data in all_data:
            count = data["comparison"].with_change.cascade_event_counts.get(et, 0.0)
            row += f" {count:>20.1f}"
        lines.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "overview.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run demo presets for M9 review.")
    parser.add_argument("--n-runs", type=int, default=_DEFAULT_N_RUNS)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument("--output-dir", type=str, default=str(_DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--with-report",
        action="store_true",
        help="Generate LLM reports (requires LLM provider configured).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rules = SimulationRules.load(_RULES_DIR)

    print(f"Preset Preview Runner")
    print(f"  Presets: {len(DEMO_PRESETS)}")
    print(f"  n_runs={args.n_runs}, seed={args.seed}, rng_policy={RNG_POLICY}")
    print(f"  Output: {output_dir}")
    print()

    all_data: list[dict] = []

    for i, preset in enumerate(DEMO_PRESETS, 1):
        print(f"[{i}/{len(DEMO_PRESETS)}] {preset.label}")
        data = _run_preset(preset, rules, args.n_runs, args.seed)
        all_data.append(data)

        preset_dir = output_dir / f"{i:02d}_{preset.slug}"
        _write_summary(data, preset_dir)
        _write_comparison_json(data, preset_dir)
        print(f"  -> {preset_dir}/summary.txt")
        print(f"  -> {preset_dir}/comparison.json")

        # Optional LLM report.
        if args.with_report:
            try:
                from iffootball.llm.providers import create_client
                from iffootball.llm.report_generation import generate_report
                from iffootball.llm.explanation_completion import complete_skeleton
                from iffootball.simulation.skeleton_builder import build_skeleton
                from iffootball.simulation.report_planner import plan_report, DisplayContext
                from iffootball.llm.report_adapter import structured_to_report_input

                client = create_client()
                trigger_for_skeleton = ManagerChangeTrigger(
                    outgoing_manager_name=preset.manager_name,
                    incoming_manager_name=preset.incoming_manager_name,
                    transition_type="mid_season",
                    applied_at=preset.trigger_week,
                )
                skeleton = build_skeleton(
                    data["comparison"],
                    trigger=trigger_for_skeleton,
                    team_name=preset.team_name,
                    impacts=data["impacts"],
                )
                explanation = complete_skeleton(client, skeleton)
                plan = plan_report(explanation, DisplayContext.STANDARD)
                report_input = structured_to_report_input(
                    explanation, plan=plan, n_runs=args.n_runs
                )
                report_md = generate_report(client, report_input)
                report_path = preset_dir / "report.md"
                report_path.write_text(report_md, encoding="utf-8")
                print(f"  -> {report_path}")
            except Exception as e:
                print(f"  [SKIP report] {e}")

        print()

    # Write overview.
    _write_overview(all_data, output_dir, args.seed, args.n_runs)
    print(f"Overview -> {output_dir / 'overview.txt'}")


if __name__ == "__main__":
    main()
