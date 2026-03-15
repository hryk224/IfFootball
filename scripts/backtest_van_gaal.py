"""Backtest: Manchester United — Van Gaal dismissal at week 29.

Premier League 2015-16 (competition_id=2, season_id=27).
Scenario: What if Louis van Gaal was dismissed after match week 29?

Usage:
    uv run python scripts/backtest_van_gaal.py
"""

from __future__ import annotations

import json
from pathlib import Path

from iffootball.agents.trigger import ManagerChangeTrigger
from iffootball.collectors.statsbomb import StatsBombOpenDataCollector
from iffootball.config import SimulationRules
from iffootball.pipeline import initialize
from iffootball.simulation.comparison import run_comparison
from iffootball.simulation.turning_point import RuleBasedHandler
from iffootball.visualization.player_impact import rank_player_impact

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

COMPETITION_ID = 2
SEASON_ID = 27
TEAM_NAME = "Manchester United"
MANAGER_NAME = "Louis van Gaal"
TRIGGER_WEEK = 29
N_RUNS = 20
RNG_SEED = 42

RULES_DIR = Path(__file__).parents[1] / "config" / "simulation_rules"
OUTPUT_DIR = Path(__file__).parents[1] / "output" / "backtest_van_gaal"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("IfFootball Backtest: Manchester United — Van Gaal Dismissal")
    print("=" * 60)

    # 1. Initialize
    print("\n[1/4] Initializing from StatsBomb data...")
    collector = StatsBombOpenDataCollector()
    init_result = initialize(
        collector=collector,
        competition_id=COMPETITION_ID,
        season_id=SEASON_ID,
        team_name=TEAM_NAME,
        manager_name=MANAGER_NAME,
        trigger_week=TRIGGER_WEEK,
        league_name="Premier League",
    )

    print(f"  Squad size: {len(init_result.player_agents)}")
    print(f"  Fixtures remaining: {len(init_result.fixture_list.fixtures)}")
    print(f"  Team xG for/against: {init_result.team_baseline.xg_for_per90:.2f} / "
          f"{init_result.team_baseline.xg_against_per90:.2f}")
    print(f"  Manager: {init_result.manager_agent.manager_name}")
    print(f"  Pressing intensity: {init_result.manager_agent.pressing_intensity:.1f}")
    print(f"  Possession preference: {init_result.manager_agent.possession_preference:.3f}")
    print(f"  Preferred formation: {init_result.manager_agent.preferred_formation}")

    # 2. Run comparison
    print(f"\n[2/4] Running {N_RUNS}-run comparison (seed={RNG_SEED})...")
    trigger = ManagerChangeTrigger(
        outgoing_manager_name=MANAGER_NAME,
        incoming_manager_name="New Manager",
        transition_type="mid_season",
        applied_at=TRIGGER_WEEK,
    )

    rules = SimulationRules.load(RULES_DIR)
    handler = RuleBasedHandler(rules)

    comparison = run_comparison(
        team=init_result.team_baseline,
        squad=init_result.player_agents,
        manager=init_result.manager_agent,
        fixture_list=init_result.fixture_list,
        opponent_strengths=init_result.opponent_strengths,
        rules=rules,
        handler=handler,
        trigger=trigger,
        n_runs=N_RUNS,
        rng_seed=RNG_SEED,
    )

    # 3. Results summary
    print("\n[3/4] Results Summary")
    print("-" * 40)
    a = comparison.no_change
    b = comparison.with_change
    d = comparison.delta

    print(f"  Branch A (no change):  mean={a.total_points_mean:.1f}  "
          f"median={a.total_points_median:.1f}  std={a.total_points_std:.2f}")
    print(f"  Branch B (dismissed):  mean={b.total_points_mean:.1f}  "
          f"median={b.total_points_median:.1f}  std={b.total_points_std:.2f}")
    print(f"  Delta (B - A):         mean={d.points_mean_diff:+.1f}  "
          f"median={d.points_median_diff:+.1f}")

    print("\n  Cascade event frequency diff (B - A):")
    for et, diff in sorted(d.cascade_count_diff.items()):
        print(f"    {et}: {diff:+.2f} per run")

    # 4. Player impact
    print("\n[4/4] Top 5 Impacted Players")
    print("-" * 40)
    impacts = rank_player_impact(comparison, top_n=5)
    for i, p in enumerate(impacts, 1):
        print(f"  {i}. {p.player_name} (impact={p.impact_score:.3f})")
        print(f"     form:  A={p.mean_form_a:.3f}  B={p.mean_form_b:.3f}  "
              f"diff={p.mean_form_b - p.mean_form_a:+.3f}")
        print(f"     fatigue: A={p.mean_fatigue_a:.3f}  B={p.mean_fatigue_b:.3f}  "
              f"diff={p.mean_fatigue_b - p.mean_fatigue_a:+.3f}")
        print(f"     understanding: A={p.mean_understanding_a:.3f}  B={p.mean_understanding_b:.3f}  "
              f"diff={p.mean_understanding_b - p.mean_understanding_a:+.3f}")
        print(f"     trust: A={p.mean_trust_a:.3f}  B={p.mean_trust_b:.3f}  "
              f"diff={p.mean_trust_b - p.mean_trust_a:+.3f}")

    # Save results as JSON
    results = {
        "scenario": {
            "team": TEAM_NAME,
            "manager": MANAGER_NAME,
            "trigger_week": TRIGGER_WEEK,
            "n_runs": N_RUNS,
            "rng_seed": RNG_SEED,
        },
        "branch_a": {
            "points_mean": a.total_points_mean,
            "points_median": a.total_points_median,
            "points_std": a.total_points_std,
            "cascade_event_counts": a.cascade_event_counts,
        },
        "branch_b": {
            "points_mean": b.total_points_mean,
            "points_median": b.total_points_median,
            "points_std": b.total_points_std,
            "cascade_event_counts": b.cascade_event_counts,
        },
        "delta": {
            "points_mean_diff": d.points_mean_diff,
            "points_median_diff": d.points_median_diff,
            "cascade_count_diff": d.cascade_count_diff,
        },
        "top_5_impacts": [
            {
                "player_name": p.player_name,
                "impact_score": p.impact_score,
                "form_diff": p.mean_form_b - p.mean_form_a,
                "fatigue_diff": p.mean_fatigue_b - p.mean_fatigue_a,
                "understanding_diff": p.mean_understanding_b - p.mean_understanding_a,
                "trust_diff": p.mean_trust_b - p.mean_trust_a,
            }
            for p in impacts
        ],
    }

    output_path = OUTPUT_DIR / "results.json"
    with output_path.open("w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {output_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
