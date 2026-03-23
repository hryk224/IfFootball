"""Inspect where incoming manager profile differences affect simulation.

Compares two incoming managers for the same scenario to find where
the profile diff enters the simulation and where it gets absorbed.

Usage:
    uv run python scripts/inspect_manager_sensitivity.py
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from iffootball.agents.manager import ManagerAgent
from iffootball.collectors.statsbomb import StatsBombOpenDataCollector
from iffootball.config import SimulationRules
from iffootball.incoming_profile import resolve_incoming_profile
from iffootball.pipeline import InitializationResult, initialize
from iffootball.simulation.comparison import run_comparison
from iffootball.simulation.turning_point import RuleBasedHandler
from iffootball.agents.trigger import ManagerChangeTrigger
from iffootball.storage.db import Database
from iffootball.visualization.player_impact import rank_player_impact

_CACHE_DIR = Path(__file__).parents[1] / "data" / "demo_cache"
_RULES_DIR = Path(__file__).parents[1] / "config" / "simulation_rules"

COMP_ID = 2
SEASON_ID = 27
TEAM = "Manchester United"
MANAGER = "Louis van Gaal"
TRIGGER_WEEK = 29

MANAGERS = {
    "Mourinho": "José Mario Felix dos Santos Mourinho",
    "Pochettino": "Mauricio Roberto Pochettino Trossero",
}


def _load_init() -> InitializationResult:
    """Load initialization from cache."""
    cache_path = _CACHE_DIR / "manchester_united_w29.db"
    db = Database(cache_path)
    try:
        player_agents = db.load_player_agents(COMP_ID, SEASON_ID)
        team_baseline = db.load_team_baseline(TEAM, COMP_ID, SEASON_ID)
        manager_agent = db.load_manager_agent(MANAGER, TEAM, COMP_ID, SEASON_ID)
        fixture_list = db.load_fixture_list(TEAM, COMP_ID, SEASON_ID)
        opponent_strengths = db.load_opponent_strengths(COMP_ID, SEASON_ID)
        league_context = db.load_league_context(COMP_ID, SEASON_ID)
        assert player_agents and team_baseline and manager_agent and fixture_list
        assert opponent_strengths and league_context
        return InitializationResult(
            player_agents=player_agents,
            team_baseline=team_baseline,
            manager_agent=manager_agent,
            fixture_list=fixture_list,
            opponent_strengths=opponent_strengths,
            league_context=league_context,
        )
    finally:
        db.close()


def _print_profile_diff(m1: ManagerAgent, m2: ManagerAgent, name1: str, name2: str) -> None:
    """Print a side-by-side comparison of two manager profiles."""
    print("=" * 70)
    print(f"MANAGER PROFILE COMPARISON: {name1} vs {name2}")
    print("=" * 70)

    attrs = [
        "pressing_intensity",
        "possession_preference",
        "counter_tendency",
        "preferred_formation",
        "implementation_speed",
        "style_stubbornness",
        "youth_development",
        "rotation_skill",
    ]

    for attr in attrs:
        v1 = getattr(m1, attr, "N/A")
        v2 = getattr(m2, attr, "N/A")
        same = v1 == v2
        marker = "  SAME" if same else "  DIFF"
        if isinstance(v1, float):
            print(f"  {attr:30s}: {v1:8.2f}  vs  {v2:8.2f}{marker}")
        else:
            print(f"  {attr:30s}: {str(v1):>8s}  vs  {str(v2):>8s}{marker}")

    print()


def _run_comparison_for(
    init: InitializationResult,
    profile: ManagerAgent,
    name: str,
    n_runs: int = 20,
    seed: int = 42,
) -> dict:
    """Run comparison and return summary dict."""
    rules = SimulationRules.load(_RULES_DIR)
    handler = RuleBasedHandler(rules)

    trigger = ManagerChangeTrigger(
        outgoing_manager_name=MANAGER,
        incoming_manager_name=profile.manager_name,
        transition_type="mid_season",
        applied_at=TRIGGER_WEEK,
        incoming_profile=profile,
    )

    comparison = run_comparison(
        team=init.team_baseline,
        squad=init.player_agents,
        manager=init.manager_agent,
        fixture_list=init.fixture_list,
        opponent_strengths=init.opponent_strengths,
        rules=rules,
        handler=handler,
        trigger=trigger,
        n_runs=n_runs,
        rng_seed=seed,
    )

    impacts = rank_player_impact(comparison, top_n=3)

    return {
        "name": name,
        "points_a": comparison.no_change.total_points_mean,
        "points_b": comparison.with_change.total_points_mean,
        "points_diff": comparison.delta.points_mean_diff,
        "cascade_diff": comparison.delta.cascade_count_diff,
        "impacts": [
            {
                "player": p.player_name,
                "score": p.impact_score,
                "form_diff": p.mean_form_b - p.mean_form_a,
                "trust_diff": p.mean_trust_b - p.mean_trust_a,
            }
            for p in impacts
        ],
    }


def _print_sim_result(result: dict) -> None:
    print(f"  Points (no change): {result['points_a']:.2f}")
    print(f"  Points (with change): {result['points_b']:.2f}")
    print(f"  Points diff: {result['points_diff']:+.2f}")
    print(f"  Cascade diffs:")
    for k, v in sorted(result["cascade_diff"].items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {k}: {v:+.2f}")
    print(f"  Top impacts:")
    for p in result["impacts"]:
        print(f"    {p['player']}: score={p['score']:.3f} form={p['form_diff']:+.3f} trust={p['trust_diff']:+.3f}")


def main() -> None:
    print("Loading initialization data...")
    init = _load_init()
    print(f"  Squad: {len(init.player_agents)} players")
    print()

    # Resolve profiles
    profiles = {}
    for short, full in MANAGERS.items():
        profiles[short] = resolve_incoming_profile(full, COMP_ID, SEASON_ID, cache_dir=_CACHE_DIR)

    # Profile diff
    _print_profile_diff(profiles["Mourinho"], profiles["Pochettino"], "Mourinho", "Pochettino")

    # 1-run deterministic comparison
    print("=" * 70)
    print("SINGLE RUN COMPARISON (seed=42, n_runs=1)")
    print("=" * 70)
    for short in MANAGERS:
        print(f"\n--- {short} ---")
        result = _run_comparison_for(init, profiles[short], short, n_runs=1, seed=42)
        _print_sim_result(result)

    # Multi-run comparison
    print()
    print("=" * 70)
    print("MULTI-RUN COMPARISON (seed=42, n_runs=20)")
    print("=" * 70)
    results = {}
    for short in MANAGERS:
        print(f"\n--- {short} ---")
        result = _run_comparison_for(init, profiles[short], short, n_runs=20, seed=42)
        _print_sim_result(result)
        results[short] = result

    # Final diff
    print()
    print("=" * 70)
    print("MANAGER-TO-MANAGER DELTA")
    print("=" * 70)
    m = results["Mourinho"]
    p = results["Pochettino"]
    print(f"  Points diff delta: {m['points_diff'] - p['points_diff']:+.3f}")
    print(f"  (Mourinho: {m['points_diff']:+.2f}, Pochettino: {p['points_diff']:+.2f})")

    # Check which cascade events differ
    all_events = set(m["cascade_diff"].keys()) | set(p["cascade_diff"].keys())
    print(f"  Cascade diff deltas:")
    for event in sorted(all_events):
        mv = m["cascade_diff"].get(event, 0)
        pv = p["cascade_diff"].get(event, 0)
        delta = mv - pv
        if abs(delta) > 0.01:
            print(f"    {event}: Mourinho={mv:+.2f} Pochettino={pv:+.2f} delta={delta:+.2f}")
        else:
            print(f"    {event}: SAME ({mv:+.2f})")


if __name__ == "__main__":
    main()
