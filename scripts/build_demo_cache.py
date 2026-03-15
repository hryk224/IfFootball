"""Build demo data cache for Live Demo preset scenarios.

Pre-fetches StatsBomb data for each preset scenario (team + manager +
trigger_week) and saves initialization snapshots to per-scenario SQLite
files. This eliminates the 2-5 minute API wait time on first demo run.

Usage:
    uv run python scripts/build_demo_cache.py

Cache files are saved to data/demo_cache/{team}_w{week}.db and
committed to the repository for Streamlit Cloud deployment.
"""

from __future__ import annotations

from pathlib import Path

from iffootball.collectors.statsbomb import StatsBombOpenDataCollector
from iffootball.pipeline import initialize
from iffootball.storage.db import Database

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path(__file__).parents[1] / "config"
_TARGETS_PATH = _CONFIG_DIR / "targets.toml"
_CACHE_DIR = Path(__file__).parents[1] / "data" / "demo_cache"

# Live Demo: PL 2015-16 only.
_DEMO_COMPETITION_ID = 2
_DEMO_SEASON_ID = 27

# Preset scenarios to cache (must match app.py _PRESETS).
_DEMO_PRESETS: list[dict[str, object]] = [
    {
        "team_name": "Manchester United",
        "manager_name": "Louis van Gaal",
        "trigger_week": 29,
    },
    {
        "team_name": "Chelsea",
        "manager_name": "José Mario Felix dos Santos Mourinho",
        "trigger_week": 16,
    },
]


def _cache_key(team_name: str, trigger_week: int) -> str:
    """Return a cache filename for a team + trigger_week combination."""
    safe_name = team_name.replace(" ", "_").lower()
    return f"{safe_name}_w{trigger_week}"


def main() -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Building demo cache for {len(_DEMO_PRESETS)} presets...")
    print(f"  Competition: {_DEMO_COMPETITION_ID}, Season: {_DEMO_SEASON_ID}")
    print(f"  Cache dir: {_CACHE_DIR}")
    print()

    collector = StatsBombOpenDataCollector()

    for i, preset in enumerate(_DEMO_PRESETS, 1):
        team = str(preset["team_name"])
        manager = str(preset["manager_name"])
        week_val = preset["trigger_week"]
        week = int(str(week_val))

        cache_name = _cache_key(team, week)
        db_path = _CACHE_DIR / f"{cache_name}.db"

        if db_path.exists():
            db_path.unlink()

        print(f"[{i}/{len(_DEMO_PRESETS)}] {team} / {manager} / week {week}...")
        try:
            with Database(db_path) as db:
                initialize(
                    collector=collector,
                    competition_id=_DEMO_COMPETITION_ID,
                    season_id=_DEMO_SEASON_ID,
                    team_name=team,
                    manager_name=manager,
                    trigger_week=week,
                    league_name="Premier League",
                    db=db,
                )
            print(f"  Cached to {db_path.name}")
        except Exception as e:
            print(f"  Failed: {e}")

    print(f"\nDone. Cache saved to {_CACHE_DIR}/")


if __name__ == "__main__":
    main()
