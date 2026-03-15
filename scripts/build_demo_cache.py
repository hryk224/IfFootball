"""Build demo data cache for Live Demo.

Pre-fetches StatsBomb data for all demo target teams and saves
initialization snapshots to a SQLite cache file. This eliminates
the 2-5 minute API wait time on first demo run.

Usage:
    uv run python scripts/build_demo_cache.py

The cache file is saved to data/demo_cache.db (gitignored).
"""

from __future__ import annotations

import tomllib
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

# Live Demo: PL 2015-16 only, trigger_week=1 (season start).
_DEMO_COMPETITION_ID = 2
_DEMO_SEASON_ID = 27
_DEMO_TRIGGER_WEEK = 1


def _load_demo_teams() -> list[str]:
    """Load demo target teams from targets.toml."""
    with _TARGETS_PATH.open("rb") as f:
        data = tomllib.load(f)
    for comp in data["competitions"]:
        if (
            comp["competition_id"] == _DEMO_COMPETITION_ID
            and comp["season_id"] == _DEMO_SEASON_ID
        ):
            return comp["clubs"]  # type: ignore[no-any-return]
    return []


def _team_db_path(team_name: str) -> Path:
    """Return the cache DB path for a specific team."""
    safe_name = team_name.replace(" ", "_").lower()
    return _CACHE_DIR / f"{safe_name}.db"


def main() -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    teams = _load_demo_teams()
    if not teams:
        print("No demo teams found in targets.toml")
        return

    print(f"Building demo cache for {len(teams)} teams...")
    print(f"  Competition: {_DEMO_COMPETITION_ID}, Season: {_DEMO_SEASON_ID}")
    print(f"  Trigger week: {_DEMO_TRIGGER_WEEK}")
    print(f"  Cache dir: {_CACHE_DIR}")
    print()

    collector = StatsBombOpenDataCollector()

    for i, team in enumerate(teams, 1):
        db_path = _team_db_path(team)
        # Remove old cache for this team.
        if db_path.exists():
            db_path.unlink()

        print(f"[{i}/{len(teams)}] Initializing {team}...")
        try:
            # Find managers for this team from StatsBomb data.
            matches = collector.get_matches(
                _DEMO_COMPETITION_ID, _DEMO_SEASON_ID
            )
            team_matches = matches[
                (matches["home_team"] == team)
                | (matches["away_team"] == team)
            ]

            managers: set[str] = set()
            for _, row in team_matches.iterrows():
                if row["home_team"] == team:
                    for m in str(row.get("home_managers", "")).split(", "):
                        if m.strip():
                            managers.add(m.strip())
                else:
                    for m in str(row.get("away_managers", "")).split(", "):
                        if m.strip():
                            managers.add(m.strip())

            if not managers:
                print(f"  No managers found for {team}, skipping")
                continue

            # Use the first manager (alphabetically) for cache.
            manager_name = sorted(managers)[0]
            print(f"  Manager: {manager_name}")

            with Database(db_path) as db:
                initialize(
                    collector=collector,
                    competition_id=_DEMO_COMPETITION_ID,
                    season_id=_DEMO_SEASON_ID,
                    team_name=team,
                    manager_name=manager_name,
                    trigger_week=_DEMO_TRIGGER_WEEK,
                    league_name="Premier League",
                    db=db,
                )
            print(f"  Cached to {db_path.name}")
        except Exception as e:
            print(f"  Failed: {e}")

    print(f"\nDone. Cache saved to {_CACHE_DIR}/")


if __name__ == "__main__":
    main()
