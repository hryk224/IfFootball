"""Build season cache DB for all teams in a competition/season.

Pre-computes full-season retrospective data for every team and saves
to a single SQLite file. This is the season-scenario equivalent of
build_demo_cache.py, replacing per-team × per-trigger_week cache files
with one competition/season-level cache.

Usage:
    uv run python scripts/build_season_cache.py
    uv run python scripts/build_season_cache.py --competition 2 --season 27

Output:
    data/season_cache/premier_league_2015-16.db
"""

from __future__ import annotations

import argparse
from pathlib import Path

from iffootball.collectors.statsbomb import StatsBombOpenDataCollector
from iffootball.pipeline import initialize_season
from iffootball.storage.db import Database

_CACHE_DIR = Path(__file__).parents[1] / "data" / "season_cache"

# Known competition/season labels for readable filenames.
_LABELS: dict[tuple[int, int], str] = {
    (2, 27): "premier_league_2015-16",
}


def _cache_filename(competition_id: int, season_id: int) -> str:
    label = _LABELS.get((competition_id, season_id))
    if label:
        return f"{label}.db"
    return f"comp{competition_id}_season{season_id}.db"


def _progress(team: str, index: int, total: int) -> None:
    print(f"  [{index}/{total}] {team}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build season cache DB")
    parser.add_argument(
        "--competition", type=int, default=2, help="StatsBomb competition ID"
    )
    parser.add_argument(
        "--season", type=int, default=27, help="StatsBomb season ID"
    )
    parser.add_argument(
        "--league-name", type=str, default="Premier League",
        help="Human-readable league name",
    )
    args = parser.parse_args()

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    db_path = _CACHE_DIR / _cache_filename(args.competition, args.season)

    if db_path.exists():
        db_path.unlink()

    print("Building season cache...")
    print(f"  Competition: {args.competition}, Season: {args.season}")
    print(f"  Output: {db_path}")
    print()

    collector = StatsBombOpenDataCollector()

    with Database(db_path) as db:
        result = initialize_season(
            collector=collector,
            competition_id=args.competition,
            season_id=args.season,
            league_name=args.league_name,
            db=db,
            progress_fn=_progress,
        )

    print()
    print("Done.")
    print(f"  Teams: {len(result.teams)}")
    print(f"  Players: {result.player_count}")
    print(f"  Managers: {result.manager_count}")
    print(f"  Opponents: {result.opponent_count}")
    print(f"  File: {db_path} ({db_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
