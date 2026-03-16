"""Tests for incoming manager profile resolution.

Tests the real resolve_incoming_profile() function from the
incoming_profile module. No logic duplication — these call the
actual production code.
"""

from __future__ import annotations

from pathlib import Path

from iffootball.incoming_profile import (
    neutral_manager_profile,
    resolve_incoming_profile,
)

_CACHE_DIR = Path(__file__).parents[1] / "data" / "demo_cache"
_HAS_CACHE = _CACHE_DIR.exists() and any(_CACHE_DIR.glob("*.db"))


class TestNeutralProfile:
    def test_returns_neutral_values(self) -> None:
        profile = neutral_manager_profile("Unknown Manager")
        assert profile.manager_name == "Unknown Manager"
        assert profile.pressing_intensity == 50.0
        assert profile.possession_preference == 0.5
        assert profile.counter_tendency == 0.5
        assert profile.preferred_formation == "4-4-2"
        assert profile.team_name == ""


class TestResolveIncomingProfile:
    def test_mourinho_from_cache(self) -> None:
        """Mourinho should resolve from Chelsea demo cache."""
        if not _HAS_CACHE:
            return

        profile = resolve_incoming_profile(
            "José Mario Felix dos Santos Mourinho", 2, 27,
            cache_dir=_CACHE_DIR,
        )
        assert profile.pressing_intensity != 50.0
        assert profile.team_name == "Chelsea"

    def test_klopp_from_runtime(self) -> None:
        """Klopp is not in cache — should build from StatsBomb data."""
        profile = resolve_incoming_profile(
            "Jürgen Klopp", 2, 27,
            cache_dir=_CACHE_DIR,
        )
        # Klopp exists in StatsBomb PL 2015-16 for Liverpool.
        assert profile.pressing_intensity != 50.0
        assert profile.team_name == "Liverpool"

    def test_unknown_manager_returns_neutral(self) -> None:
        """Unknown manager falls back to neutral defaults."""
        profile = resolve_incoming_profile(
            "Nonexistent Manager XYZ", 2, 27,
            cache_dir=_CACHE_DIR,
        )
        assert profile.pressing_intensity == 50.0
        assert profile.preferred_formation == "4-4-2"
        assert profile.team_name == ""

    def test_mourinho_and_klopp_differ(self) -> None:
        """Different incoming managers produce different profiles."""
        if not _HAS_CACHE:
            return

        mourinho = resolve_incoming_profile(
            "José Mario Felix dos Santos Mourinho", 2, 27,
            cache_dir=_CACHE_DIR,
        )
        klopp = resolve_incoming_profile(
            "Jürgen Klopp", 2, 27,
            cache_dir=_CACHE_DIR,
        )
        assert mourinho.pressing_intensity != klopp.pressing_intensity

    def test_cache_disabled_falls_through_to_runtime(self) -> None:
        """With cache_dir=None, cache is skipped, runtime is used."""
        profile = resolve_incoming_profile(
            "José Mario Felix dos Santos Mourinho", 2, 27,
            cache_dir=None,  # No cache — must use runtime.
        )
        # Mourinho still exists in StatsBomb data.
        assert profile.pressing_intensity != 50.0

    def test_full_resolution_order(self) -> None:
        """The 3-level resolution: cache -> runtime -> neutral."""
        if not _HAS_CACHE:
            return

        # Mourinho: cache hit (Chelsea).
        m = resolve_incoming_profile(
            "José Mario Felix dos Santos Mourinho", 2, 27,
            cache_dir=_CACHE_DIR,
        )
        assert m.team_name == "Chelsea"

        # Klopp: cache miss -> runtime (Liverpool).
        k = resolve_incoming_profile(
            "Jürgen Klopp", 2, 27,
            cache_dir=_CACHE_DIR,
        )
        assert k.team_name == "Liverpool"

        # Unknown: cache miss -> runtime miss -> neutral.
        u = resolve_incoming_profile(
            "Unknown Coach", 2, 27,
            cache_dir=_CACHE_DIR,
        )
        assert u.pressing_intensity == 50.0
        assert u.team_name == ""
