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
        """Mourinho resolves from Chelsea demo cache."""
        if not _HAS_CACHE:
            return
        profile = resolve_incoming_profile(
            "José Mario Felix dos Santos Mourinho", 2, 27,
            cache_dir=_CACHE_DIR,
        )
        assert profile.pressing_intensity != 50.0
        assert profile.team_name == "Chelsea"

    def test_pochettino_from_cache(self) -> None:
        """Pochettino resolves from Spurs demo cache."""
        if not _HAS_CACHE:
            return
        profile = resolve_incoming_profile(
            "Mauricio Roberto Pochettino Trossero", 2, 27,
            cache_dir=_CACHE_DIR,
        )
        assert profile.pressing_intensity != 50.0
        assert profile.team_name == "Tottenham Hotspur"

    def test_hiddink_from_cache(self) -> None:
        """Hiddink resolves from Chelsea (w25) demo cache."""
        if not _HAS_CACHE:
            return
        profile = resolve_incoming_profile(
            "Guus Hiddink", 2, 27,
            cache_dir=_CACHE_DIR,
        )
        assert profile.pressing_intensity != 50.0
        assert profile.team_name == "Chelsea"

    def test_all_preset_managers_differ(self) -> None:
        """All 3 preset incoming managers have different pressing values."""
        if not _HAS_CACHE:
            return
        mourinho = resolve_incoming_profile(
            "José Mario Felix dos Santos Mourinho", 2, 27,
            cache_dir=_CACHE_DIR,
        )
        pochettino = resolve_incoming_profile(
            "Mauricio Roberto Pochettino Trossero", 2, 27,
            cache_dir=_CACHE_DIR,
        )
        hiddink = resolve_incoming_profile(
            "Guus Hiddink", 2, 27,
            cache_dir=_CACHE_DIR,
        )
        values = {
            mourinho.pressing_intensity,
            pochettino.pressing_intensity,
            hiddink.pressing_intensity,
        }
        assert len(values) == 3

    def test_unknown_manager_returns_neutral(self) -> None:
        """Unknown manager falls back to neutral defaults."""
        profile = resolve_incoming_profile(
            "Nonexistent Manager XYZ", 2, 27,
            cache_dir=_CACHE_DIR,
        )
        assert profile.pressing_intensity == 50.0
        assert profile.team_name == ""

    def test_cache_disabled_falls_to_runtime(self) -> None:
        """With cache_dir=None, resolves via StatsBomb runtime."""
        profile = resolve_incoming_profile(
            "José Mario Felix dos Santos Mourinho", 2, 27,
            cache_dir=None,
        )
        assert profile.pressing_intensity != 50.0

    def test_all_preset_managers_resolve_from_cache(self) -> None:
        """All 3 preset incoming managers resolve from cache, not runtime."""
        if not _HAS_CACHE:
            return

        mourinho = resolve_incoming_profile(
            "José Mario Felix dos Santos Mourinho", 2, 27,
            cache_dir=_CACHE_DIR,
        )
        assert mourinho.team_name == "Chelsea"

        pochettino = resolve_incoming_profile(
            "Mauricio Roberto Pochettino Trossero", 2, 27,
            cache_dir=_CACHE_DIR,
        )
        assert pochettino.team_name == "Tottenham Hotspur"

        hiddink = resolve_incoming_profile(
            "Guus Hiddink", 2, 27,
            cache_dir=_CACHE_DIR,
        )
        assert hiddink.team_name == "Chelsea"

        # None should be neutral.
        for p in [mourinho, pochettino, hiddink]:
            assert p.pressing_intensity != 50.0
