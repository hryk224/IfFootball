"""Tests for stats_to_attributes conversion layer."""

from __future__ import annotations

import pytest
import pandas as pd

from iffootball.agents.player import BroadPosition, RoleFamily
from iffootball.converters.stats_to_attributes import (
    SUPPORTED_POSITION_NAMES,
    build_player_agents,
    calc_minutes_played,
    normalize_percentile,
    to_broad_position,
    to_role_family,
)


# ---------------------------------------------------------------------------
# to_role_family
# ---------------------------------------------------------------------------


class TestToRoleFamily:
    def test_all_supported_positions_map_without_error(self) -> None:
        """Every entry in SUPPORTED_POSITION_NAMES must resolve without raising."""
        for name in SUPPORTED_POSITION_NAMES:
            to_role_family(name)  # must not raise

    def test_goalkeeper_maps_to_goalkeeper(self) -> None:
        assert to_role_family("Goalkeeper") == RoleFamily.GOALKEEPER

    def test_center_back_maps_to_center_back(self) -> None:
        assert to_role_family("Center Back") == RoleFamily.CENTER_BACK

    def test_wing_back_is_distinct_from_full_back(self) -> None:
        assert to_role_family("Right Wing Back") == RoleFamily.WING_BACK
        assert to_role_family("Right Back") == RoleFamily.FULL_BACK
        assert to_role_family("Right Wing Back") != to_role_family("Right Back")

    def test_right_midfield_maps_to_central_midfielder(self) -> None:
        # M1: side midfielders treated as central MF (may be revisited later)
        assert to_role_family("Right Midfield") == RoleFamily.CENTRAL_MIDFIELDER
        assert to_role_family("Left Midfield") == RoleFamily.CENTRAL_MIDFIELDER

    def test_winger_maps_to_winger(self) -> None:
        assert to_role_family("Right Wing") == RoleFamily.WINGER
        assert to_role_family("Left Wing") == RoleFamily.WINGER

    def test_center_forward_maps_to_forward(self) -> None:
        assert to_role_family("Center Forward") == RoleFamily.FORWARD

    def test_unknown_position_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported position_name"):
            to_role_family("Unknown Position")

    def test_case_sensitive_match(self) -> None:
        with pytest.raises(ValueError):
            to_role_family("goalkeeper")


# ---------------------------------------------------------------------------
# to_broad_position
# ---------------------------------------------------------------------------


class TestToBroadPosition:
    def test_goalkeeper_broad_is_gk(self) -> None:
        assert to_broad_position(RoleFamily.GOALKEEPER) == BroadPosition.GK

    def test_center_back_broad_is_df(self) -> None:
        assert to_broad_position(RoleFamily.CENTER_BACK) == BroadPosition.DF

    def test_wing_back_broad_is_df(self) -> None:
        assert to_broad_position(RoleFamily.WING_BACK) == BroadPosition.DF

    def test_winger_broad_is_mf(self) -> None:
        # M1: wingers are classified as MF in BroadPosition
        assert to_broad_position(RoleFamily.WINGER) == BroadPosition.MF

    def test_forward_broad_is_fw(self) -> None:
        assert to_broad_position(RoleFamily.FORWARD) == BroadPosition.FW

    def test_all_role_families_have_broad_mapping(self) -> None:
        """Every RoleFamily must map to a BroadPosition without raising."""
        for role in RoleFamily:
            to_broad_position(role)  # must not raise


# ---------------------------------------------------------------------------
# calc_minutes_played
# ---------------------------------------------------------------------------


def _make_events(rows: list[dict]) -> pd.DataFrame:  # type: ignore[type-arg]
    return pd.DataFrame(rows)


class TestCalcMinutesPlayed:
    def test_single_player_single_match(self) -> None:
        events = _make_events(
            [
                {"player_id": 1, "match_id": 10, "minute": 45},
                {"player_id": 1, "match_id": 10, "minute": 88},
            ]
        )
        result = calc_minutes_played(events)
        assert result[1] == 88

    def test_accumulates_across_matches(self) -> None:
        events = _make_events(
            [
                {"player_id": 1, "match_id": 10, "minute": 90},
                {"player_id": 1, "match_id": 11, "minute": 85},
            ]
        )
        result = calc_minutes_played(events)
        assert result[1] == 90 + 85

    def test_multiple_players(self) -> None:
        events = _make_events(
            [
                {"player_id": 1, "match_id": 10, "minute": 90},
                {"player_id": 2, "match_id": 10, "minute": 60},
            ]
        )
        result = calc_minutes_played(events)
        assert result[1] == 90
        assert result[2] == 60

    def test_missing_player_id_is_excluded(self) -> None:
        events = _make_events(
            [
                {"player_id": None, "match_id": 10, "minute": 90},
                {"player_id": 1, "match_id": 10, "minute": 45},
            ]
        )
        result = calc_minutes_played(events)
        assert 1 in result.index
        assert len(result) == 1

    def test_stoppage_time_is_capped_at_90(self) -> None:
        # Stoppage time (minute > 90) must not inflate per-match minutes.
        events = _make_events(
            [
                {"player_id": 1, "match_id": 10, "minute": 95},  # stoppage time
            ]
        )
        result = calc_minutes_played(events)
        assert result[1] == 90

    def test_stoppage_time_cap_does_not_affect_sub_90_values(self) -> None:
        events = _make_events(
            [
                {"player_id": 1, "match_id": 10, "minute": 75},
            ]
        )
        result = calc_minutes_played(events)
        assert result[1] == 75


# ---------------------------------------------------------------------------
# normalize_percentile
# ---------------------------------------------------------------------------


class TestNormalizePercentile:
    def _make_stats(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "minutes": [1000.0, 950.0, 800.0, 500.0],
                "xg_per90": [10.0, 20.0, 30.0, 40.0],
            },
            index=[1, 2, 3, 4],
        )

    def test_players_below_threshold_excluded(self) -> None:
        result = normalize_percentile(self._make_stats(), min_minutes=900.0)
        assert 3 not in result.index
        assert 4 not in result.index

    def test_percentile_values_are_0_to_100(self) -> None:
        result = normalize_percentile(self._make_stats(), min_minutes=900.0)
        assert (result["xg_per90"] >= 0.0).all()
        assert (result["xg_per90"] <= 100.0).all()

    def test_higher_raw_value_gets_higher_percentile(self) -> None:
        result = normalize_percentile(self._make_stats(), min_minutes=900.0)
        # player 2 (xg=20) should rank higher than player 1 (xg=10)
        v2: float = result.loc[2, "xg_per90"]  # type: ignore[assignment]
        v1: float = result.loc[1, "xg_per90"]  # type: ignore[assignment]
        assert v2 > v1

    def test_empty_stats_returns_empty(self) -> None:
        empty = pd.DataFrame(columns=["minutes", "xg_per90"])
        result = normalize_percentile(empty, min_minutes=900.0)
        assert result.empty


# ---------------------------------------------------------------------------
# build_player_agents (integration)
# ---------------------------------------------------------------------------


def _make_lineups() -> dict[str, pd.DataFrame]:
    return {
        "Team A": pd.DataFrame(
            {
                "player_id": [1, 2],
                "player_name": ["Alice", "Bob"],
            }
        )
    }


def _make_full_events() -> pd.DataFrame:
    """Events giving players 1 and 2 enough minutes to qualify (>= 900).

    Player 1 plays as Goalkeeper, player 2 as Center Forward.
    """
    rows = []
    # 10 matches x 90 min each = 900 min for both players
    positions = {1: "Goalkeeper", 2: "Center Forward"}
    for match_id in range(10):
        for player_id, position in positions.items():
            rows.append(
                {
                    "player_id": player_id,
                    "match_id": match_id,
                    "minute": 90,
                    "type": "Pass",
                    "shot_statsbomb_xg": None,
                    "position": position,
                }
            )
    return pd.DataFrame(rows)


class TestBuildPlayerAgents:
    def test_returns_list_of_player_agents(self) -> None:
        from iffootball.agents.player import PlayerAgent

        agents = build_player_agents(_make_full_events(), _make_lineups(), "Team A")
        assert all(isinstance(a, PlayerAgent) for a in agents)

    def test_qualified_players_are_included(self) -> None:
        agents = build_player_agents(_make_full_events(), _make_lineups(), "Team A")
        ids = {a.player_id for a in agents}
        assert 1 in ids
        assert 2 in ids

    def test_position_fields_are_populated(self) -> None:
        agents = build_player_agents(_make_full_events(), _make_lineups(), "Team A")
        gk = next(a for a in agents if a.player_id == 1)
        assert gk.position_name == "Goalkeeper"
        assert gk.role_family == RoleFamily.GOALKEEPER
        assert gk.broad_position == BroadPosition.GK

    def test_technical_attributes_in_range(self) -> None:
        agents = build_player_agents(_make_full_events(), _make_lineups(), "Team A")
        for agent in agents:
            assert 0.0 <= agent.passing <= 100.0
            assert 0.0 <= agent.pressing <= 100.0

    def test_unknown_position_in_events_raises(self) -> None:
        # Events containing an unsupported position string must raise ValueError.
        events_bad = _make_full_events().copy()
        events_bad.loc[events_bad["player_id"] == 1, "position"] = "Unknown Role"
        with pytest.raises(ValueError, match="Unsupported position_name"):
            build_player_agents(events_bad, _make_lineups(), "Team A")

    def test_representative_position_uses_event_frequency(self) -> None:
        # Player 1 appears as Right Center Back (7 events) and Right Back (3 events).
        # Right Center Back must win as the most frequent.
        rows = []
        for match_id in range(10):
            position = "Right Center Back" if match_id < 7 else "Right Back"
            rows.append(
                {
                    "player_id": 1,
                    "match_id": match_id,
                    "minute": 90,
                    "type": "Pass",
                    "shot_statsbomb_xg": None,
                    "position": position,
                }
            )
        events_multi = pd.DataFrame(rows)
        lineups_single = {
            "Team A": pd.DataFrame({"player_id": [1], "player_name": ["Alice"]})
        }
        agents = build_player_agents(events_multi, lineups_single, "Team A")
        alice = next(a for a in agents if a.player_id == 1)
        assert alice.position_name == "Right Center Back"
        assert alice.role_family == RoleFamily.CENTER_BACK

    def test_player_with_no_position_in_events_raises(self) -> None:
        # Player whose events all have position=None must raise ValueError,
        # not be silently skipped.
        events_no_pos = _make_full_events().copy()
        events_no_pos.loc[events_no_pos["player_id"] == 1, "position"] = None
        with pytest.raises(ValueError, match="has no position data in events"):
            build_player_agents(events_no_pos, _make_lineups(), "Team A")

    def test_player_in_events_but_not_in_lineup_is_skipped(self) -> None:
        # Player 2 qualifies by minutes but has no lineup entry; must not appear in output.
        lineups_partial = {
            "Team A": pd.DataFrame({"player_id": [1], "player_name": ["Alice"]})
        }
        agents = build_player_agents(_make_full_events(), lineups_partial, "Team A")
        ids = {a.player_id for a in agents}
        assert 2 not in ids
        assert 1 in ids

    def test_player_in_lineup_but_not_in_events_is_not_agent_ized(self) -> None:
        # Player 3 is in the lineup but never appears in events;
        # they won't accumulate 900 minutes and must not be included.
        lineups_extra = {
            "Team A": pd.DataFrame(
                {
                    "player_id": [1, 2, 3],
                    "player_name": ["Alice", "Bob", "Carol"],
                }
            )
        }
        agents = build_player_agents(_make_full_events(), lineups_extra, "Team A")
        ids = {a.player_id for a in agents}
        assert 3 not in ids
