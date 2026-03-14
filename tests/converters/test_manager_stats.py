"""Tests for manager_stats converter module."""

from __future__ import annotations

import pandas as pd
import pytest

from iffootball.converters.manager_stats import (
    _infer_formation,
    _parse_managers,
    build_manager_agent,
    calc_cultural_inertia,
    calc_manager_possession_preference,
    calc_manager_pressing_intensity,
    calc_preferred_formation,
    extract_manager_tenure,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_matches(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _make_events(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _make_lineup_row(position_name: str, start_reason: str = "Starting XI") -> dict[str, object]:
    """Build a minimal lineup row with the given starting position."""
    return {
        "player_id": 1,
        "player_name": "Player A",
        "positions": [{"position": position_name, "start_reason": start_reason}],
    }


def _make_starting_xi(
    position_names: list[str],
) -> pd.DataFrame:
    """Build a lineup DataFrame with starting XI from the given position names."""
    rows = [_make_lineup_row(p) for p in position_names]
    return pd.DataFrame(rows)


# Standard 4-3-3 position list (using StatsBomb names)
_433_POSITIONS = [
    "Goalkeeper",           # GK
    "Right Back",           # DF
    "Right Center Back",    # DF
    "Left Center Back",     # DF
    "Left Back",            # DF
    "Center Defensive Midfield",  # MF
    "Right Center Midfield",      # MF
    "Left Center Midfield",       # MF
    "Right Wing",           # MF (winger → MF in BroadPosition)
    "Left Wing",            # MF
    "Center Forward",       # FW
]

# Standard 4-4-2 position list
_442_POSITIONS = [
    "Goalkeeper",
    "Right Back",
    "Right Center Back",
    "Left Center Back",
    "Left Back",
    "Right Midfield",
    "Right Center Midfield",
    "Left Center Midfield",
    "Left Midfield",
    "Right Center Forward",
    "Left Center Forward",
]

# Standard 3-5-2 position list
_352_POSITIONS = [
    "Goalkeeper",
    "Right Center Back",
    "Center Back",
    "Left Center Back",
    "Right Wing Back",   # DF
    "Left Wing Back",    # DF
    "Center Defensive Midfield",
    "Right Center Midfield",
    "Left Center Midfield",
    "Right Center Forward",
    "Left Center Forward",
]


# ---------------------------------------------------------------------------
# TestParseManagers
# ---------------------------------------------------------------------------


class TestParseManagers:
    def test_single_name(self) -> None:
        assert _parse_managers("Louis van Gaal") == ["Louis van Gaal"]

    def test_comma_separated(self) -> None:
        result = _parse_managers("Tiago Cardoso Mendes, Diego Pablo Simeone")
        assert result == ["Tiago Cardoso Mendes", "Diego Pablo Simeone"]

    def test_empty_string(self) -> None:
        assert _parse_managers("") == []

    def test_whitespace_only(self) -> None:
        assert _parse_managers("   ") == []

    def test_nan(self) -> None:
        assert _parse_managers(float("nan")) == []

    def test_none(self) -> None:
        assert _parse_managers(None) == []

    def test_pd_na(self) -> None:
        assert _parse_managers(pd.NA) == []

    def test_strips_whitespace_around_names(self) -> None:
        result = _parse_managers("  Alice ,  Bob  ")
        assert result == ["Alice", "Bob"]


# ---------------------------------------------------------------------------
# TestExtractManagerTenure
# ---------------------------------------------------------------------------


class TestExtractManagerTenure:
    def _make_matches_df(self) -> pd.DataFrame:
        return _make_matches([
            {
                "match_id": 1,
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "home_managers": "Arsène Wenger",
                "away_managers": "José Mourinho",
                "match_week": 1,
            },
            {
                "match_id": 2,
                "home_team": "Chelsea",
                "away_team": "Arsenal",
                "home_managers": "José Mourinho",
                "away_managers": "Arsène Wenger",
                "match_week": 2,
            },
            {
                "match_id": 3,
                "home_team": "Arsenal",
                "away_team": "Liverpool",
                "home_managers": "Arsène Wenger",
                "away_managers": "Jürgen Klopp",
                "match_week": 3,
            },
        ])

    def test_home_matches_included(self) -> None:
        matches = self._make_matches_df()
        result = extract_manager_tenure(matches, "Arsenal", "Arsène Wenger")
        assert 1 in result

    def test_away_matches_included(self) -> None:
        matches = self._make_matches_df()
        result = extract_manager_tenure(matches, "Arsenal", "Arsène Wenger")
        assert 2 in result

    def test_other_team_excluded(self) -> None:
        matches = self._make_matches_df()
        result = extract_manager_tenure(matches, "Arsenal", "Arsène Wenger")
        # match_id=2 is Chelsea's home — Arsenal is away; still included since
        # Arsène Wenger is the away manager for Arsenal.
        # match_id=3: Arsenal home with Wenger — included.
        assert result == frozenset({1, 2, 3})

    def test_different_manager_excluded(self) -> None:
        matches = self._make_matches_df()
        result = extract_manager_tenure(matches, "Chelsea", "Arsène Wenger")
        assert result == frozenset()

    def test_empty_manager_field_skipped(self) -> None:
        matches = _make_matches([
            {
                "match_id": 10,
                "home_team": "Málaga",
                "away_team": "Valencia",
                "home_managers": "",
                "away_managers": "Gary Neville",
                "match_week": 25,
            },
        ])
        # Empty manager string → match not included for either "unknown" manager
        result = extract_manager_tenure(matches, "Málaga", "")
        assert result == frozenset()

    def test_comma_separated_includes_both_managers(self) -> None:
        matches = _make_matches([
            {
                "match_id": 20,
                "home_team": "Real Madrid",
                "away_team": "Barcelona",
                "home_managers": "Rafael Benítez Maudes, Zinédine Zidane",
                "away_managers": "Luis Enrique Martínez García",
                "match_week": 18,
            },
        ])
        result_benitez = extract_manager_tenure(matches, "Real Madrid", "Rafael Benítez Maudes")
        result_zidane = extract_manager_tenure(matches, "Real Madrid", "Zinédine Zidane")
        assert 20 in result_benitez
        assert 20 in result_zidane

    def test_nan_manager_field_skipped(self) -> None:
        matches = _make_matches([
            {
                "match_id": 30,
                "home_team": "Valencia",
                "away_team": "Getafe",
                "home_managers": float("nan"),
                "away_managers": "Francisco Escriba Segura",
                "match_week": 24,
            },
        ])
        result = extract_manager_tenure(matches, "Valencia", "Gary Neville")
        assert result == frozenset()

    def test_returns_frozenset(self) -> None:
        matches = self._make_matches_df()
        result = extract_manager_tenure(matches, "Arsenal", "Arsène Wenger")
        assert isinstance(result, frozenset)


# ---------------------------------------------------------------------------
# TestCalcManagerPressingIntensity
# ---------------------------------------------------------------------------


class TestCalcManagerPressingIntensity:
    def _make_events_df(self) -> pd.DataFrame:
        return _make_events([
            {"match_id": 1, "team": "Arsenal", "type": "Pressure"},
            {"match_id": 1, "team": "Arsenal", "type": "Pressure"},
            {"match_id": 1, "team": "Arsenal", "type": "Pass"},
            {"match_id": 1, "team": "Chelsea", "type": "Pressure"},
            {"match_id": 2, "team": "Arsenal", "type": "Pressure"},
            {"match_id": 2, "team": "Arsenal", "type": "Pass"},
        ])

    def test_pressing_per_90(self) -> None:
        events = self._make_events_df()
        # match_id 1: 2 Pressures by Arsenal; match_id 2: 1 Pressure → avg = 1.5
        result = calc_manager_pressing_intensity(events, "Arsenal", frozenset({1, 2}))
        assert result == pytest.approx(1.5)

    def test_opponent_pressure_excluded(self) -> None:
        events = self._make_events_df()
        result = calc_manager_pressing_intensity(events, "Chelsea", frozenset({1}))
        assert result == pytest.approx(1.0)

    def test_empty_tenure_returns_zero(self) -> None:
        events = self._make_events_df()
        result = calc_manager_pressing_intensity(events, "Arsenal", frozenset())
        assert result == 0.0

    def test_no_pressure_events_returns_zero(self) -> None:
        events = _make_events([
            {"match_id": 1, "team": "Arsenal", "type": "Pass"},
        ])
        result = calc_manager_pressing_intensity(events, "Arsenal", frozenset({1}))
        assert result == 0.0


# ---------------------------------------------------------------------------
# TestCalcManagerPossessionPreference
# ---------------------------------------------------------------------------


class TestCalcManagerPossessionPreference:
    def _make_events_df(self) -> pd.DataFrame:
        return _make_events([
            {"match_id": 1, "team": "Arsenal", "type": "Pass"},
            {"match_id": 1, "team": "Arsenal", "type": "Pass"},
            {"match_id": 1, "team": "Arsenal", "type": "Carry"},
            {"match_id": 1, "team": "Chelsea", "type": "Pass"},
            {"match_id": 1, "team": "Chelsea", "type": "Carry"},
        ])

    def test_possession_ratio(self) -> None:
        events = self._make_events_df()
        # Arsenal: 3 events; total: 5 → 0.6
        result = calc_manager_possession_preference(events, "Arsenal", frozenset({1}))
        assert result == pytest.approx(0.6)

    def test_empty_tenure_returns_zero(self) -> None:
        result = calc_manager_possession_preference(
            self._make_events_df(), "Arsenal", frozenset()
        )
        assert result == 0.0

    def test_no_pass_carry_events_returns_zero(self) -> None:
        events = _make_events([
            {"match_id": 1, "team": "Arsenal", "type": "Pressure"},
        ])
        result = calc_manager_possession_preference(events, "Arsenal", frozenset({1}))
        assert result == 0.0


# ---------------------------------------------------------------------------
# TestInferFormation
# ---------------------------------------------------------------------------


class TestInferFormation:
    def test_4_3_3(self) -> None:
        # _433 has: 1 GK, 4 DF, 5 MF (3 CM + 2 Wingers), 1 FW → "4-5-1"
        # Wait — with BroadPosition: Wingers → MF, so 3+2=5 MF, 1 FW
        # Let me adjust: use a real 4-3-3 with 3 FW
        positions = [
            "Goalkeeper",
            "Right Back",
            "Right Center Back",
            "Left Center Back",
            "Left Back",
            "Center Defensive Midfield",
            "Right Center Midfield",
            "Left Center Midfield",
            "Right Wing",     # MF (winger → MF)
            "Left Wing",      # MF
            "Center Forward", # FW
        ]
        lineup = _make_starting_xi(positions)
        # GK=1, DF=4, MF=5 (3 CM + 2 Wingers), FW=1 → "4-5-1"
        assert _infer_formation(lineup) == "4-5-1"

    def test_4_4_2(self) -> None:
        lineup = _make_starting_xi(_442_POSITIONS)
        # GK=1, DF=4, MF=4 (Right/Left/CM×2), FW=2 → "4-4-2"
        assert _infer_formation(lineup) == "4-4-2"

    def test_3_5_2(self) -> None:
        lineup = _make_starting_xi(_352_POSITIONS)
        # GK=1, DF=3 CB + 2 WB = 5 DF, MF=3, FW=2 → "5-3-2"
        assert _infer_formation(lineup) == "5-3-2"

    def test_fewer_than_11_starters_returns_none(self) -> None:
        positions = _433_POSITIONS[:10]  # only 10 starters
        lineup = _make_starting_xi(positions)
        assert _infer_formation(lineup) is None

    def test_unsupported_position_returns_none(self) -> None:
        positions = _433_POSITIONS[:10] + ["Unknown Position"]
        lineup = _make_starting_xi(positions)
        assert _infer_formation(lineup) is None

    def test_non_starter_excluded(self) -> None:
        # One player is a substitute; only 10 starters → returns None
        rows = [_make_lineup_row(p) for p in _433_POSITIONS[:10]]
        rows.append(_make_lineup_row("Center Forward", start_reason="Substitution - On (Tactical)"))
        lineup = pd.DataFrame(rows)
        assert _infer_formation(lineup) is None

    def test_empty_positions_list_skipped(self) -> None:
        # Player with empty positions list should be skipped
        rows = [_make_lineup_row(p) for p in _433_POSITIONS]
        rows.append({"player_id": 99, "player_name": "X", "positions": []})
        lineup = pd.DataFrame(rows)
        # Still 11 starters → should return a valid formation
        result = _infer_formation(lineup)
        assert result is not None

    def test_returns_none_for_empty_lineup(self) -> None:
        lineup = pd.DataFrame(
            columns=["player_id", "player_name", "positions"]
        )
        assert _infer_formation(lineup) is None


# ---------------------------------------------------------------------------
# TestCalcPreferredFormation
# ---------------------------------------------------------------------------


class TestCalcPreferredFormation:
    def _build_lineups(
        self, match_ids: list[int], positions_per_match: list[list[str]]
    ) -> dict[int, dict[str, pd.DataFrame]]:
        lineups: dict[int, dict[str, pd.DataFrame]] = {}
        for mid, positions in zip(match_ids, positions_per_match):
            lineups[mid] = {"Arsenal": _make_starting_xi(positions)}
        return lineups

    def test_most_frequent_formation_returned(self) -> None:
        lineups = self._build_lineups(
            [1, 2, 3],
            [_442_POSITIONS, _442_POSITIONS, _433_POSITIONS],
        )
        result = calc_preferred_formation(lineups, "Arsenal", frozenset({1, 2, 3}))
        assert result == "4-4-2"

    def test_tiebreak_is_lexicographic(self) -> None:
        # 4-4-2 and 4-5-1 each appear once → lexicographic: "4-4-2" < "4-5-1"
        lineups = self._build_lineups(
            [1, 2],
            [_442_POSITIONS, _433_POSITIONS],
        )
        result = calc_preferred_formation(lineups, "Arsenal", frozenset({1, 2}))
        assert result == "4-4-2"

    def test_empty_tenure_returns_none(self) -> None:
        lineups = self._build_lineups([1], [_442_POSITIONS])
        result = calc_preferred_formation(lineups, "Arsenal", frozenset())
        assert result is None

    def test_missing_match_in_lineups_skipped(self) -> None:
        lineups: dict[int, dict[str, pd.DataFrame]] = {}
        result = calc_preferred_formation(lineups, "Arsenal", frozenset({1, 2}))
        assert result is None

    def test_team_not_in_match_lineups_skipped(self) -> None:
        lineups = {1: {"Chelsea": _make_starting_xi(_442_POSITIONS)}}
        result = calc_preferred_formation(lineups, "Arsenal", frozenset({1}))
        assert result is None


# ---------------------------------------------------------------------------
# TestCalcCulturalInertia
# ---------------------------------------------------------------------------


class TestCalcCulturalInertia:
    def test_zero_matches(self) -> None:
        assert calc_cultural_inertia(0) == pytest.approx(0.0)

    def test_half_season(self) -> None:
        assert calc_cultural_inertia(19) == pytest.approx(19 / 38)

    def test_full_season(self) -> None:
        assert calc_cultural_inertia(38) == pytest.approx(1.0)

    def test_beyond_full_season_capped(self) -> None:
        assert calc_cultural_inertia(50) == pytest.approx(1.0)

    def test_one_match(self) -> None:
        assert calc_cultural_inertia(1) == pytest.approx(1 / 38)


# ---------------------------------------------------------------------------
# TestBuildManagerAgent
# ---------------------------------------------------------------------------


class TestBuildManagerAgent:
    def _fixtures(self) -> tuple[
        pd.DataFrame, pd.DataFrame, dict[int, dict[str, pd.DataFrame]]
    ]:
        matches = _make_matches([
            {
                "match_id": 1,
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "home_managers": "Arsène Wenger",
                "away_managers": "José Mourinho",
                "match_week": 1,
            },
            {
                "match_id": 2,
                "home_team": "Chelsea",
                "away_team": "Arsenal",
                "home_managers": "José Mourinho",
                "away_managers": "Arsène Wenger",
                "match_week": 2,
            },
        ])
        events = _make_events([
            {"match_id": 1, "team": "Arsenal", "type": "Pressure"},
            {"match_id": 1, "team": "Arsenal", "type": "Pass"},
            {"match_id": 1, "team": "Arsenal", "type": "Carry"},
            {"match_id": 1, "team": "Chelsea", "type": "Pass"},
            {"match_id": 2, "team": "Arsenal", "type": "Pressure"},
            {"match_id": 2, "team": "Arsenal", "type": "Pass"},
            {"match_id": 2, "team": "Chelsea", "type": "Pass"},
        ])
        lineups_by_match = {
            1: {"Arsenal": _make_starting_xi(_442_POSITIONS)},
            2: {"Arsenal": _make_starting_xi(_442_POSITIONS)},
        }
        return matches, events, lineups_by_match

    def test_returns_manager_agent(self) -> None:
        from iffootball.agents.manager import ManagerAgent

        matches, events, lineups = self._fixtures()
        result = build_manager_agent(
            events, matches, lineups, "Arsenal", "Arsène Wenger", 2, 27
        )
        assert isinstance(result, ManagerAgent)

    def test_team_and_manager_name(self) -> None:
        matches, events, lineups = self._fixtures()
        result = build_manager_agent(
            events, matches, lineups, "Arsenal", "Arsène Wenger", 2, 27
        )
        assert result.team_name == "Arsenal"
        assert result.manager_name == "Arsène Wenger"

    def test_tenure_match_ids_populated(self) -> None:
        matches, events, lineups = self._fixtures()
        result = build_manager_agent(
            events, matches, lineups, "Arsenal", "Arsène Wenger", 2, 27
        )
        assert result.tenure_match_ids == frozenset({1, 2})

    def test_counter_tendency_is_complement(self) -> None:
        matches, events, lineups = self._fixtures()
        result = build_manager_agent(
            events, matches, lineups, "Arsenal", "Arsène Wenger", 2, 27
        )
        assert result.counter_tendency == pytest.approx(1.0 - result.possession_preference)

    def test_provisional_defaults(self) -> None:
        matches, events, lineups = self._fixtures()
        result = build_manager_agent(
            events, matches, lineups, "Arsenal", "Arsène Wenger", 2, 27
        )
        assert result.implementation_speed == pytest.approx(50.0)
        assert result.youth_development == pytest.approx(50.0)
        assert result.style_stubbornness == pytest.approx(50.0)

    def test_job_security_default(self) -> None:
        matches, events, lineups = self._fixtures()
        result = build_manager_agent(
            events, matches, lineups, "Arsenal", "Arsène Wenger", 2, 27
        )
        assert result.job_security == pytest.approx(1.0)

    def test_squad_trust_empty_by_default(self) -> None:
        matches, events, lineups = self._fixtures()
        result = build_manager_agent(
            events, matches, lineups, "Arsenal", "Arsène Wenger", 2, 27
        )
        assert result.squad_trust == {}

    def test_unknown_manager_empty_tenure(self) -> None:
        matches, events, lineups = self._fixtures()
        result = build_manager_agent(
            events, matches, lineups, "Arsenal", "Unknown Manager", 2, 27
        )
        assert result.tenure_match_ids == frozenset()
        assert result.pressing_intensity == 0.0
        assert result.possession_preference == 0.0
        assert result.preferred_formation is None
