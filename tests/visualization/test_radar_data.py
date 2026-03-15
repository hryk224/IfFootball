"""Tests for radar data extraction and normalization."""

from __future__ import annotations

from iffootball.agents.fixture import Fixture, FixtureList, OpponentStrength
from iffootball.agents.league import LeagueContext
from iffootball.agents.manager import ManagerAgent
from iffootball.agents.player import BroadPosition, PlayerAgent, RoleFamily
from iffootball.agents.team import TeamBaseline
from iffootball.agents.trigger import ManagerChangeTrigger
from iffootball.config import (
    ActionDistributionConfig,
    AdaptationConfig,
    ManagerTurningPointConfig,
    MatchConfig,
    PlayerTurningPointConfig,
    SimulationRules,
    TurningPointConfig,
)
from iffootball.simulation.comparison import ComparisonResult, run_comparison
from iffootball.simulation.turning_point import RuleBasedHandler
from iffootball.visualization.radar_data import (
    RadarAxes,
    RadarChartData,
    _normalize,
    build_normalization_ranges,
    extract_radar_data,
)

# Return type for _make_comparison helper.
_ComparisonBundle = tuple[
    ComparisonResult, TeamBaseline, ManagerAgent, ManagerAgent, LeagueContext
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_433_POSITIONS = [
    ("Goalkeeper", RoleFamily.GOALKEEPER, BroadPosition.GK),
    ("Right Back", RoleFamily.FULL_BACK, BroadPosition.DF),
    ("Right Center Back", RoleFamily.CENTER_BACK, BroadPosition.DF),
    ("Left Center Back", RoleFamily.CENTER_BACK, BroadPosition.DF),
    ("Left Back", RoleFamily.FULL_BACK, BroadPosition.DF),
    ("Center Defensive Midfield", RoleFamily.DEFENSIVE_MIDFIELDER, BroadPosition.MF),
    ("Right Center Midfield", RoleFamily.CENTRAL_MIDFIELDER, BroadPosition.MF),
    ("Left Center Midfield", RoleFamily.CENTRAL_MIDFIELDER, BroadPosition.MF),
    ("Right Wing", RoleFamily.WINGER, BroadPosition.MF),
    ("Left Wing", RoleFamily.WINGER, BroadPosition.MF),
    ("Center Forward", RoleFamily.FORWARD, BroadPosition.FW),
]


def _make_squad() -> list[PlayerAgent]:
    squad: list[PlayerAgent] = []
    for i, (pos_name, role, broad) in enumerate(_433_POSITIONS):
        squad.append(
            PlayerAgent(
                player_id=i + 1,
                player_name=f"Player {i + 1}",
                position_name=pos_name,
                role_family=role,
                broad_position=broad,
                pace=50.0,
                passing=50.0,
                shooting=50.0,
                pressing=50.0,
                defending=50.0,
                physicality=50.0,
                consistency=50.0,
            )
        )
    # 3 subs
    squad.append(PlayerAgent(12, "Player 12", "Center Back", RoleFamily.CENTER_BACK, BroadPosition.DF, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0))
    squad.append(PlayerAgent(13, "Player 13", "Center Defensive Midfield", RoleFamily.DEFENSIVE_MIDFIELDER, BroadPosition.MF, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0))
    squad.append(PlayerAgent(14, "Player 14", "Center Forward", RoleFamily.FORWARD, BroadPosition.FW, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0))
    return squad


def _make_baseline() -> TeamBaseline:
    return TeamBaseline(
        team_name="Team A",
        competition_id=1,
        season_id=1,
        played_match_ids=frozenset({1, 2}),
        xg_for_per90=1.5,
        xg_against_per90=1.0,
        ppda=10.0,
        progressive_passes_per90=50.0,
        possession_pct=0.55,
        league_position=5,
        points_to_safety=10,
        points_to_title=-5,
        matches_remaining=3,
    )


def _make_manager() -> ManagerAgent:
    return ManagerAgent(
        manager_name="Original Manager",
        team_name="Team A",
        competition_id=1,
        season_id=1,
        tenure_match_ids=frozenset(),
        pressing_intensity=55.0,
        possession_preference=0.55,
        counter_tendency=0.45,
        preferred_formation="4-3-3",
    )


def _make_incoming_manager() -> ManagerAgent:
    return ManagerAgent(
        manager_name="New Manager",
        team_name="Other Team",
        competition_id=2,
        season_id=2,
        tenure_match_ids=frozenset(),
        pressing_intensity=70.0,
        possession_preference=0.65,
        counter_tendency=0.35,
        preferred_formation="3-5-2",
    )


def _make_league() -> LeagueContext:
    return LeagueContext(
        competition_id=1,
        season_id=1,
        name="Test League",
        avg_ppda=11.0,
        avg_progressive_passes_per90=45.0,
        avg_xg_per90=1.3,
    )


def _make_rules() -> SimulationRules:
    return SimulationRules(
        adaptation=AdaptationConfig(
            base_fatigue_increase=0.05,
            base_fatigue_recovery=0.03,
            tactical_understanding_gain=0.04,
            fatigue_penalty_weight=0.5,
            trust_increase_on_start=0.02,
            trust_decrease_on_bench=0.01,
        ),
        turning_points=TurningPointConfig(
            player=PlayerTurningPointConfig(
                bench_streak_threshold=3,
                tactical_understanding_low=0.40,
                short_term_window=4,
                trust_low=0.40,
            ),
            manager=ManagerTurningPointConfig(
                job_security_warning=0.30,
                job_security_critical=0.10,
                style_stubbornness_threshold=80,
            ),
            action_distribution=ActionDistributionConfig(
                bench_streak_low_trust={"resist": 0.6, "adapt": 0.3, "transfer": 0.1},
                low_understanding={"adapt": 0.5, "resist": 0.4, "transfer": 0.1},
                default={"adapt": 0.8, "resist": 0.2, "transfer": 0.0},
            ),
        ),
        match=MatchConfig(home_advantage_factor=1.1),
    )


def _make_fixtures() -> FixtureList:
    return FixtureList(
        team_name="Team A",
        trigger_week=10,
        fixtures=(
            Fixture(match_week=11, opponent_name="Opp A", is_home=True),
            Fixture(match_week=12, opponent_name="Opp B", is_home=False),
            Fixture(match_week=13, opponent_name="Opp C", is_home=True),
        ),
    )


def _make_opponents() -> dict[str, OpponentStrength]:
    return {
        "Opp A": OpponentStrength("Opp A", 1.2, 1.1, 1500.0),
        "Opp B": OpponentStrength("Opp B", 1.0, 1.3, 1450.0),
        "Opp C": OpponentStrength("Opp C", 1.4, 0.9, 1520.0),
    }


def _make_comparison() -> _ComparisonBundle:
    """Run a comparison and return (ComparisonResult, baseline, managers, league)."""
    rules = _make_rules()
    incoming = _make_incoming_manager()
    trigger = ManagerChangeTrigger(
        outgoing_manager_name="Original Manager",
        incoming_manager_name="New Manager",
        transition_type="mid_season",
        applied_at=10,
        incoming_profile=incoming,
    )
    comparison = run_comparison(
        team=_make_baseline(),
        squad=_make_squad(),
        manager=_make_manager(),
        fixture_list=_make_fixtures(),
        opponent_strengths=_make_opponents(),
        rules=rules,
        handler=RuleBasedHandler(rules),
        trigger=trigger,
        n_runs=5,
        rng_seed=42,
    )
    return comparison, _make_baseline(), _make_manager(), incoming, _make_league()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_midpoint(self) -> None:
        assert _normalize(5.0, 0.0, 10.0) == 0.5

    def test_min_value(self) -> None:
        assert _normalize(0.0, 0.0, 10.0) == 0.0

    def test_max_value(self) -> None:
        assert _normalize(10.0, 0.0, 10.0) == 1.0

    def test_clips_below(self) -> None:
        assert _normalize(-5.0, 0.0, 10.0) == 0.0

    def test_clips_above(self) -> None:
        assert _normalize(15.0, 0.0, 10.0) == 1.0

    def test_invert(self) -> None:
        assert _normalize(0.0, 0.0, 10.0, invert=True) == 1.0
        assert _normalize(10.0, 0.0, 10.0, invert=True) == 0.0

    def test_zero_span_returns_half(self) -> None:
        assert _normalize(5.0, 5.0, 5.0) == 0.5


class TestBuildNormalizationRanges:
    def test_uses_league_data(self) -> None:
        league = _make_league()
        ranges = build_normalization_ranges(league)
        lo, hi = ranges["xg_for"]
        # Center at 1.3 ± 1.0
        assert lo < 1.3 < hi

    def test_falls_back_on_zero(self) -> None:
        league = LeagueContext(1, 1, "Test")
        ranges = build_normalization_ranges(league)
        # Should use defaults.
        assert ranges["xg_for"] == (0.5, 3.0)


class TestExtractRadarData:
    def test_returns_radar_chart_data(self) -> None:
        comparison, baseline, mgr_a, mgr_b, league = _make_comparison()
        data = extract_radar_data(comparison, baseline, mgr_b, league)
        assert isinstance(data, RadarChartData)
        assert isinstance(data.branch_a, RadarAxes)
        assert isinstance(data.branch_b, RadarAxes)

    def test_values_in_zero_one_range(self) -> None:
        comparison, baseline, mgr_a, mgr_b, league = _make_comparison()
        data = extract_radar_data(comparison, baseline, mgr_b, league)
        for val in data.branch_a.values():
            assert 0.0 <= val <= 1.0, f"Branch A value {val} out of range"
        for val in data.branch_b.values():
            assert 0.0 <= val <= 1.0, f"Branch B value {val} out of range"

    def test_axes_count_matches_labels(self) -> None:
        comparison, baseline, mgr_a, mgr_b, league = _make_comparison()
        data = extract_radar_data(comparison, baseline, mgr_b, league)
        assert len(data.branch_a.values()) == len(data.labels)
        assert len(data.branch_b.values()) == len(data.labels)

    def test_branches_differ_on_tactical_estimates(self) -> None:
        comparison, baseline, mgr_a, mgr_b, league = _make_comparison()
        data = extract_radar_data(comparison, baseline, mgr_b, league)
        # PPDA should differ (Branch A = baseline, Branch B = estimate).
        assert data.branch_a.ppda != data.branch_b.ppda
        # Possession should differ.
        assert data.branch_a.possession != data.branch_b.possession

    def test_xga_identical_for_both_branches(self) -> None:
        comparison, baseline, mgr_a, mgr_b, league = _make_comparison()
        data = extract_radar_data(comparison, baseline, mgr_b, league)
        # xGA/90 is fixed baseline — must be identical for A and B.
        assert data.branch_a.xg_against == data.branch_b.xg_against

    def test_none_incoming_uses_neutral(self) -> None:
        comparison, baseline, mgr_a, _, league = _make_comparison()
        data = extract_radar_data(comparison, baseline, None, league)
        assert isinstance(data, RadarChartData)
        # All values should be valid.
        for val in data.branch_b.values():
            assert 0.0 <= val <= 1.0
