"""IfFootball Streamlit UI.

Season-scenario application: select a team, choose a what-if scenario
(manager change, player add, player remove), and view comparison results.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from iffootball.agents.manager import ManagerAgent
from iffootball.config import SimulationRules
from iffootball.llm.client import LLMClient
from iffootball.llm.report_generation import (
    DEFAULT_LIMITATIONS,
    PlayerImpactEntry,
    ReportInput,
    generate_report,
)
from iffootball.scenario import ScenarioDefinition, load_team_season_state, run_scenario
from iffootball.simulation.comparison import ComparisonResult
from iffootball.storage.db import Database
from iffootball.visualization.player_impact import PlayerImpact, rank_player_impact
from iffootball.visualization.player_radar import create_player_radar_figure
from iffootball.visualization.radar_chart import create_radar_figure
from iffootball.visualization.radar_data import extract_radar_data

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path(__file__).parent / "config"
_RULES_DIR = _CONFIG_DIR / "simulation_rules"
_SEASON_CACHE_DIR = Path(__file__).parent / "data" / "season_cache"

_DEFAULT_N_RUNS = 10
_DEFAULT_SEED = 42
_DEFAULT_TOP_PLAYERS = 3

# Premier League 2015-16 (only supported season for now).
_COMP_ID = 2
_SEASON_ID = 27
_LEAGUE_LABEL = "Premier League 2015-16"


# ---------------------------------------------------------------------------
# Season cache helpers
# ---------------------------------------------------------------------------


def _season_cache_path() -> Path:
    return _SEASON_CACHE_DIR / "premier_league_2015-16.db"


def _open_season_db() -> Database | None:
    """Open the season cache DB.

    Creates a new connection per call. SQLite connections are not
    thread-safe, and Streamlit runs scripts in different threads
    across reruns, so caching a single connection causes errors.
    """
    path = _season_cache_path()
    if not path.exists():
        return None
    try:
        return Database(path)
    except Exception:
        return None


def _load_team_list(db: Database) -> list[str]:
    """Get all team names from season cache."""
    rows = db._conn.execute(  # noqa: SLF001
        "SELECT DISTINCT team_name FROM team_baselines "
        "WHERE competition_id=? AND season_id=? ORDER BY team_name",
        (_COMP_ID, _SEASON_ID),
    ).fetchall()
    return [str(r["team_name"]) for r in rows]


def _load_manager_candidates(
    db: Database, exclude_team: str,
) -> list[tuple[str, str]]:
    """Get (manager_name, team_name) pairs excluding the target team.

    Candidates are restricted to the same competition/season.
    Cross-league candidates are not supported in the current model.
    """
    rows = db._conn.execute(  # noqa: SLF001
        "SELECT manager_name, team_name FROM manager_agents "
        "WHERE competition_id=? AND season_id=? AND team_name != ? "
        "ORDER BY team_name, manager_name",
        (_COMP_ID, _SEASON_ID, exclude_team),
    ).fetchall()
    return [(str(r["manager_name"]), str(r["team_name"])) for r in rows]


def _load_player_candidates(
    db: Database, exclude_team: str,
) -> list[tuple[int, str, str]]:
    """Get (player_id, player_name, team_name) for other teams.

    Candidates are restricted to the same competition/season.
    Cross-league player additions are not supported in the current model.
    """
    rows = db._conn.execute(  # noqa: SLF001
        "SELECT player_id, player_name, team_name FROM player_agents "
        "WHERE competition_id=? AND season_id=? AND team_name != ? "
        "ORDER BY team_name, player_name",
        (_COMP_ID, _SEASON_ID, exclude_team),
    ).fetchall()
    return [
        (int(r["player_id"]), str(r["player_name"]), str(r["team_name"]))
        for r in rows
    ]


def _load_squad_players(
    db: Database, team_name: str,
) -> list[tuple[int, str]]:
    """Get (player_id, player_name) for a team's squad."""
    agents = db.load_player_agents(_COMP_ID, _SEASON_ID, team_name=team_name)
    return [(a.player_id, a.player_name) for a in agents]


# ---------------------------------------------------------------------------
# Input UI
# ---------------------------------------------------------------------------


def _render_input() -> ScenarioDefinition | None:
    """Render scenario input and return definition on selection."""
    # --- Sidebar: Advanced Settings + LLM status ---
    with st.sidebar.expander("Advanced Settings"):
        n_runs = int(
            st.number_input(
                "Number of Runs", min_value=1, max_value=100, value=_DEFAULT_N_RUNS
            )
        )
        seed = int(
            st.number_input(
                "Random Seed", min_value=0, max_value=99999, value=_DEFAULT_SEED
            )
        )
    st.session_state["_n_runs"] = n_runs
    st.session_state["_seed"] = seed

    # LLM status display.
    try:
        from iffootball.llm.providers import available_providers

        providers = available_providers()
        if providers:
            st.sidebar.success(f"LLM: {', '.join(providers)}")
        else:
            st.sidebar.info("LLM: not configured (data-only mode)")
    except Exception:
        st.sidebar.info("LLM: not available")

    # --- Check season cache ---
    db = _open_season_db()
    if db is None:
        st.warning(
            "Season cache not found. Run:\n\n"
            "```\nuv run python scripts/build_season_cache.py\n```"
        )
        return None

    # --- Main area ---
    st.subheader("What if...?")
    st.caption(_LEAGUE_LABEL)

    # Team selection
    teams = _load_team_list(db)
    if not teams:
        st.error("No teams found in season cache.")
        return None

    team_name = str(st.selectbox("Team", teams) or teams[0])

    # Scenario type
    scenario_type = str(
        st.radio(
            "Scenario",
            options=["manager_change", "player_add", "player_remove"],
            format_func={
                "manager_change": "What if a different manager?",
                "player_add": "What if we signed a player?",
                "player_remove": "What if without a player?",
            }.get,
            horizontal=True,
        )
        or "manager_change"
    )

    # Scenario-specific inputs
    alt_manager_name: str | None = None
    alt_manager_team_name: str | None = None
    player_id: int | None = None
    player_name: str | None = None
    expected_role: str | None = None

    if scenario_type == "manager_change":
        candidates = _load_manager_candidates(db, team_name)
        if not candidates:
            st.warning("No manager candidates found.")
            return None
        labels = [f"{mgr} ({team})" for mgr, team in candidates]
        selected = str(st.selectbox("Incoming Manager", labels) or labels[0])
        idx = labels.index(selected)
        alt_manager_name = candidates[idx][0]
        alt_manager_team_name = candidates[idx][1]

    elif scenario_type == "player_add":
        candidates_p = _load_player_candidates(db, team_name)
        if not candidates_p:
            st.warning("No player candidates found.")
            return None
        labels_p = [f"{name} ({team})" for _, name, team in candidates_p]
        selected_p = str(st.selectbox("Player to Add", labels_p) or labels_p[0])
        idx_p = labels_p.index(selected_p)
        player_id = candidates_p[idx_p][0]
        player_name = candidates_p[idx_p][1]
        expected_role = str(
            st.selectbox("Expected Role", ["starter", "rotation", "squad"])
            or "starter"
        )

    else:  # player_remove
        squad = _load_squad_players(db, team_name)
        if not squad:
            st.warning("No players found in squad.")
            return None
        labels_s = [f"{name} (ID: {pid})" for pid, name in squad]
        selected_s = str(st.selectbox("Player to Remove", labels_s) or labels_s[0])
        idx_s = labels_s.index(selected_s)
        player_id = squad[idx_s][0]
        player_name = squad[idx_s][1]

    # Run button
    if st.button("Run Scenario", use_container_width=True, type="primary"):
        return ScenarioDefinition(
            team_name=team_name,
            competition_id=_COMP_ID,
            season_id=_SEASON_ID,
            scenario_type=scenario_type,  # type: ignore[arg-type]
            alt_manager_name=alt_manager_name,
            alt_manager_team_name=alt_manager_team_name,
            player_id=player_id,
            player_name=player_name,
            expected_role=expected_role,
        )

    return None


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------


def _run_pipeline(scenario: ScenarioDefinition) -> bool:
    """Execute scenario comparison and display results.

    Returns True on success, False on failure.
    """
    db = _open_season_db()
    if db is None:
        st.error("Season cache not available.")
        return False

    progress = st.progress(0, text="Loading team data...")

    n_runs: int = st.session_state.get("_n_runs", _DEFAULT_N_RUNS)
    seed: int = st.session_state.get("_seed", _DEFAULT_SEED)
    rules = SimulationRules.load(_RULES_DIR)

    progress.progress(10, text="Running simulation...")

    try:
        comparison = run_scenario(
            scenario, db, rules, n_runs=n_runs, rng_seed=seed,
        )
    except Exception as e:
        st.error(f"Simulation failed: {e}")
        return False

    progress.progress(70, text="Generating visualizations...")

    # Compute player impacts.
    impacts = rank_player_impact(comparison, top_n=_DEFAULT_TOP_PLAYERS)

    # Load state for radar chart.
    try:
        state = load_team_season_state(
            db, scenario.team_name, _COMP_ID, _SEASON_ID
        )
    except Exception:
        state = None

    # Resolve incoming manager profile for radar.
    incoming_profile: ManagerAgent | None = None
    if scenario.scenario_type == "manager_change" and scenario.alt_manager_name:
        if scenario.alt_manager_team_name:
            incoming_profile = db.load_manager_agent(
                scenario.alt_manager_name,
                scenario.alt_manager_team_name,
                _COMP_ID, _SEASON_ID,
            )
        else:
            rows = db._conn.execute(  # noqa: SLF001
                "SELECT manager_name, team_name FROM manager_agents "
                "WHERE manager_name=? AND competition_id=? AND season_id=?",
                (scenario.alt_manager_name, _COMP_ID, _SEASON_ID),
            ).fetchall()
            if rows:
                incoming_profile = db.load_manager_agent(
                    str(rows[0]["manager_name"]),
                    str(rows[0]["team_name"]),
                    _COMP_ID, _SEASON_ID,
                )

    # Cache in session.
    st.session_state["_sim_comparison"] = comparison
    st.session_state["_sim_impacts"] = impacts
    st.session_state["_sim_state"] = state
    st.session_state["_sim_incoming_profile"] = incoming_profile

    progress.progress(100, text="Complete.")

    _display_results(comparison, scenario, impacts, state, incoming_profile)
    return True


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _display_cached_results() -> None:
    """Display results from session_state without recomputing."""
    scenario: ScenarioDefinition = st.session_state["_sim_scenario"]
    comparison: ComparisonResult = st.session_state["_sim_comparison"]
    impacts: list[PlayerImpact] = st.session_state["_sim_impacts"]
    state = st.session_state.get("_sim_state")
    incoming_profile: ManagerAgent | None = st.session_state.get("_sim_incoming_profile")

    _display_results(comparison, scenario, impacts, state, incoming_profile)


def _display_results(
    comparison: ComparisonResult,
    scenario: ScenarioDefinition,
    impacts: list[PlayerImpact],
    state: object | None,
    incoming_profile: ManagerAgent | None,
) -> None:
    """Render all output sections."""
    llm_client = _get_llm_client()

    _render_summary(comparison, scenario, impacts)
    _render_player_impact(impacts)
    _render_report(comparison, scenario, impacts, llm_client)
    if scenario.scenario_type == "manager_change" and state is not None:
        _render_team_radar(comparison, state, incoming_profile)


# ---------------------------------------------------------------------------
# LLM client resolution
# ---------------------------------------------------------------------------


def _get_llm_client() -> LLMClient | None:
    """Resolve an LLM client from environment, or return None."""
    try:
        from iffootball.llm.providers import available_providers, create_client

        providers = available_providers()
        if not providers:
            return None
        return create_client()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------


def _render_summary(
    comparison: ComparisonResult,
    scenario: ScenarioDefinition,
    impacts: list[PlayerImpact],
) -> None:
    """Display one-line conclusion + key metrics."""
    st.header("Summary")

    delta = comparison.delta.points_mean_diff
    if delta > 0.5:
        direction = "positive"
    elif delta < -0.5:
        direction = "negative"
    else:
        direction = "marginal"

    top_player = impacts[0].player_name if impacts else "no player"
    n_runs = len(comparison.no_change.run_results) or comparison.no_change.n_runs

    if scenario.scenario_type == "manager_change":
        headline = (
            f"**{scenario.team_name}**: What if {scenario.alt_manager_name} "
            f"managed the team? **{direction}** impact. "
            f"Most affected: **{top_player}**."
        )
    elif scenario.scenario_type == "player_add":
        headline = (
            f"**{scenario.team_name}**: What if {scenario.player_name} "
            f"joined? **{direction}** impact. "
            f"Most affected: **{top_player}**."
        )
    else:
        headline = (
            f"**{scenario.team_name}**: What if without {scenario.player_name}? "
            f"**{direction}** impact. Most affected: **{top_player}**."
        )

    with st.container(border=True):
        st.markdown(headline)
        st.caption(
            f"Points are mean total across 38 fixtures over {n_runs} simulation runs."
        )
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Points (mean)",
            f"{comparison.with_change.total_points_mean:.1f}",
            delta=f"{delta:+.1f}",
        )
        col2.metric(
            "Points (median)",
            f"{comparison.with_change.total_points_median:.1f}",
            delta=f"{comparison.delta.points_median_diff:+.1f}",
        )
        col3.metric("Runs", f"{n_runs}")
        st.caption("This is a scenario comparison, not a prediction.")


def _render_team_radar(
    comparison: ComparisonResult,
    state: object,
    incoming_profile: ManagerAgent | None,
) -> None:
    """Display team radar chart."""
    from iffootball.scenario import TeamSeasonState

    if not isinstance(state, TeamSeasonState):
        return

    st.header("Team Comparison")
    st.caption("Tactical estimates for baseline vs alternative manager.")

    league_ctx = None
    db = _open_season_db()
    if db is not None:
        league_ctx = db.load_league_context(_COMP_ID, _SEASON_ID)

    if league_ctx is None:
        return

    radar_data = extract_radar_data(
        comparison, state.baseline, incoming_profile, league_ctx,
    )
    fig = create_radar_figure(radar_data)
    _, center, _ = st.columns(3)
    with center:
        st.pyplot(fig)
    plt.close(fig)


def _impact_direction(impact: PlayerImpact) -> tuple[str, str]:
    """Return arrow symbol and one-line reason for a player's impact."""
    form_diff = impact.mean_form_b - impact.mean_form_a
    trust_diff = impact.mean_trust_b - impact.mean_trust_a
    understanding_diff = impact.mean_understanding_b - impact.mean_understanding_a

    if form_diff > 0.02:
        arrow = "▲"
    elif form_diff < -0.02:
        arrow = "▼"
    else:
        arrow = "─"

    changes: list[tuple[float, str]] = [
        (abs(form_diff), f"form {'up' if form_diff > 0 else 'down'}"),
        (abs(trust_diff), f"trust {'gained' if trust_diff > 0 else 'lost'}"),
        (
            abs(understanding_diff),
            f"tactical understanding {'improved' if understanding_diff > 0 else 'dropped'}",
        ),
    ]
    changes.sort(key=lambda x: x[0], reverse=True)
    reason = changes[0][1] if changes[0][0] > 0.01 else "minimal change"

    return arrow, reason


def _render_player_impact(impacts: list[PlayerImpact]) -> None:
    """Display player impact radar charts."""
    st.header("Player Impact")

    if not impacts:
        st.info("No significant player impact detected.")
        return

    for impact in impacts:
        arrow, reason = _impact_direction(impact)
        st.markdown(f"{arrow} **{impact.player_name}** — {reason}")

    st.divider()

    cols = st.columns(len(impacts))
    for col, impact in zip(cols, impacts):
        with col:
            st.markdown(f"**{impact.player_name}**")
            st.caption(f"Impact: {impact.impact_score:.3f}")
            fig = create_player_radar_figure(impact)
            st.pyplot(fig)
            plt.close(fig)


def _render_report(
    comparison: ComparisonResult,
    scenario: ScenarioDefinition,
    impacts: list[PlayerImpact],
    llm_client: LLMClient | None,
) -> None:
    """Display the comparison report."""
    st.header("Detailed Analysis")

    if scenario.scenario_type == "manager_change":
        trigger_desc = (
            f"Season-start scenario: {scenario.alt_manager_name} "
            f"replaces the baseline manager at {scenario.team_name}"
        )
    elif scenario.scenario_type == "player_add":
        trigger_desc = (
            f"Season-start scenario: {scenario.player_name} "
            f"joins {scenario.team_name} ({scenario.expected_role})"
        )
    else:
        trigger_desc = (
            f"Season-start scenario: {scenario.team_name} "
            f"without {scenario.player_name}"
        )

    player_entries = [
        PlayerImpactEntry(
            player_name=p.player_name,
            impact_score=p.impact_score,
            form_diff=p.mean_form_b - p.mean_form_a,
            fatigue_diff=p.mean_fatigue_b - p.mean_fatigue_a,
            understanding_diff=p.mean_understanding_b - p.mean_understanding_a,
            trust_diff=p.mean_trust_b - p.mean_trust_a,
        )
        for p in impacts
    ]

    report_input = ReportInput(
        trigger_description=trigger_desc,
        points_mean_a=comparison.no_change.total_points_mean,
        points_mean_b=comparison.with_change.total_points_mean,
        points_mean_diff=comparison.delta.points_mean_diff,
        cascade_count_diff=comparison.delta.cascade_count_diff,
        n_runs=comparison.no_change.n_runs,
        player_impacts=player_entries,
        action_explanations=[],
        limitations=list(DEFAULT_LIMITATIONS["en"]),
    )

    if llm_client is not None:
        st.caption(
            "Generating report via LLM. Scenario data is sent to the configured provider."
        )
        try:
            report_md = generate_report(llm_client, report_input)
            report_md = _strip_summary_section(report_md)
            st.markdown(report_md)
            return
        except Exception:
            st.warning("LLM report generation failed. Falling back to data report.")

    _render_data_report(report_input)


def _strip_summary_section(report_md: str) -> str:
    """Remove the Summary section from LLM report (shown separately at top)."""
    summary_heading = "## Summary"
    if summary_heading not in report_md:
        return report_md

    start = report_md.index(summary_heading)
    rest = report_md[start + len(summary_heading) :]
    next_heading = rest.find("\n## ")
    if next_heading == -1:
        return report_md[:start].strip()
    return (report_md[:start] + rest[next_heading + 1 :]).strip()


def _render_data_report(report_input: ReportInput) -> None:
    """Render a structured report from data without LLM."""
    st.subheader("Key Differences")
    st.caption(
        f"Mean total points across 38 fixtures over "
        f"{report_input.n_runs} simulation runs."
    )
    st.write(
        f"- Mean total points: no change = {report_input.points_mean_a:.1f}, "
        f"with change = {report_input.points_mean_b:.1f} "
        f"(diff: {report_input.points_mean_diff:+.1f}) [data]"
    )
    for event_type, diff in report_input.cascade_count_diff.items():
        st.write(f"- {event_type}: {diff:+.2f} /run [data]")

    st.subheader("Causal Chain")
    if report_input.cascade_count_diff:
        for et, diff in report_input.cascade_count_diff.items():
            direction = "increased by" if diff > 0 else "decreased by"
            st.write(f"- **{et}** {direction} {abs(diff):.1f} /run [data]")
    else:
        st.write("No cascade events recorded.")

    st.subheader("Limitations")
    for limitation in report_input.limitations:
        st.write(f"- {limitation}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _scenario_key(scenario: ScenarioDefinition) -> str:
    """Create a hashable key from scenario for session caching."""
    n_runs = st.session_state.get("_n_runs", _DEFAULT_N_RUNS)
    seed = st.session_state.get("_seed", _DEFAULT_SEED)
    return f"{scenario.scenario_key}_{n_runs}_{seed}"


def main() -> None:
    """Streamlit app entry point."""
    st.set_page_config(
        page_title="IfFootball",
        page_icon="",
        layout="wide",
    )

    st.title("IfFootball - What-If Simulation")
    st.caption(
        "IfFootball is a what-if simulation tool, not a prediction engine. "
        "Results should not be used for real-world decision-making."
    )

    new_scenario = _render_input()

    if new_scenario is not None:
        key = _scenario_key(new_scenario)
        if st.session_state.get("_sim_key") != key:
            success = _run_pipeline(new_scenario)
            if success:
                st.session_state["_sim_key"] = key
                st.session_state["_sim_scenario"] = new_scenario
        else:
            _display_cached_results()
    elif "_sim_scenario" in st.session_state:
        _display_cached_results()
    else:
        st.info("Select a team and scenario above to start.")


if __name__ == "__main__":
    main()
