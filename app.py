"""IfFootball Streamlit UI.

Minimal single-page application for running what-if simulations and
displaying comparison results (radar charts, player impact, reports).

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import streamlit as st

from iffootball.agents.manager import ManagerAgent
from iffootball.candidates import CandidateResolver
from iffootball.agents.player import BroadPosition, PlayerAgent, RoleFamily
from iffootball.agents.trigger import (
    ChangeTrigger,
    ManagerChangeTrigger,
    TransferInTrigger,
)
from iffootball.collectors.statsbomb import StatsBombOpenDataCollector
from iffootball.config import SimulationRules
from iffootball.llm.client import LLMClient
from iffootball.llm.report_generation import (
    DEFAULT_LIMITATIONS,
    PlayerImpactEntry,
    ReportInput,
    generate_report,
)
from iffootball.pipeline import InitializationResult, initialize
from iffootball.storage.db import Database
from iffootball.simulation.comparison import ComparisonResult, run_comparison
from iffootball.simulation.turning_point import RuleBasedHandler
from iffootball.visualization.player_impact import PlayerImpact, rank_player_impact
from iffootball.visualization.player_radar import create_player_radar_figure
from iffootball.visualization.radar_chart import create_radar_figure
from iffootball.visualization.radar_data import extract_radar_data

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path(__file__).parent / "config"
_RULES_DIR = _CONFIG_DIR / "simulation_rules"
_TARGETS_PATH = _CONFIG_DIR / "targets.toml"
_CACHE_DIR = Path(__file__).parent / "data" / "demo_cache"

_DEFAULT_N_RUNS = 10
_DEFAULT_SEED = 42
_DEFAULT_TOP_PLAYERS = 3


# ---------------------------------------------------------------------------
# Typed params
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulationParams:
    """Typed container for sidebar input parameters."""

    competition_id: int
    season_id: int
    team_name: str
    manager_name: str
    trigger_week: int
    n_runs: int
    seed: int
    trigger_type: str  # "manager_change" | "transfer_in"
    # Manager change fields
    incoming_manager_name: str = ""
    # Transfer fields
    transfer_player_name: str = ""
    transfer_expected_role: str = "starter"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


@st.cache_data
def _load_targets() -> list[dict[str, Any]]:
    """Load competition targets from TOML config."""
    with _TARGETS_PATH.open("rb") as f:
        data = tomllib.load(f)
    return data["competitions"]  # type: ignore[no-any-return]


# Known competition names for human-readable display.
_COMPETITION_NAMES: dict[int, str] = {
    2: "Premier League",
    11: "La Liga",
}

# Known season labels.
_SEASON_NAMES: dict[int, str] = {
    27: "2015-16",
}


@st.cache_resource
def _get_resolver() -> CandidateResolver:
    """Create a cached CandidateResolver instance."""
    return CandidateResolver(StatsBombOpenDataCollector())


def _competition_label(target: dict[str, Any]) -> str:
    """Human-readable label for a competition target."""
    comp_id = int(target["competition_id"])
    season_id = int(target["season_id"])
    comp_name = _COMPETITION_NAMES.get(comp_id, f"Competition {comp_id}")
    season_name = _SEASON_NAMES.get(season_id, f"Season {season_id}")
    return f"{comp_name} {season_name}"


# ---------------------------------------------------------------------------
# Incoming manager profile
# ---------------------------------------------------------------------------

from iffootball.incoming_profile import resolve_incoming_profile  # noqa: E402


def _build_transfer_trigger(params: SimulationParams) -> TransferInTrigger:
    """Build a TransferInTrigger with a neutral-attribute PlayerAgent.

    The transfer player uses 50th-percentile technical attributes since
    we don't have StatsBomb data for hypothetical signings.
    """
    # Generate a unique player_id unlikely to collide with existing squad.
    player = PlayerAgent(
        player_id=99999,
        player_name=params.transfer_player_name,
        team_name="",
        position_name="Center Forward",
        role_family=RoleFamily.FORWARD,
        broad_position=BroadPosition.FW,
        pace=50.0,
        passing=50.0,
        shooting=50.0,
        pressing=50.0,
        defending=50.0,
        physicality=50.0,
        consistency=50.0,
    )
    return TransferInTrigger(
        player_name=params.transfer_player_name,
        expected_role=params.transfer_expected_role,
        applied_at=params.trigger_week,
        player=player,
    )


# ---------------------------------------------------------------------------
# Preset scenarios (shared with scripts/preview_presets.py)
# ---------------------------------------------------------------------------

from iffootball.presets import DEMO_PRESETS


# ---------------------------------------------------------------------------
# Input UI
# ---------------------------------------------------------------------------


def _render_input() -> SimulationParams | None:
    """Render guided scenario input and return parameters on selection."""
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

    # Report language is EN-only (canonical output).
    st.session_state["_report_lang"] = "en"

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

    # --- Main area: Preset cards ---
    st.subheader("What if...?")
    st.caption("Premier League 2015-16")

    # Render all preset cards first, then check which was pressed.
    selected_preset = None
    cols = st.columns(len(DEMO_PRESETS))
    for i, (col, preset) in enumerate(zip(cols, DEMO_PRESETS)):
        with col:
            with st.container(border=True):
                st.markdown(f"**{preset.label}**")
                st.caption(f"Week {preset.trigger_week}")
                if st.button("Run", key=f"preset_{i}", use_container_width=True):
                    selected_preset = preset

    if selected_preset is not None:
        return SimulationParams(
            competition_id=2,
            season_id=27,
            team_name=selected_preset.team_name,
            manager_name=selected_preset.manager_name,
            trigger_week=selected_preset.trigger_week,
            n_runs=n_runs,
            seed=seed,
            trigger_type="manager_change",
            incoming_manager_name=selected_preset.incoming_manager_name,
        )

    # --- Custom (experimental) ---
    with st.expander("Custom scenario (experimental)"):
        st.caption(
            "Advanced: build your own scenario. Candidates are resolved "
            "from StatsBomb data. Transfer triggers use neutral attributes."
        )

        targets = _load_targets()
        target_labels = [_competition_label(t) for t in targets]
        selected_idx = int(
            st.selectbox(
                "Competition / Season",
                range(len(targets)),
                format_func=lambda i: target_labels[i],
            )
            or 0
        )
        target = targets[selected_idx]
        comp_id = int(target["competition_id"])
        szn_id = int(target["season_id"])

        # Resolve teams from targets.
        clubs: list[str] = target["clubs"]
        team_name = str(st.selectbox("Team", clubs) or clubs[0])

        trigger_week: int = st.slider(
            "Trigger Week", min_value=1, max_value=38, value=10
        )

        # Resolve current manager at the selected trigger_week.
        resolver = _get_resolver()
        canonical = resolver.manager_at_week(comp_id, szn_id, team_name, trigger_week)
        team_managers = resolver.managers(
            comp_id, szn_id, team_name, at_week=trigger_week
        )
        mgr_names = [c.manager_name for c in team_managers]
        if mgr_names:
            # Default to the canonical manager at that week.
            default_idx = 0
            if canonical and canonical.manager_name in mgr_names:
                default_idx = mgr_names.index(canonical.manager_name)
            manager_name = str(
                st.selectbox("Current Manager", mgr_names, index=default_idx)
                or mgr_names[0]
            )
        else:
            manager_name = st.text_input(
                "Current Manager Name",
                placeholder="e.g., Louis van Gaal",
            )

        trigger_type = str(
            st.radio(
                "Trigger Type",
                options=["manager_change", "transfer_in"],
                format_func=lambda x: {
                    "manager_change": "Manager Change",
                    "transfer_in": "Player Transfer (Experimental)",
                }[x],
            )
            or "manager_change"
        )

        incoming_manager_name = ""
        transfer_player_name = ""
        transfer_expected_role = "starter"

        if trigger_type == "manager_change":
            # Resolve incoming candidates (same league, exclude current team).
            incoming_candidates = resolver.incoming_candidates(
                comp_id, szn_id, exclude_team=team_name
            )
            incoming_names = [
                f"{c.manager_name} ({c.team_name})" for c in incoming_candidates
            ]
            if incoming_names:
                selected_incoming = str(
                    st.selectbox("Incoming Manager", incoming_names)
                    or incoming_names[0]
                )
                # Extract manager name (before the parenthetical team name).
                idx = incoming_names.index(selected_incoming)
                incoming_manager_name = incoming_candidates[idx].manager_name
            else:
                incoming_manager_name = st.text_input(
                    "Incoming Manager Name",
                    placeholder="e.g., José Mourinho",
                )
        else:
            transfer_player_name = st.text_input(
                "Player Name",
                placeholder="e.g., Kylian Mbappe",
            )
            transfer_expected_role = str(
                st.selectbox(
                    "Expected Role",
                    options=["starter", "rotation", "squad"],
                )
                or "starter"
            )

        if st.button("Run Custom Scenario", use_container_width=True):
            if not manager_name.strip():
                st.error("Current Manager Name is required.")
                return None
            if trigger_type == "manager_change" and not incoming_manager_name.strip():
                st.error("Incoming Manager Name is required.")
                return None
            if trigger_type == "transfer_in" and not transfer_player_name.strip():
                st.error("Player Name is required.")
                return None

            return SimulationParams(
                competition_id=comp_id,
                season_id=szn_id,
                team_name=team_name,
                manager_name=manager_name.strip(),
                trigger_week=trigger_week,
                n_runs=n_runs,
                seed=seed,
                trigger_type=trigger_type,
                incoming_manager_name=incoming_manager_name.strip(),
                transfer_player_name=transfer_player_name.strip(),
                transfer_expected_role=transfer_expected_role,
            )

    return None


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------


def _team_cache_path(team_name: str, trigger_week: int) -> Path:
    """Return the per-team+week cache DB path."""
    safe_name = team_name.replace(" ", "_").lower()
    return _CACHE_DIR / f"{safe_name}_w{trigger_week}.db"


def _load_from_cache(params: SimulationParams) -> InitializationResult | None:
    """Try to load initialization data from per-scenario demo cache DB.

    Cache is keyed by team_name + trigger_week. Returns None if cache
    file doesn't exist or any required component (player_agents,
    team_baseline, manager_agent, fixture_list, opponent_strengths,
    league_context) is missing. Manager lookup requires exact name match.
    """
    cache_path = _team_cache_path(params.team_name, params.trigger_week)
    if not cache_path.exists():
        return None

    try:
        db = Database(cache_path)
    except Exception:
        return None

    try:
        player_agents = db.load_player_agents(
            params.competition_id, params.season_id
        )
        if not player_agents:
            return None

        team_baseline = db.load_team_baseline(
            params.team_name, params.competition_id, params.season_id
        )
        if team_baseline is None:
            return None

        # Load manager by team name — cache stores the canonical manager.
        # Try user-specified name first, then fall back to any cached manager.
        manager_agent = db.load_manager_agent(
            params.manager_name,
            params.team_name,
            params.competition_id,
            params.season_id,
        )
        if manager_agent is None:
            # Manager name mismatch — cache miss for this specific manager.
            return None

        fixture_list = db.load_fixture_list(
            params.team_name,
            params.competition_id,
            params.season_id,
        )
        if fixture_list is None:
            return None

        opponent_strengths = db.load_opponent_strengths(
            params.competition_id, params.season_id, params.trigger_week
        )
        if not opponent_strengths:
            return None

        league_context = db.load_league_context(
            params.competition_id, params.season_id
        )
        if league_context is None:
            return None

        return InitializationResult(
            player_agents=player_agents,
            team_baseline=team_baseline,
            manager_agent=manager_agent,
            fixture_list=fixture_list,
            opponent_strengths=opponent_strengths,
            league_context=league_context,
        )
    except Exception:
        return None
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------


def _run_pipeline(params: SimulationParams) -> bool:
    """Execute the full pipeline, cache results in session_state, and display.

    Returns True on success, False on failure.
    """
    progress = st.progress(0, text="Initializing...")

    # 1. Initialize agents (try cache first, then StatsBomb API).
    init_result = _load_from_cache(params)
    if init_result is not None:
        st.caption("Loaded from demo cache.")
    else:
        collector = StatsBombOpenDataCollector()
        try:
            init_result = initialize(
                collector=collector,
                competition_id=params.competition_id,
                season_id=params.season_id,
                team_name=params.team_name,
                manager_name=params.manager_name,
                trigger_week=params.trigger_week,
                league_name=f"Competition {params.competition_id}",
            )
        except Exception as e:
            st.error(f"Initialization failed: {e}")
            return False

    progress.progress(30, text="Running simulation...")

    # 2. Build trigger.
    trigger: ChangeTrigger
    incoming_profile: ManagerAgent | None = None

    if params.trigger_type == "transfer_in":
        trigger = _build_transfer_trigger(params)
    else:
        incoming_profile = resolve_incoming_profile(
            params.incoming_manager_name,
            params.competition_id,
            params.season_id,
            cache_dir=_CACHE_DIR,
        )
        trigger = ManagerChangeTrigger(
            outgoing_manager_name=params.manager_name,
            incoming_manager_name=params.incoming_manager_name,
            transition_type="mid_season",
            applied_at=params.trigger_week,
            incoming_profile=incoming_profile,
        )

    # 3. Run comparison.
    rules = SimulationRules.load(_RULES_DIR)
    handler = RuleBasedHandler(rules)

    try:
        comparison = run_comparison(
            team=init_result.team_baseline,
            squad=init_result.player_agents,
            manager=init_result.manager_agent,
            fixture_list=init_result.fixture_list,
            opponent_strengths=init_result.opponent_strengths,
            rules=rules,
            handler=handler,
            trigger=trigger,
            n_runs=params.n_runs,
            rng_seed=params.seed,
        )
    except Exception as e:
        st.error(f"Simulation failed: {e}")
        return False

    progress.progress(70, text="Generating visualizations...")

    # 4. Compute player impacts.
    impacts = rank_player_impact(comparison, top_n=_DEFAULT_TOP_PLAYERS)

    # 5. Cache results in session_state for rerun display.
    st.session_state["_sim_comparison"] = comparison
    st.session_state["_sim_init_result"] = init_result
    st.session_state["_sim_impacts"] = impacts
    st.session_state["_sim_incoming_profile"] = incoming_profile

    progress.progress(100, text="Complete.")

    # 6. Display results.
    _display_results(comparison, params, init_result, impacts, incoming_profile)
    return True


def _display_cached_results() -> None:
    """Display results from session_state without recomputing."""
    params: SimulationParams = st.session_state["_sim_params"]
    comparison: ComparisonResult = st.session_state["_sim_comparison"]
    init_result: InitializationResult = st.session_state["_sim_init_result"]
    impacts: list[PlayerImpact] = st.session_state["_sim_impacts"]
    incoming_profile: ManagerAgent | None = st.session_state["_sim_incoming_profile"]

    _display_results(comparison, params, init_result, impacts, incoming_profile)


def _display_results(
    comparison: ComparisonResult,
    params: SimulationParams,
    init_result: InitializationResult,
    impacts: list[PlayerImpact],
    incoming_profile: ManagerAgent | None,
) -> None:
    """Render all output sections from precomputed results."""
    llm_client = _get_llm_client()

    _render_summary(comparison, params, impacts)
    _render_player_impact(impacts, params)
    _render_report(comparison, params, impacts, llm_client)
    if params.trigger_type == "manager_change":
        _render_team_radar(comparison, init_result, incoming_profile)
    else:
        st.header("Team Comparison")
        st.info(
            "Team radar chart is not applicable for transfer triggers. "
            "Tactical estimates are manager-driven and unchanged by player transfers."
        )


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
    params: SimulationParams,
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

    # One-line conclusion (varies by trigger type).
    top_player = impacts[0].player_name if impacts else "no player"
    n_runs = len(comparison.no_change.run_results) or comparison.no_change.n_runs

    if params.trigger_type == "transfer_in":
        headline = (
            f"**{params.team_name}**: Signing {params.transfer_player_name} "
            f"shows a **{direction}** impact. Most affected: **{top_player}**."
        )
    else:
        headline = (
            f"**{params.team_name}**: Dismissing {params.manager_name} shows a "
            f"**{direction}** impact. Most affected: **{top_player}**."
        )

    # Summary card.
    with st.container(border=True):
        st.markdown(headline)
        st.caption(
            f"Points below are mean total points across remaining fixtures over {n_runs} simulation runs."
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
    init_result: InitializationResult,
    incoming_profile: ManagerAgent | None,
) -> None:
    """Display team radar chart."""
    st.header("Team Comparison")
    st.caption(
        "Detailed view for deeper analysis. "
        "Tactical estimates use neutral defaults for the incoming manager."
    )

    radar_data = extract_radar_data(
        comparison,
        init_result.team_baseline,
        incoming_profile,
        init_result.league_context,
    )
    fig = create_radar_figure(radar_data)
    # Center the chart at player-radar width (1/3 of page).
    _, center, _ = st.columns(3)
    with center:
        st.pyplot(fig)
    plt.close(fig)


def _impact_direction(impact: PlayerImpact) -> tuple[str, str]:
    """Return arrow symbol and one-line reason for a player's impact."""
    form_diff = impact.mean_form_b - impact.mean_form_a
    trust_diff = impact.mean_trust_b - impact.mean_trust_a
    understanding_diff = impact.mean_understanding_b - impact.mean_understanding_a

    # Determine overall direction from form change.
    if form_diff > 0.02:
        arrow = "▲"
    elif form_diff < -0.02:
        arrow = "▼"
    else:
        arrow = "─"

    # Pick the most notable change for the one-line reason.
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


def _render_player_impact(
    impacts: list[PlayerImpact],
    params: SimulationParams,
) -> None:
    """Display player impact radar charts."""
    st.header("Player Impact")

    if params.trigger_type == "transfer_in":
        st.caption(
            "Transfer trigger: the new signing only exists in the 'with change' "
            "branch, so they are shown separately below. Existing player rankings "
            "reflect indirect effects of the squad addition."
        )
        # Show new signing info (with-change branch only).
        _render_transfer_player_info(params)

    if not impacts:
        st.info("No significant player impact detected among existing squad.")
        return

    # Direction arrows + one-line reason for each player.
    for impact in impacts:
        arrow, reason = _impact_direction(impact)
        st.markdown(f"{arrow} **{impact.player_name}** — {reason}")

    st.divider()

    # Horizontal radar layout: one column per player.
    cols = st.columns(len(impacts))
    for col, impact in zip(cols, impacts):
        with col:
            st.markdown(f"**{impact.player_name}**")
            st.caption(f"Impact: {impact.impact_score:.3f}")
            fig = create_player_radar_figure(impact)
            st.pyplot(fig)
            plt.close(fig)


def _render_transfer_player_info(params: SimulationParams) -> None:
    """Display a summary card for the transferred player."""
    st.subheader(f"Simulated Signing: {params.transfer_player_name}")
    st.write(
        f"- **Simulated role:** {params.transfer_expected_role}\n"
        f"- **Available from:** week {params.trigger_week + 1} (simulated)\n"
        f"- **Attributes:** neutral (50th percentile) — no StatsBomb data\n"
        f"- **Note:** This is a hypothetical signing. The player only exists "
        f"in the 'with change' branch only and is not included in the impact ranking above."
    )


def _render_report(
    comparison: ComparisonResult,
    params: SimulationParams,
    impacts: list[PlayerImpact],
    llm_client: LLMClient | None,
) -> None:
    """Display the comparison report.

    Uses LLM-generated report when a client is available, otherwise
    falls back to a data-only structured report.
    """
    st.header("Detailed Analysis")

    if params.trigger_type == "transfer_in":
        trigger_desc = (
            f"Simulated transfer: {params.transfer_player_name} "
            f"({params.transfer_expected_role}) joining at week {params.trigger_week}"
        )
    else:
        trigger_desc = (
            f"Simulated manager change: {params.manager_name} -> "
            f"{params.incoming_manager_name} at week {params.trigger_week}"
        )

    # Build player impact entries.
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
            "Generating report via LLM. Scenario data (team names, player "
            "names, simulation results) is sent to the configured provider. "
            "Data handling follows the provider's policy."
        )
        try:
            report_md = generate_report(llm_client, report_input)
            # Strip Summary section — already shown at page top.
            report_md = _strip_summary_section(report_md)
            st.markdown(report_md)
            return
        except Exception:
            st.warning("LLM report generation failed. Falling back to data report.")

    # Fallback: data-only report.
    _render_data_report(report_input)


def _strip_summary_section(report_md: str) -> str:
    """Remove the Summary section from LLM report (shown separately at top)."""
    summary_heading = "## Summary"
    if summary_heading not in report_md:
        return report_md

    # Find Summary heading and the next ## heading after it.
    start = report_md.index(summary_heading)
    rest = report_md[start + len(summary_heading) :]
    next_heading = rest.find("\n## ")
    if next_heading == -1:
        # Summary is the only section — return empty.
        return report_md[:start].strip()
    # Remove from Summary heading to the next heading.
    return (report_md[:start] + rest[next_heading + 1 :]).strip()


def _render_data_report(report_input: ReportInput) -> None:
    """Render a structured report from data without LLM.

    Summary and Player Impact are rendered separately above, so this
    section covers Key Differences, Causal Chain, and Limitations only.
    """
    st.subheader("Key Differences")
    st.caption(
        f"Mean total points across remaining fixtures over "
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
    if report_input.action_explanations:
        for a in report_input.action_explanations:
            st.write(
                f"- **{a.tp_type} -> {a.action}**: "
                f"{a.explanation} [{a.label}]"
            )
    else:
        if report_input.cascade_count_diff:
            st.write(
                "Causal chain summary from cascade event frequencies "
                "(LLM-based detailed analysis available with API key):"
            )
            for et, diff in report_input.cascade_count_diff.items():
                direction = "increased by" if diff > 0 else "decreased by"
                st.write(f"- **{et}** {direction} {abs(diff):.1f} /run [data]")
        else:
            st.write("No cascade events recorded in this simulation.")

    st.subheader("Limitations")
    for limitation in report_input.limitations:
        st.write(f"- {limitation}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _params_key(params: SimulationParams) -> str:
    """Create a hashable key from simulation params for session caching."""
    return (
        f"{params.competition_id}_{params.season_id}_{params.team_name}_"
        f"{params.manager_name}_{params.trigger_week}_{params.trigger_type}_"
        f"{params.incoming_manager_name}_{params.transfer_player_name}_"
        f"{params.transfer_expected_role}_{params.n_runs}_{params.seed}"
    )


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

    # Input: buttons store params in session_state.
    new_params = _render_input()

    if new_params is not None:
        # New simulation requested — compute and cache.
        key = _params_key(new_params)
        if st.session_state.get("_sim_key") != key:
            success = _run_pipeline(new_params)
            if success:
                st.session_state["_sim_key"] = key
                st.session_state["_sim_params"] = new_params
        else:
            # Same params — just display cached results.
            _display_cached_results()
    elif "_sim_params" in st.session_state:
        # No new button press — display previous results.
        _display_cached_results()
    else:
        st.info("Select a scenario above or build a custom one to start.")


if __name__ == "__main__":
    main()
