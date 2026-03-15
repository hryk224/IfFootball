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


def _competition_label(target: dict[str, Any]) -> str:
    """Human-readable label for a competition target."""
    comp_id = target["competition_id"]
    season_id = target["season_id"]
    return f"Competition {comp_id} / Season {season_id}"


# ---------------------------------------------------------------------------
# Incoming manager profile
# ---------------------------------------------------------------------------


def _build_incoming_profile(name: str) -> ManagerAgent:
    """Build a ManagerAgent stub for the incoming manager.

    Uses neutral defaults for StatsBomb-derived attributes since we
    don't have the incoming manager's historical data. The profile
    is used for tactical estimate display in the radar chart.
    """
    return ManagerAgent(
        manager_name=name,
        team_name="",
        competition_id=0,
        season_id=0,
        tenure_match_ids=frozenset(),
        pressing_intensity=50.0,
        possession_preference=0.5,
        counter_tendency=0.5,
        preferred_formation="4-4-2",
    )


def _build_transfer_trigger(params: SimulationParams) -> TransferInTrigger:
    """Build a TransferInTrigger with a neutral-attribute PlayerAgent.

    The transfer player uses 50th-percentile technical attributes since
    we don't have StatsBomb data for hypothetical signings.
    """
    # Generate a unique player_id unlikely to collide with existing squad.
    player = PlayerAgent(
        player_id=99999,
        player_name=params.transfer_player_name,
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
# Sidebar - Input
# ---------------------------------------------------------------------------


def _render_sidebar() -> SimulationParams | None:
    """Render sidebar inputs and return parameters on button press."""
    st.sidebar.header("Simulation Parameters")

    targets = _load_targets()
    target_labels = [_competition_label(t) for t in targets]
    selected_idx = int(
        st.sidebar.selectbox(
            "Competition / Season",
            range(len(targets)),
            format_func=lambda i: target_labels[i],
        )
        or 0
    )
    target = targets[selected_idx]

    clubs: list[str] = target["clubs"]
    team_name = str(st.sidebar.selectbox("Team", clubs) or clubs[0])

    manager_name: str = st.sidebar.text_input(
        "Current Manager Name",
        placeholder="e.g., Erik ten Hag",
    )

    trigger_week: int = st.sidebar.slider(
        "Trigger Week", min_value=1, max_value=38, value=10
    )

    st.sidebar.divider()
    trigger_type = str(
        st.sidebar.radio(
            "Trigger Type",
            options=["manager_change", "transfer_in"],
            format_func=lambda x: {
                "manager_change": "Manager Change",
                "transfer_in": "Player Transfer (Experimental)",
            }[x],
        )
        or "manager_change"
    )

    # Manager change fields.
    incoming_manager_name = ""
    transfer_player_name = ""
    transfer_expected_role = "starter"

    if trigger_type == "manager_change":
        incoming_manager_name = st.sidebar.text_input(
            "Incoming Manager Name",
            placeholder="e.g., Jose Mourinho",
        )
    else:
        st.sidebar.caption(
            "Experimental: Transfer trigger adds a player with neutral "
            "attributes (50th percentile). Squad composition differs between "
            "branches, which may affect player impact rankings."
        )
        transfer_player_name = st.sidebar.text_input(
            "Player Name",
            placeholder="e.g., Kylian Mbappe",
        )
        transfer_expected_role = str(
            st.sidebar.selectbox(
                "Expected Role",
                options=["starter", "rotation", "squad"],
            )
            or "starter"
        )

    st.sidebar.divider()
    st.sidebar.subheader("Simulation Settings")

    n_runs = int(
        st.sidebar.number_input(
            "Number of Runs", min_value=1, max_value=100, value=_DEFAULT_N_RUNS
        )
    )
    seed = int(
        st.sidebar.number_input(
            "Random Seed", min_value=0, max_value=99999, value=_DEFAULT_SEED
        )
    )

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

    if st.sidebar.button("Run Simulation", type="primary", use_container_width=True):
        if not manager_name.strip():
            st.sidebar.error("Current Manager Name is required.")
            return None
        if trigger_type == "manager_change" and not incoming_manager_name.strip():
            st.sidebar.error("Incoming Manager Name is required.")
            return None
        if trigger_type == "transfer_in" and not transfer_player_name.strip():
            st.sidebar.error("Player Name is required.")
            return None

        return SimulationParams(
            competition_id=int(target["competition_id"]),
            season_id=int(target["season_id"]),
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
# Pipeline execution
# ---------------------------------------------------------------------------


def _run_pipeline(params: SimulationParams) -> None:
    """Execute the full pipeline and display results."""
    progress = st.progress(0, text="Initializing...")

    # 1. Initialize agents from StatsBomb data.
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
        return

    progress.progress(30, text="Running simulation...")

    # 2. Build trigger.
    trigger: ChangeTrigger
    incoming_profile: ManagerAgent | None = None

    if params.trigger_type == "transfer_in":
        trigger = _build_transfer_trigger(params)
    else:
        incoming_profile = _build_incoming_profile(params.incoming_manager_name)
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
        return

    progress.progress(70, text="Generating visualizations...")

    # 4. Compute player impacts.
    impacts = rank_player_impact(comparison, top_n=_DEFAULT_TOP_PLAYERS)

    # 5. Resolve LLM client (if configured).
    llm_client = _get_llm_client()

    # 6. Display results.
    _render_delta_metrics(comparison)
    if params.trigger_type == "manager_change":
        _render_team_radar(comparison, init_result, incoming_profile)
    else:
        st.header("Team Radar Chart")
        st.info(
            "Team radar chart is not applicable for transfer triggers. "
            "Tactical estimates are manager-driven and unchanged by player transfers."
        )
    _render_player_impact(impacts, params)
    _render_report(comparison, params, impacts, llm_client)

    progress.progress(100, text="Complete.")


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


def _render_delta_metrics(comparison: ComparisonResult) -> None:
    """Display delta metrics summary table."""
    st.header("Delta Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Points (mean)",
        f"{comparison.with_change.total_points_mean:.1f}",
        delta=f"{comparison.delta.points_mean_diff:+.1f}",
    )
    col2.metric(
        "Points (median)",
        f"{comparison.with_change.total_points_median:.1f}",
        delta=f"{comparison.delta.points_median_diff:+.1f}",
    )
    col3.metric("Runs", f"{comparison.no_change.n_runs}")

    # Cascade event diff table.
    cascade_diff = comparison.delta.cascade_count_diff
    if cascade_diff:
        st.subheader("Cascade Event Frequency (B - A)")
        st.table(
            {
                "Event Type": list(cascade_diff.keys()),
                "Diff (mean/run)": [f"{v:+.2f}" for v in cascade_diff.values()],
            }
        )


def _render_team_radar(
    comparison: ComparisonResult,
    init_result: InitializationResult,
    incoming_profile: ManagerAgent | None,
) -> None:
    """Display team radar chart."""
    st.header("Team Radar Chart")

    st.caption(
        "Tactical estimate axes (PPDA, Possession, Prog Passes) use neutral "
        "defaults for the incoming manager. StatsBomb data for the incoming "
        "manager is not available in this version."
    )

    radar_data = extract_radar_data(
        comparison,
        init_result.team_baseline,
        incoming_profile,
        init_result.league_context,
    )
    fig = create_radar_figure(radar_data)
    st.pyplot(fig)
    plt.close(fig)


def _render_player_impact(
    impacts: list[PlayerImpact],
    params: SimulationParams,
) -> None:
    """Display player impact radar charts."""
    st.header("Player Impact")

    if params.trigger_type == "transfer_in":
        st.caption(
            "Transfer trigger: the new signing only exists in Branch B, so "
            "they are shown separately below. Existing player rankings reflect "
            "indirect effects of the squad addition."
        )
        # Show new signing info from Branch B.
        _render_transfer_player_info(params)

    if not impacts:
        st.info("No significant player impact detected among existing squad.")
        return

    for impact in impacts:
        st.subheader(f"{impact.player_name} (impact: {impact.impact_score:.3f})")
        fig = create_player_radar_figure(impact)
        st.pyplot(fig)
        plt.close(fig)


def _render_transfer_player_info(params: SimulationParams) -> None:
    """Display a summary card for the transferred player."""
    st.subheader(f"New Signing: {params.transfer_player_name}")
    st.write(
        f"- **Role:** {params.transfer_expected_role}\n"
        f"- **Joined at:** week {params.trigger_week + 1}\n"
        f"- **Attributes:** neutral (50th percentile) — no StatsBomb data\n"
        f"- **Note:** This player only exists in Branch B. "
        f"Their state is not included in the A/B impact ranking above."
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
    st.header("Comparison Report")

    if params.trigger_type == "transfer_in":
        trigger_desc = (
            f"Transfer in: {params.transfer_player_name} "
            f"({params.transfer_expected_role}) at week {params.trigger_week}"
        )
    else:
        trigger_desc = (
            f"Manager change: {params.manager_name} -> "
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
        limitations=list(DEFAULT_LIMITATIONS),
    )

    if llm_client is not None:
        st.caption("Generating report via LLM...")
        try:
            report_md = generate_report(llm_client, report_input)
            st.markdown(report_md)
            return
        except Exception:
            st.warning("LLM report generation failed. Falling back to data report.")

    # Fallback: data-only report.
    _render_data_report(report_input)


def _render_data_report(report_input: ReportInput) -> None:
    """Render a structured report from data without LLM."""
    st.subheader("Summary")
    st.write(
        f"**Trigger:** {report_input.trigger_description}  \n"
        f"**Runs:** {report_input.n_runs}  \n"
        f"**Points impact:** {report_input.points_mean_diff:+.2f} "
        f"(mean, B - A)"
    )

    st.subheader("Key Differences")
    st.write(
        f"- Mean points: Branch A = {report_input.points_mean_a:.1f}, "
        f"Branch B = {report_input.points_mean_b:.1f} "
        f"(diff: {report_input.points_mean_diff:+.1f}) [fact]"
    )
    for event_type, diff in report_input.cascade_count_diff.items():
        st.write(f"- {event_type}: {diff:+.2f} per run [fact]")

    if report_input.player_impacts:
        st.subheader("Player Impact")
        for p in report_input.player_impacts:
            st.write(
                f"- **{p.player_name}** (impact: {p.impact_score:.3f}): "
                f"form {p.form_diff:+.2f}, fatigue {p.fatigue_diff:+.2f}, "
                f"understanding {p.understanding_diff:+.2f}, "
                f"trust {p.trust_diff:+.2f} [fact]"
            )

    st.subheader("Causal Chain")
    if report_input.action_explanations:
        for a in report_input.action_explanations:
            st.write(
                f"- **{a.tp_type} -> {a.action}**: "
                f"{a.explanation} [{a.label}]"
            )
    else:
        st.write(
            "No action explanations available. "
            "LLM-based causal chain analysis requires an LLMClient configuration."
        )

    st.subheader("Limitations")
    for limitation in report_input.limitations:
        st.write(f"- {limitation}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Streamlit app entry point."""
    st.set_page_config(
        page_title="IfFootball",
        page_icon="",
        layout="wide",
    )

    st.title("IfFootball - What-If Simulation")
    st.caption(
        "Compare simulation branches with and without a manager change trigger."
    )

    params = _render_sidebar()

    if params is not None:
        _run_pipeline(params)
    else:
        st.info(
            "Configure parameters in the sidebar and click 'Run Simulation'."
        )


if __name__ == "__main__":
    main()
