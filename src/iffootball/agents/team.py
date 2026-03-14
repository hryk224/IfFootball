"""Team baseline domain model.

TeamBaseline holds a snapshot of team state derived from StatsBomb data,
used as the starting point for simulation and as radar chart baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TeamBaseline:
    """Snapshot of a team's state at a given point in the season.

    Attributes:
        team_name: StatsBomb team name string used as the M1 identifier.
            Must exactly match the StatsBomb spelling (e.g., "Manchester United").
            Spelling variations will be treated as different teams.
        competition_id: StatsBomb competition ID.
        season_id: StatsBomb season ID.
        played_match_ids: Frozenset of match IDs included in this snapshot.
            All metrics are aggregated over exactly this set of matches.
            Caller must ensure all IDs belong to the same competition/season.

        StatsBomb metrics (aggregated over played_match_ids):
            xg_for_per90:           attacking quality (xG / 90 min)
            xg_against_per90:       defensive quality (xGA / 90 min)
            ppda:                   pressing intensity — lower = more pressing.
                                    M1 definition: opponent passes / own defensive actions.
            progressive_passes_per90: forward ball progression per 90 min.
                                    M1 definition: completed passes advancing >= 10
                                    pitch x-coordinate units (StatsBomb range 0–120).
            possession_pct:         share of Pass + Carry events belonging to this team (0–1).

        League standing at played_match_ids snapshot:
            league_position:  1 = top of table.
            points_to_safety: team_points − max(bottom-3 teams' points).
                              Positive = above relegation zone, negative = in relegation zone.
            points_to_title:  team_points − leader_points. Always <= 0.
            matches_remaining: total season matches per team − matches already played.

        Simulation state:
            cultural_inertia: placeholder 0.5 for M1.
                Updated by manager-agent-initialization once manager tenure is known.
                Represents how entrenched the current playing style is (0–1).
    """

    team_name: str
    competition_id: int
    season_id: int
    played_match_ids: frozenset[int]

    # StatsBomb metrics
    xg_for_per90: float
    xg_against_per90: float
    ppda: float
    progressive_passes_per90: float
    possession_pct: float

    # League standing
    league_position: int
    points_to_safety: int
    points_to_title: int
    matches_remaining: int

    # Simulation state
    # placeholder: updated by manager-agent-initialization
    cultural_inertia: float = field(default=0.5)
