"""Manager agent domain model.

ManagerAgent holds a snapshot of a manager's tactical profile derived from
StatsBomb matches and events data for the tenure period.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ManagerAgent:
    """Snapshot of a manager's tactical profile for a given tenure.

    Attributes:
        manager_name: Manager name as recorded in StatsBomb matches
            (home_managers / away_managers). Comma-separated entries are
            split upstream; this field holds a single name.
        team_name: StatsBomb team name (exact spelling required).
        competition_id: StatsBomb competition ID.
        season_id: StatsBomb season ID.
        tenure_match_ids: Frozenset of match IDs for this manager's tenure
            within the competition/season. Derived from matches where
            team_name appears with this manager_name.

        StatsBomb-derived tactical attributes:
            pressing_intensity:    Average Pressure events per 90 min
                                   over tenure matches (raw float).
            possession_preference: Average team (Pass + Carry) / all
                                   (Pass + Carry) ratio (0.0–1.0).
            counter_tendency:      1.0 - possession_preference.
            preferred_formation:   Most frequent starting-XI formation
                                   as "{DF}-{MF}-{FW}" string derived
                                   from lineup BroadPosition counts.
                                   None if tenure has no matches or
                                   starting XI could not be determined.

        Provisional attributes (M1 — not yet derived from data):
            implementation_speed: Matches until tactical metrics stabilise
                                   after appointment. Provisional 50.0.
            youth_development:     Young-player opportunity change under
                                   this manager. Provisional 50.0
                                   (age data unavailable in Open Data).
            style_stubbornness:    Tactical rigidity (LLM-derived,
                                   filled in by llm-knowledge-query).
                                   Provisional 50.0.

        Dynamic simulation state (initialised to defaults):
            job_security: Manager's tenure security (0.0–1.0).
                          High = secure; low = facing dismissal.
            squad_trust:  Per-player trust mapping (player_name → float).
                          Initialised empty; populated by simulation.
    """

    manager_name: str
    team_name: str
    competition_id: int
    season_id: int
    tenure_match_ids: frozenset[int]

    # StatsBomb-derived
    pressing_intensity: float
    possession_preference: float
    counter_tendency: float
    preferred_formation: str | None

    # Provisional values (M1)
    implementation_speed: float = field(default=50.0)
    youth_development: float = field(default=50.0)
    style_stubbornness: float = field(default=50.0)

    # Dynamic simulation state
    job_security: float = field(default=1.0)
    squad_trust: dict[str, float] = field(default_factory=dict)
