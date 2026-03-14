"""League context domain model.

LeagueContext holds league-level information used for cross-league comparison
and as simulation context. Fields are divided into two categories:

  Fact fields (StatsBomb-derived):
      avg_ppda, avg_progressive_passes_per90, avg_xg_per90
      These are computed from StatsBomb event aggregates across all
      competition/season matches.
      Placeholder 0.0 in M1; full derivation is deferred to M2.

  Hypothesis fields (LLM-derived):
      pressing_level, physicality_level, tactical_complexity
      Populated by llm.knowledge_query. Represent LLM assessments, not
      StatsBomb measurements. Callers must treat these as [hypothesis]
      labels, not facts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LeagueContext:
    """Immutable league-level context snapshot.

    Frozen to prevent accidental mutation after construction. To update
    hypothesis fields after a knowledge query, use dataclasses.replace():

        from dataclasses import replace
        ctx = replace(ctx, pressing_level="high", physicality_level="mid")

    Attributes:
        competition_id: StatsBomb competition ID.
        season_id:      StatsBomb season ID.
        name:           Human-readable league name (e.g. "Premier League").

        Fact attributes — StatsBomb-derived (placeholder 0.0 in M1;
        replace with actual StatsBomb aggregation in M2):
            avg_ppda:                        League average PPDA
                                             (lower = more pressing).
            avg_progressive_passes_per90:    League average progressive
                                             passes per 90 min.
            avg_xg_per90:                    League average xG per 90 min.

        Hypothesis attributes — LLM-derived via knowledge query:
            None means "not yet queried". These values are [hypothesis]
            labels and must not be used as facts in simulation logic.
            pressing_level:     Perceived pressing intensity.
            physicality_level:  Perceived physicality.
            tactical_complexity: Perceived tactical sophistication.
    """

    competition_id: int
    season_id: int
    name: str

    # StatsBomb-derived facts — placeholder 0.0 in M1;
    # replace with actual StatsBomb aggregation in M2.
    avg_ppda: float = 0.0
    avg_progressive_passes_per90: float = 0.0
    avg_xg_per90: float = 0.0

    # LLM hypothesis labels — None until populated by knowledge_query.
    # Treat as [hypothesis], not fact.
    pressing_level: str | None = None       # "high" / "mid" / "low"
    physicality_level: str | None = None    # "high" / "mid" / "low"
    tactical_complexity: str | None = None  # "high" / "mid" / "low"
