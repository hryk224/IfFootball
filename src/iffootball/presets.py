"""Shared demo preset definitions.

Defines the representative scenarios used by the Live Demo UI (app.py)
and the preset preview runner (scripts/preview_presets.py). All presets
target Premier League 2015-16 (competition_id=2, season_id=27).

Note on coupling: DEMO_PRESETS is shared between UI and CLI. Adding or
removing entries affects both. If review-only presets are needed without
changing the UI, pass a custom preset list to the preview runner instead
of modifying this module.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DemoPreset:
    """A predefined manager-change scenario for demo/testing.

    Attributes:
        slug:                   Short identifier for filenames (e.g. "van_gaal_to_mourinho").
        label:                  Human-readable display label.
        team_name:              StatsBomb team name.
        manager_name:           Outgoing manager (StatsBomb name).
        incoming_manager_name:  Incoming manager (StatsBomb name).
        trigger_week:           Match week at which the change is applied.
    """

    slug: str
    label: str
    team_name: str
    manager_name: str
    incoming_manager_name: str
    trigger_week: int


# Premier League 2015-16 (competition_id=2, season_id=27).
COMPETITION_ID = 2
SEASON_ID = 27

DEMO_PRESETS: tuple[DemoPreset, ...] = (
    DemoPreset(
        slug="van_gaal_to_mourinho",
        label="Man United: Van Gaal \u2192 Mourinho",
        team_name="Manchester United",
        manager_name="Louis van Gaal",
        incoming_manager_name="Jos\u00e9 Mario Felix dos Santos Mourinho",
        trigger_week=29,
    ),
    DemoPreset(
        slug="van_gaal_to_klopp",
        label="Man United: Van Gaal \u2192 Klopp",
        team_name="Manchester United",
        manager_name="Louis van Gaal",
        incoming_manager_name="J\u00fcrgen Klopp",
        trigger_week=29,
    ),
    DemoPreset(
        slug="mourinho_to_hiddink",
        label="Chelsea: Mourinho \u2192 Hiddink",
        team_name="Chelsea",
        manager_name="Jos\u00e9 Mario Felix dos Santos Mourinho",
        incoming_manager_name="Guus Hiddink",
        trigger_week=16,
    ),
)
