"""Weekly simulation engine.

Orchestrates the 10-step weekly loop over remaining fixtures:
  1. Trigger application
  2. Lineup selection
  3. Match result (Poisson)
  4. Fatigue update
  5. Tactical understanding update
  6. Manager trust update
  7. Job security update
  8. Turning point detection
  9. TP response → state update
  10. CascadeEvent recording

Branch A/B independence:
    Each Simulation instance holds its own mutable squad/manager state.
    Callers must deepcopy inputs before constructing separate instances
    to ensure branches do not share state.

RNG split (paired_split_v1):
    Simulation accepts two separate generators: match_rng for Poisson
    match result sampling (step 3) and action_rng for turning-point
    action sampling (step 9). This separation ensures that divergent
    TP activity in one branch does not shift the match result sequence
    in a paired A/B comparison.

    PAIRED CONTRACT: simulate_match() consumes exactly 2 RNG calls
    (goals_for, goals_against) in fixed order per fixture. Any change
    to this consumption pattern breaks paired comparison guarantees.

Trigger timing:
    A trigger with applied_at=N takes effect at week N+1.
    _apply_pending_triggers(week) checks trigger.applied_at + 1 == week,
    applies the trigger, and removes it from the queue.

TransferInTrigger:
    Adds the incoming player to the squad with role-appropriate initial
    trust and low tactical understanding. Requires trigger.player to be
    a resolved PlayerAgent (raises ValueError if None).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from iffootball.agents.fixture import FixtureList, OpponentStrength
from iffootball.agents.manager import ManagerAgent
from iffootball.agents.player import PlayerAgent
from iffootball.agents.trigger import (
    ChangeTrigger,
    ManagerChangeTrigger,
    TransferInTrigger,
)
from iffootball.agents.team import TeamBaseline
from iffootball.config import SimulationRules
from iffootball.simulation.cascade_tracker import CascadeEvent, CascadeTracker
from iffootball.simulation.lineup_selection import select_lineup
from iffootball.simulation.match_result import MatchResult, simulate_match
from iffootball.simulation.state_update import (
    update_current_form,
    update_fatigue,
    update_job_security,
    update_manager_trust,
    update_tactical_understanding,
)
from iffootball.simulation.turning_point import (
    ActionDistribution,
    SimContext,
    TurningPointHandler,
    detect_manager_turning_points,
    detect_player_turning_points,
)

# Rolling window size for job_security calculation.
_JOB_SECURITY_WINDOW = 5


@dataclass(frozen=True)
class SimulationResult:
    """Complete result of a simulation run.

    Attributes:
        match_results:   Ordered list of match results per fixture.
        cascade_events:  All recorded cascade events.
        final_squad:     Squad state at end of simulation.
        final_manager:   Manager state at end of simulation.
    """

    match_results: list[MatchResult]
    cascade_events: list[CascadeEvent]
    final_squad: list[PlayerAgent]
    final_manager: ManagerAgent


class Simulation:
    """Weekly simulation engine.

    Processes each fixture in order, applying the 10-step weekly update
    loop. Triggers are queued via apply_trigger() and take effect at
    trigger.applied_at + 1.

    Args:
        team:               Team baseline metrics.
        squad:              Mutable list of PlayerAgent (caller must
                            deepcopy if sharing across branches).
        manager:            Mutable ManagerAgent.
        fixture_list:       Remaining fixtures to simulate.
        opponent_strengths: Pre-computed opponent strength snapshots.
        rules:              Simulation rules config.
        handler:            Turning point response handler.
        match_rng:          Seeded generator for match result Poisson
                            draws. In paired comparison, A/B share the
                            same seed so that Poisson noise cancels out.
        action_rng:         Seeded generator for TP action sampling.
                            Independent between A/B so that divergent TP
                            activity does not shift match_rng sequences.
    """

    def __init__(
        self,
        team: TeamBaseline,
        squad: list[PlayerAgent],
        manager: ManagerAgent,
        fixture_list: FixtureList,
        opponent_strengths: dict[str, OpponentStrength],
        rules: SimulationRules,
        handler: TurningPointHandler,
        match_rng: np.random.Generator,
        action_rng: np.random.Generator,
    ) -> None:
        self._team = team
        self._squad = squad
        self._manager = manager
        self._fixture_list = fixture_list
        self._opponent_strengths = opponent_strengths
        self._rules = rules
        self._handler = handler
        self._match_rng = match_rng
        self._action_rng = action_rng

        self._triggers: list[ChangeTrigger] = []
        self._recent_points: list[int] = []
        self._matches_since_appointment: int | None = None
        # Baseline pressing for manager profile → match result adjustment.
        # Captured at init so Branch A/B delta is relative to the starting manager.
        self._baseline_pressing: float = manager.pressing_intensity

    def apply_trigger(self, trigger: ChangeTrigger) -> None:
        """Queue a trigger for application during run().

        The trigger takes effect at week trigger.applied_at + 1.
        Both ManagerChangeTrigger and TransferInTrigger are supported.
        """
        self._triggers.append(trigger)

    def run(self) -> SimulationResult:
        """Execute the simulation over all remaining fixtures.

        Returns:
            SimulationResult with match results, cascade events,
            and final squad/manager state.
        """
        tracker = CascadeTracker()
        match_results: list[MatchResult] = []

        for fixture in self._fixture_list.fixtures:
            week = fixture.match_week

            # 1. Trigger application
            self._apply_pending_triggers(week)

            # 2. Lineup selection
            lineup = select_lineup(
                self._squad,
                self._manager,
                self._rules,
                self._matches_since_appointment,
            )
            starter_ids = {p.player_id for p in lineup.starters}

            # 3. Match result
            opponent = self._opponent_strengths[fixture.opponent_name]
            result = simulate_match(
                self._team,
                opponent,
                lineup.starters,
                fixture,
                self._rules.adaptation,
                self._rules.match,
                self._match_rng,
                manager_pressing=self._manager.pressing_intensity,
                baseline_pressing=self._baseline_pressing,
            )
            match_results.append(result)

            # 4. Fatigue update
            update_fatigue(self._squad, starter_ids, self._rules)

            # 4b. Current form update (team result momentum)
            update_current_form(
                self._squad, starter_ids, result.points_earned, self._rules
            )

            # 5. Tactical understanding update
            update_tactical_understanding(
                self._squad, self._manager, self._rules
            )

            # 6. Manager trust update
            update_manager_trust(self._squad, starter_ids, self._rules)

            # 7. Job security update
            self._recent_points.append(result.points_earned)
            recent = self._recent_points[-_JOB_SECURITY_WINDOW:]
            update_job_security(self._manager, recent)

            # 8-9. Turning point detection + response
            context = SimContext(
                current_week=week,
                matches_since_appointment=self._matches_since_appointment,
                manager=self._manager,
                recent_points=tuple(recent),
            )
            self._process_player_turning_points(context, tracker, week, starter_ids)
            self._process_manager_turning_points(tracker, week)

            # Advance matches_since_appointment
            if self._matches_since_appointment is not None:
                self._matches_since_appointment += 1

        return SimulationResult(
            match_results=match_results,
            cascade_events=tracker.events,
            final_squad=copy.deepcopy(self._squad),
            final_manager=copy.deepcopy(self._manager),
        )

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _apply_pending_triggers(self, week: int) -> None:
        """Apply triggers whose applied_at + 1 == week, then remove them.

        Trigger timing: a trigger with applied_at=N takes effect at
        week N+1. This method is called at the start of each week
        before lineup selection.
        """
        remaining: list[ChangeTrigger] = []
        for trigger in self._triggers:
            if trigger.applied_at + 1 == week:
                self._execute_trigger(trigger)
            else:
                remaining.append(trigger)
        self._triggers = remaining

    # Initial manager_trust by expected_role for transfer signings.
    _TRANSFER_TRUST: dict[str, float] = {
        "starter": 0.7,
        "rotation": 0.5,
        "squad": 0.3,
    }

    def _execute_trigger(self, trigger: ChangeTrigger) -> None:
        """Execute a single trigger's immediate effects.

        ManagerChangeTrigger:
            Replaces manager identity and sets tactical attributes from
            the incoming_profile if provided, otherwise falls back to
            neutral defaults.

            Static profile attributes copied from incoming_profile:
                pressing_intensity, possession_preference, counter_tendency,
                preferred_formation, implementation_speed, youth_development,
                style_stubbornness.

            Always reset regardless of profile:
                manager_name (from trigger.incoming_manager_name),
                job_security (1.0), squad_trust ({}).
            Also resets squad tactical_understanding and manager_trust.

        TransferInTrigger:
            Adds the incoming player to the squad with role-appropriate
            initial trust and low tactical understanding. Raises ValueError
            if player is None or player_id already exists in squad.

            Note: A/B comparison with TransferInTrigger produces unequal
            squad compositions between branches. Downstream consumers
            (e.g. player_impact ranking) that assume equal squad sizes
            may need adjustment.
        """
        if isinstance(trigger, ManagerChangeTrigger):
            # Replace manager identity (trigger name is authoritative).
            self._manager.manager_name = trigger.incoming_manager_name
            self._matches_since_appointment = 0

            profile = trigger.incoming_profile
            if profile is not None:
                # Copy static tactical attributes from provided profile.
                self._manager.pressing_intensity = profile.pressing_intensity
                self._manager.possession_preference = profile.possession_preference
                self._manager.counter_tendency = profile.counter_tendency
                self._manager.preferred_formation = profile.preferred_formation
                self._manager.implementation_speed = profile.implementation_speed
                self._manager.youth_development = profile.youth_development
                self._manager.style_stubbornness = profile.style_stubbornness
            else:
                # No profile provided — neutral defaults.
                self._manager.pressing_intensity = 50.0
                self._manager.possession_preference = 0.5
                self._manager.counter_tendency = 0.5
                self._manager.preferred_formation = "4-4-2"
                self._manager.implementation_speed = 50.0
                self._manager.youth_development = 50.0
                self._manager.style_stubbornness = 50.0

            # Dynamic state always reset on new appointment.
            self._manager.job_security = 1.0
            self._manager.squad_trust = {}

            # Reset squad trust and tactical understanding for new manager.
            for p in self._squad:
                p.tactical_understanding = 0.25  # low starting point
                p.manager_trust = 0.5  # neutral

        elif isinstance(trigger, TransferInTrigger):
            if trigger.player is None:
                raise ValueError(
                    f"TransferInTrigger for '{trigger.player_name}' has no "
                    f"player payload. Resolve the PlayerAgent before simulation."
                )

            # Check for duplicate player_id.
            existing_ids = {p.player_id for p in self._squad}
            if trigger.player.player_id in existing_ids:
                raise ValueError(
                    f"Player ID {trigger.player.player_id} already in squad."
                )

            # Copy to avoid mutating the trigger's payload (branch isolation).
            player = copy.deepcopy(trigger.player)

            # Set dynamic state for new signing.
            player.tactical_understanding = 0.25  # new to the team
            player.manager_trust = self._TRANSFER_TRUST.get(
                trigger.expected_role, 0.5
            )
            player.bench_streak = 0
            player.fatigue = 0.0  # fresh arrival

            self._squad.append(player)

    def _process_player_turning_points(
        self,
        context: SimContext,
        tracker: CascadeTracker,
        week: int,
        starter_ids: set[int],
    ) -> None:
        """Detect player TPs, sample actions, apply effects, record events."""
        resist_count = 0

        for player in self._squad:
            tps = detect_player_turning_points(
                player, context, self._rules,
                is_starter=player.player_id in starter_ids,
            )
            if not tps:
                continue

            is_starter = player.player_id in starter_ids
            dist = self._handler.handle(player, context, is_starter=is_starter)
            action = self._sample_action(dist)

            # Record cascade events and apply state effects.
            if action == "resist":
                resist_count += 1
                # Form drops (config-driven) and trust declines.
                form_penalty = self._rules.adaptation.form_drop_on_resist
                player.current_form = max(0.0, player.current_form - form_penalty)
                player.manager_trust = max(0.0, player.manager_trust - 0.03)
                form_event = tracker.record(
                    week=week,
                    event_type="form_drop",
                    affected_agent=player.player_name,
                    cause_chain=tuple(tps),
                    magnitude=round(form_penalty, 4),
                    depth=1,
                )
                if form_event is not None:
                    tracker.record_chained(
                        parent=form_event,
                        event_type="trust_decline",
                        affected_agent=player.player_name,
                        magnitude=0.2,
                    )
            elif action == "transfer":
                # Playing time change signal.
                tracker.record(
                    week=week,
                    event_type="playing_time_change",
                    affected_agent=player.player_name,
                    cause_chain=tuple(tps),
                    magnitude=0.4,
                    depth=1,
                )
            elif action == "adapt":
                # Record positive adaptation progress.
                tracker.record(
                    week=week,
                    event_type="adaptation_progress",
                    affected_agent=player.player_name,
                    cause_chain=tuple(tps),
                    magnitude=0.15,
                    depth=1,
                )
                # Confusion alongside adaptation when understanding is low.
                if "low_understanding" in tps:
                    tracker.record(
                        week=week,
                        event_type="tactical_confusion",
                        affected_agent=player.player_name,
                        cause_chain=tuple(tps),
                        magnitude=0.2,
                        depth=1,
                    )

        # Squad-level: unrest when multiple players resist in the same week.
        if resist_count >= 2:
            tracker.record(
                week=week,
                event_type="squad_unrest",
                affected_agent=self._manager.manager_name,
                cause_chain=("multiple_resist",),
                magnitude=0.5,
                depth=1,
            )

    def _process_manager_turning_points(
        self,
        tracker: CascadeTracker,
        week: int,
    ) -> None:
        """Detect manager TPs and record events."""
        mgr_tps = detect_manager_turning_points(
            self._manager, self._rules
        )
        for tp in mgr_tps:
            if tp == "job_security_warning":
                tracker.record(
                    week=week,
                    event_type="manager_tactical_shift",
                    affected_agent=self._manager.manager_name,
                    cause_chain=("job_security_warning",),
                    magnitude=0.5,
                    depth=1,
                )
            elif tp == "job_security_critical":
                tracker.record(
                    week=week,
                    event_type="manager_dismissal",
                    affected_agent=self._manager.manager_name,
                    cause_chain=("job_security_critical",),
                    magnitude=0.8,
                    depth=1,
                )

    def _sample_action(self, dist: ActionDistribution) -> str:
        """Sample a single action from an ActionDistribution."""
        actions = list(dist.choices.keys())
        probs = [dist.choices[a] for a in actions]
        return str(self._action_rng.choice(actions, p=probs))
