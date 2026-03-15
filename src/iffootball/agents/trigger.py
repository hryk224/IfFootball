"""Change trigger domain models.

Triggers represent external changes injected into the simulation at a
specific match week. The weekly simulation loop uses these to determine
when and what change occurred.

Timing convention:
    applied_at = N means the trigger is injected after match week N has
    completed. Effects begin from match week N+1 onwards.

    Example: ManagerChangeTrigger(applied_at=10) means the new manager
    takes charge starting from match week 11.

Trigger type discrimination:
    Each trigger class carries a fixed trigger_type field (init=False)
    for serialisation scenarios. For runtime branching, use isinstance()
    or match statements on the concrete class.

    ChangeTrigger is a Union type alias for use in function signatures
    that accept any trigger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from iffootball.agents.manager import ManagerAgent
    from iffootball.agents.player import PlayerAgent


class TriggerType(str, Enum):
    """Discriminator for change trigger variants."""

    MANAGER_CHANGE = "manager_change"
    PLAYER_TRANSFER_IN = "player_transfer_in"


@dataclass(frozen=True)
class ManagerChangeTrigger:
    """Trigger for a managerial change.

    Represents the dismissal of one manager and appointment of another.
    The transition takes effect after the match week specified by
    applied_at: the outgoing manager's last match is week N, and the
    incoming manager's first match is week N+1.

    Attributes:
        outgoing_manager_name: Name of the departing manager
            (StatsBomb spelling).
        incoming_manager_name: Name of the incoming manager
            (StatsBomb spelling).
        transition_type: Context of the change.
            "mid_season" — replacement during the season.
            "pre_season" — appointment before the season starts.
        applied_at: Match week after which the change takes effect.
            Effects start from applied_at + 1.
        incoming_profile: Optional ManagerAgent whose static tactical
            attributes (pressing_intensity, possession_preference, etc.)
            are copied to the active manager on trigger execution.
            When None, neutral defaults are used instead.
            Note: manager_name is always taken from incoming_manager_name,
            not from the profile. Dynamic state (job_security, squad_trust)
            is always reset regardless of the profile.
        trigger_type: Fixed to MANAGER_CHANGE. Not settable via __init__.
    """

    outgoing_manager_name: str
    incoming_manager_name: str
    transition_type: str  # "mid_season" | "pre_season"
    applied_at: int
    incoming_profile: ManagerAgent | None = None

    trigger_type: TriggerType = field(
        default=TriggerType.MANAGER_CHANGE,
        init=False,
    )


@dataclass(frozen=True)
class TransferInTrigger:
    """Trigger for an incoming player transfer.

    Represents a new player joining the squad. The player becomes
    available for selection starting from the match week after applied_at.

    Attributes:
        player_name: Name of the incoming player.
        expected_role: Anticipated squad role for the new player.
            "starter"  — expected to start most matches.
            "rotation" — expected to rotate with existing starters.
            "squad"    — depth signing, available as backup.
        applied_at: Match week after which the player is available.
            Player can be selected from applied_at + 1.
        player: Optional PlayerAgent for the incoming player. When
            provided, the player is added to the squad on trigger
            execution. When None, the trigger is queued but execution
            raises ValueError (player must be resolved before simulation).
        trigger_type: Fixed to PLAYER_TRANSFER_IN. Not settable via __init__.
    """

    player_name: str
    expected_role: str  # "starter" | "rotation" | "squad"
    applied_at: int
    player: PlayerAgent | None = None

    trigger_type: TriggerType = field(
        default=TriggerType.PLAYER_TRANSFER_IN,
        init=False,
    )


# Union type for function signatures that accept any trigger.
ChangeTrigger = Union[ManagerChangeTrigger, TransferInTrigger]
