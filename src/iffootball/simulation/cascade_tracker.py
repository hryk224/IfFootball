"""Cascade event tracking for simulation cause-effect chains.

Records state changes as CascadeEvent instances, forming traceable
cause chains. Filters by depth limit and importance threshold to
prevent chain bloat.

Depth convention:
    depth 1 = direct effect of a trigger
    depth 2 = secondary effect caused by a depth-1 event
    ...
    depth N = Nth-order effect

    Events with depth > depth_limit are not recorded.
"""

from __future__ import annotations

from dataclasses import dataclass

# Event type vocabulary. Extend as new event types are needed.
VALID_EVENT_TYPES: frozenset[str] = frozenset(
    {
        # Player negative events
        "form_drop",
        "playing_time_change",
        "trust_decline",
        "tactical_confusion",
        # Player positive events
        "adaptation_progress",
        # Squad-level events
        "squad_unrest",
        # Manager events
        "manager_tactical_shift",
        "manager_dismissal",
    }
)

# Default limits (from task specification).
DEFAULT_DEPTH_LIMIT = 4
DEFAULT_IMPORTANCE_THRESHOLD = 0.05


@dataclass(frozen=True)
class CascadeEvent:
    """A single recorded state change in the simulation.

    Frozen to prevent mutation after recording.

    Attributes:
        week:           Match week when the event occurred.
        event_type:     Type of state change (must be in VALID_EVENT_TYPES).
        affected_agent: Name of the affected player or manager.
        cause_chain:    Ordered tuple of event_types that led to this event.
                        Empty for root events (direct trigger effects).
        magnitude:      Impact size (0.0-1.0). Higher = more significant.
        depth:          Chain depth (1 = direct, 2 = secondary, ...).
    """

    week: int
    event_type: str
    affected_agent: str
    cause_chain: tuple[str, ...]
    magnitude: float
    depth: int


class CascadeTracker:
    """Tracks cascade events with depth and importance filtering.

    Events exceeding the depth limit or below the importance threshold
    are silently dropped. Use record() for root events and
    record_chained() to extend an existing chain.

    Args:
        depth_limit:          Maximum allowed depth (inclusive).
                              Events with depth > depth_limit are dropped.
        importance_threshold: Minimum magnitude to record.
                              Events with magnitude < threshold are dropped.
    """

    def __init__(
        self,
        depth_limit: int = DEFAULT_DEPTH_LIMIT,
        importance_threshold: float = DEFAULT_IMPORTANCE_THRESHOLD,
    ) -> None:
        self._depth_limit = depth_limit
        self._importance_threshold = importance_threshold
        self._events: list[CascadeEvent] = []

    def record(
        self,
        week: int,
        event_type: str,
        affected_agent: str,
        cause_chain: tuple[str, ...],
        magnitude: float,
        depth: int,
    ) -> CascadeEvent | None:
        """Record a cascade event if it passes depth and importance filters.

        Args:
            week:           Match week.
            event_type:     Must be in VALID_EVENT_TYPES.
            affected_agent: Name of affected player or manager.
            cause_chain:    Cause trace leading to this event.
            magnitude:      Impact size (0.0-1.0).
            depth:          Chain depth.

        Returns:
            The recorded CascadeEvent, or None if filtered out.

        Raises:
            ValueError: if event_type is not in VALID_EVENT_TYPES.
        """
        if event_type not in VALID_EVENT_TYPES:
            raise ValueError(
                f"Unknown event_type: {event_type!r}. "
                f"Valid types: {sorted(VALID_EVENT_TYPES)}"
            )

        if depth > self._depth_limit:
            return None

        if magnitude < self._importance_threshold:
            return None

        event = CascadeEvent(
            week=week,
            event_type=event_type,
            affected_agent=affected_agent,
            cause_chain=cause_chain,
            magnitude=magnitude,
            depth=depth,
        )
        self._events.append(event)
        return event

    def record_chained(
        self,
        parent: CascadeEvent,
        event_type: str,
        affected_agent: str,
        magnitude: float,
    ) -> CascadeEvent | None:
        """Record a chained event derived from a parent event.

        Automatically extends the cause chain and increments depth.
        Inherits week from the parent event (chained effects occur in
        the same match week as their cause).

        Args:
            parent:         The parent event that caused this one.
            event_type:     Type of the new event.
            affected_agent: Name of affected player or manager.
            magnitude:      Impact size (0.0-1.0).

        Returns:
            The recorded CascadeEvent, or None if filtered out.
        """
        return self.record(
            week=parent.week,
            event_type=event_type,
            affected_agent=affected_agent,
            cause_chain=parent.cause_chain + (parent.event_type,),
            magnitude=magnitude,
            depth=parent.depth + 1,
        )

    @property
    def events(self) -> list[CascadeEvent]:
        """Return a copy of all recorded events."""
        return list(self._events)
