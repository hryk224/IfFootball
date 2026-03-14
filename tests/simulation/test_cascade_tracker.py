"""Tests for cascade event tracking."""

from __future__ import annotations

import pytest

from iffootball.simulation.cascade_tracker import (
    DEFAULT_DEPTH_LIMIT,
    DEFAULT_IMPORTANCE_THRESHOLD,
    VALID_EVENT_TYPES,
    CascadeEvent,
    CascadeTracker,
)


# ---------------------------------------------------------------------------
# CascadeEvent
# ---------------------------------------------------------------------------


class TestCascadeEvent:
    def test_creation(self) -> None:
        e = CascadeEvent(
            week=5,
            event_type="form_drop",
            affected_agent="Player A",
            cause_chain=("trust_decline",),
            magnitude=0.3,
            depth=2,
        )
        assert e.week == 5
        assert e.event_type == "form_drop"
        assert e.affected_agent == "Player A"
        assert e.cause_chain == ("trust_decline",)
        assert e.magnitude == 0.3
        assert e.depth == 2

    def test_frozen(self) -> None:
        e = CascadeEvent(
            week=1,
            event_type="form_drop",
            affected_agent="P",
            cause_chain=(),
            magnitude=0.5,
            depth=1,
        )
        with pytest.raises(AttributeError):
            e.magnitude = 0.9  # type: ignore[misc]

    def test_cause_chain_immutable(self) -> None:
        e = CascadeEvent(
            week=1,
            event_type="form_drop",
            affected_agent="P",
            cause_chain=("trust_decline",),
            magnitude=0.5,
            depth=2,
        )
        assert isinstance(e.cause_chain, tuple)


# ---------------------------------------------------------------------------
# CascadeTracker.record
# ---------------------------------------------------------------------------


class TestCascadeTrackerRecord:
    def test_records_valid_event(self) -> None:
        tracker = CascadeTracker()
        result = tracker.record(
            week=3,
            event_type="form_drop",
            affected_agent="Player A",
            cause_chain=(),
            magnitude=0.5,
            depth=1,
        )
        assert result is not None
        assert len(tracker.events) == 1
        assert tracker.events[0].event_type == "form_drop"

    def test_depth_at_limit_recorded(self) -> None:
        """depth == depth_limit should be recorded (inclusive)."""
        tracker = CascadeTracker(depth_limit=4)
        result = tracker.record(
            week=1,
            event_type="form_drop",
            affected_agent="P",
            cause_chain=(),
            magnitude=0.5,
            depth=4,
        )
        assert result is not None

    def test_depth_exceeds_limit_filtered(self) -> None:
        """depth > depth_limit should be filtered out."""
        tracker = CascadeTracker(depth_limit=4)
        result = tracker.record(
            week=1,
            event_type="form_drop",
            affected_agent="P",
            cause_chain=(),
            magnitude=0.5,
            depth=5,
        )
        assert result is None
        assert len(tracker.events) == 0

    def test_magnitude_below_threshold_filtered(self) -> None:
        tracker = CascadeTracker(importance_threshold=0.05)
        result = tracker.record(
            week=1,
            event_type="form_drop",
            affected_agent="P",
            cause_chain=(),
            magnitude=0.04,
            depth=1,
        )
        assert result is None
        assert len(tracker.events) == 0

    def test_magnitude_at_threshold_recorded(self) -> None:
        tracker = CascadeTracker(importance_threshold=0.05)
        result = tracker.record(
            week=1,
            event_type="form_drop",
            affected_agent="P",
            cause_chain=(),
            magnitude=0.05,
            depth=1,
        )
        assert result is not None

    def test_invalid_event_type_raises(self) -> None:
        tracker = CascadeTracker()
        with pytest.raises(ValueError, match="Unknown event_type"):
            tracker.record(
                week=1,
                event_type="invalid_type",
                affected_agent="P",
                cause_chain=(),
                magnitude=0.5,
                depth=1,
            )

    def test_all_valid_event_types_accepted(self) -> None:
        tracker = CascadeTracker()
        for et in VALID_EVENT_TYPES:
            result = tracker.record(
                week=1,
                event_type=et,
                affected_agent="P",
                cause_chain=(),
                magnitude=0.5,
                depth=1,
            )
            assert result is not None


# ---------------------------------------------------------------------------
# CascadeTracker.record_chained
# ---------------------------------------------------------------------------


class TestCascadeTrackerRecordChained:
    def test_chain_extends_cause(self) -> None:
        tracker = CascadeTracker()
        parent = tracker.record(
            week=5,
            event_type="trust_decline",
            affected_agent="Player A",
            cause_chain=(),
            magnitude=0.5,
            depth=1,
        )
        assert parent is not None

        child = tracker.record_chained(
            parent=parent,
            event_type="form_drop",
            affected_agent="Player A",
            magnitude=0.3,
        )
        assert child is not None
        assert child.cause_chain == ("trust_decline",)
        assert child.depth == 2
        assert child.week == 5  # inherited from parent

    def test_multi_level_chain(self) -> None:
        tracker = CascadeTracker(depth_limit=4)
        e1 = tracker.record(
            week=3,
            event_type="playing_time_change",
            affected_agent="P",
            cause_chain=(),
            magnitude=0.6,
            depth=1,
        )
        assert e1 is not None

        e2 = tracker.record_chained(e1, "trust_decline", "P", 0.4)
        assert e2 is not None
        assert e2.depth == 2
        assert e2.cause_chain == ("playing_time_change",)

        e3 = tracker.record_chained(e2, "form_drop", "P", 0.2)
        assert e3 is not None
        assert e3.depth == 3
        assert e3.cause_chain == ("playing_time_change", "trust_decline")

        e4 = tracker.record_chained(e3, "tactical_confusion", "P", 0.1)
        assert e4 is not None
        assert e4.depth == 4  # at limit, still recorded

        e5 = tracker.record_chained(e4, "form_drop", "P", 0.1)
        assert e5 is None  # depth 5 > limit 4

    def test_chained_filtered_by_magnitude(self) -> None:
        tracker = CascadeTracker(importance_threshold=0.1)
        parent = tracker.record(
            week=1,
            event_type="trust_decline",
            affected_agent="P",
            cause_chain=(),
            magnitude=0.5,
            depth=1,
        )
        assert parent is not None

        child = tracker.record_chained(parent, "form_drop", "P", 0.05)
        assert child is None


# ---------------------------------------------------------------------------
# CascadeTracker.events property
# ---------------------------------------------------------------------------


class TestCascadeTrackerEvents:
    def test_returns_copy(self) -> None:
        tracker = CascadeTracker()
        tracker.record(
            week=1,
            event_type="form_drop",
            affected_agent="P",
            cause_chain=(),
            magnitude=0.5,
            depth=1,
        )
        events = tracker.events
        events.clear()
        # Internal list should not be affected
        assert len(tracker.events) == 1

    def test_empty_by_default(self) -> None:
        tracker = CascadeTracker()
        assert tracker.events == []

    def test_default_constants(self) -> None:
        assert DEFAULT_DEPTH_LIMIT == 4
        assert DEFAULT_IMPORTANCE_THRESHOLD == 0.05
