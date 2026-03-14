"""Tests for change trigger domain models."""

from __future__ import annotations

import dataclasses

import pytest

from iffootball.agents.trigger import (
    ChangeTrigger,
    ManagerChangeTrigger,
    TransferInTrigger,
    TriggerType,
)


# ---------------------------------------------------------------------------
# ManagerChangeTrigger
# ---------------------------------------------------------------------------


class TestManagerChangeTrigger:
    def test_creation(self) -> None:
        t = ManagerChangeTrigger(
            outgoing_manager_name="Old Manager",
            incoming_manager_name="New Manager",
            transition_type="mid_season",
            applied_at=10,
        )
        assert t.outgoing_manager_name == "Old Manager"
        assert t.incoming_manager_name == "New Manager"
        assert t.transition_type == "mid_season"
        assert t.applied_at == 10

    def test_trigger_type_is_fixed(self) -> None:
        t = ManagerChangeTrigger(
            outgoing_manager_name="A",
            incoming_manager_name="B",
            transition_type="mid_season",
            applied_at=5,
        )
        assert t.trigger_type == TriggerType.MANAGER_CHANGE

    def test_trigger_type_not_settable_via_init(self) -> None:
        """trigger_type is init=False; passing it to __init__ raises TypeError."""
        with pytest.raises(TypeError):
            ManagerChangeTrigger(
                outgoing_manager_name="A",
                incoming_manager_name="B",
                transition_type="mid_season",
                applied_at=5,
                trigger_type=TriggerType.PLAYER_TRANSFER_IN,  # type: ignore[call-arg]
            )

    def test_frozen(self) -> None:
        t = ManagerChangeTrigger(
            outgoing_manager_name="A",
            incoming_manager_name="B",
            transition_type="mid_season",
            applied_at=5,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            t.applied_at = 99  # type: ignore[misc]

    def test_pre_season_transition(self) -> None:
        t = ManagerChangeTrigger(
            outgoing_manager_name="A",
            incoming_manager_name="B",
            transition_type="pre_season",
            applied_at=0,
        )
        assert t.transition_type == "pre_season"
        assert t.applied_at == 0


# ---------------------------------------------------------------------------
# TransferInTrigger
# ---------------------------------------------------------------------------


class TestTransferInTrigger:
    def test_creation(self) -> None:
        t = TransferInTrigger(
            player_name="New Player",
            expected_role="starter",
            applied_at=15,
        )
        assert t.player_name == "New Player"
        assert t.expected_role == "starter"
        assert t.applied_at == 15

    def test_trigger_type_is_fixed(self) -> None:
        t = TransferInTrigger(
            player_name="P",
            expected_role="rotation",
            applied_at=3,
        )
        assert t.trigger_type == TriggerType.PLAYER_TRANSFER_IN

    def test_trigger_type_not_settable_via_init(self) -> None:
        with pytest.raises(TypeError):
            TransferInTrigger(
                player_name="P",
                expected_role="starter",
                applied_at=3,
                trigger_type=TriggerType.MANAGER_CHANGE,  # type: ignore[call-arg]
            )

    def test_frozen(self) -> None:
        t = TransferInTrigger(
            player_name="P",
            expected_role="squad",
            applied_at=3,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            t.player_name = "Other"  # type: ignore[misc]

    def test_all_expected_roles(self) -> None:
        for role in ("starter", "rotation", "squad"):
            t = TransferInTrigger(
                player_name="P",
                expected_role=role,
                applied_at=1,
            )
            assert t.expected_role == role


# ---------------------------------------------------------------------------
# TriggerType enum
# ---------------------------------------------------------------------------


class TestTriggerType:
    def test_values(self) -> None:
        assert TriggerType.MANAGER_CHANGE.value == "manager_change"
        assert TriggerType.PLAYER_TRANSFER_IN.value == "player_transfer_in"

    def test_is_str_enum(self) -> None:
        assert isinstance(TriggerType.MANAGER_CHANGE, str)


# ---------------------------------------------------------------------------
# isinstance discrimination
# ---------------------------------------------------------------------------


class TestChangeTriggerDiscrimination:
    def test_isinstance_manager_change(self) -> None:
        t: ChangeTrigger = ManagerChangeTrigger(
            outgoing_manager_name="A",
            incoming_manager_name="B",
            transition_type="mid_season",
            applied_at=10,
        )
        assert isinstance(t, ManagerChangeTrigger)
        assert not isinstance(t, TransferInTrigger)

    def test_isinstance_transfer_in(self) -> None:
        t: ChangeTrigger = TransferInTrigger(
            player_name="P",
            expected_role="starter",
            applied_at=5,
        )
        assert isinstance(t, TransferInTrigger)
        assert not isinstance(t, ManagerChangeTrigger)
