"""Tests for LLM report generation."""

from __future__ import annotations

import json

from iffootball.llm.report_generation import (
    DEFAULT_LIMITATIONS,
    REQUIRED_SECTIONS,
    ActionExplanationEntry,
    PlayerImpactEntry,
    ReportInput,
    _build_payload,
    _FALLBACK_REPORT,
    generate_report,
)


# ---------------------------------------------------------------------------
# Fake LLMClient
# ---------------------------------------------------------------------------


class FakeLLMClient:
    """LLMClient stub that returns a fixed response string."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.last_messages: list[dict[str, str]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.last_messages = messages
        return self._response


_FAKE_PROMPT = "You are a football simulation analyst."

_SAMPLE_REPORT = """## Summary

The manager change resulted in a slight points decrease. [data]

## Key Differences

- Mean points: -1.20 [data]
- form_drop events increased by 0.8 per run [data]

## Causal Chain

A bench_streak turning point caused the player to resist. [analysis]

## Player Impact

Player 7 showed the largest state change with form dropping by 0.15. [data]

## Limitations

- Match outcomes use a Poisson model.
- Tactical metrics are estimates.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_report_input() -> ReportInput:
    return ReportInput(
        trigger_description="Manager change: Original Manager → New Manager at week 10",
        points_mean_a=6.0,
        points_mean_b=4.8,
        points_mean_diff=-1.2,
        cascade_count_diff={
            "form_drop": 0.8,
            "trust_decline": 0.4,
        },
        n_runs=10,
        player_impacts=[
            PlayerImpactEntry(
                player_name="Player 7",
                impact_score=0.42,
                form_diff=-0.15,
                fatigue_diff=0.05,
                understanding_diff=-0.25,
                trust_diff=-0.10,
            ),
            PlayerImpactEntry(
                player_name="Player 10",
                impact_score=0.25,
                form_diff=-0.05,
                fatigue_diff=0.10,
                understanding_diff=-0.20,
                trust_diff=-0.05,
            ),
            PlayerImpactEntry(
                player_name="Player 3",
                impact_score=0.18,
                form_diff=-0.03,
                fatigue_diff=0.02,
                understanding_diff=-0.22,
                trust_diff=-0.02,
            ),
        ],
        action_explanations=[
            ActionExplanationEntry(
                tp_type="bench_streak",
                action="resist",
                explanation="Low trust after 3 benchings caused resistance.",
                label="analysis",
                confidence_note="Based on rule-based model",
            ),
            ActionExplanationEntry(
                tp_type="low_understanding",
                action="adapt",
                explanation="Player adapted to new tactical system despite confusion.",
                label="analysis",
                confidence_note="",
            ),
        ],
        limitations=list(DEFAULT_LIMITATIONS["en"]),
    )


# ---------------------------------------------------------------------------
# Tests: _build_payload
# ---------------------------------------------------------------------------


class TestBuildPayload:
    def test_contains_required_keys(self) -> None:
        payload = _build_payload(_make_report_input())
        expected_keys = {
            "trigger_description",
            "points_mean_a",
            "points_mean_b",
            "points_mean_diff",
            "cascade_count_diff",
            "n_runs",
            "player_impacts",
            "action_explanations",
            "limitations",
        }
        assert set(payload.keys()) == expected_keys

    def test_player_impacts_structure(self) -> None:
        payload = _build_payload(_make_report_input())
        players = payload["player_impacts"]
        assert len(players) == 3
        for p in players:
            assert "player_name" in p
            assert "impact_score" in p
            assert "form_diff" in p
            assert "understanding_diff" in p

    def test_action_explanations_have_tp_type_and_action(self) -> None:
        payload = _build_payload(_make_report_input())
        explanations = payload["action_explanations"]
        assert len(explanations) == 2
        for e in explanations:
            assert "tp_type" in e
            assert "action" in e
            assert "explanation" in e
            assert "label" in e

    def test_serializable_to_json(self) -> None:
        payload = _build_payload(_make_report_input())
        serialized = json.dumps(payload, ensure_ascii=False)
        assert isinstance(serialized, str)

    def test_values_rounded(self) -> None:
        payload = _build_payload(_make_report_input())
        assert payload["points_mean_diff"] == -1.2
        assert payload["cascade_count_diff"]["form_drop"] == 0.8


# ---------------------------------------------------------------------------
# Tests: generate_report
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_returns_report_string(self) -> None:
        client = FakeLLMClient(_SAMPLE_REPORT)
        result = generate_report(
            client,
            _make_report_input(),
            system_prompt=_FAKE_PROMPT,
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_system_user_role_separation(self) -> None:
        client = FakeLLMClient(_SAMPLE_REPORT)
        generate_report(
            client,
            _make_report_input(),
            system_prompt=_FAKE_PROMPT,
        )
        messages = client.last_messages
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_user_message_is_valid_json(self) -> None:
        client = FakeLLMClient(_SAMPLE_REPORT)
        generate_report(
            client,
            _make_report_input(),
            system_prompt=_FAKE_PROMPT,
        )
        user_content = client.last_messages[1]["content"]
        parsed = json.loads(user_content)
        assert isinstance(parsed, dict)

    def test_trigger_description_in_user_message(self) -> None:
        client = FakeLLMClient(_SAMPLE_REPORT)
        generate_report(
            client,
            _make_report_input(),
            system_prompt=_FAKE_PROMPT,
        )
        user_content = client.last_messages[1]["content"]
        assert "Original Manager" in user_content

    def test_trigger_description_not_in_system_prompt(self) -> None:
        client = FakeLLMClient(_SAMPLE_REPORT)
        generate_report(
            client,
            _make_report_input(),
            system_prompt=_FAKE_PROMPT,
        )
        system_content = client.last_messages[0]["content"]
        assert "Original Manager" not in system_content

    def test_empty_response_returns_fallback(self) -> None:
        client = FakeLLMClient("")
        result = generate_report(
            client,
            _make_report_input(),
            system_prompt=_FAKE_PROMPT,
        )
        assert result == _FALLBACK_REPORT

    def test_whitespace_response_returns_fallback(self) -> None:
        client = FakeLLMClient("   \n  ")
        result = generate_report(
            client,
            _make_report_input(),
            system_prompt=_FAKE_PROMPT,
        )
        assert result == _FALLBACK_REPORT

    def test_loads_default_prompt_file(self) -> None:
        """Verify the default prompt file exists and is loadable."""
        from iffootball.llm.report_generation import _load_system_prompt

        prompt = _load_system_prompt()
        assert len(prompt) > 0
        assert "report" in prompt.lower()

    def test_missing_sections_returns_fallback(self) -> None:
        """LLM output without required headings triggers fallback."""
        client = FakeLLMClient("Just some prose without any headings.")
        result = generate_report(
            client,
            _make_report_input(),
            system_prompt=_FAKE_PROMPT,
        )
        assert result == _FALLBACK_REPORT

    def test_partial_sections_returns_fallback(self) -> None:
        """LLM output missing some headings triggers fallback."""
        partial = "## Summary\n\nSome text.\n\n## Key Differences\n\nMore text."
        client = FakeLLMClient(partial)
        result = generate_report(
            client,
            _make_report_input(),
            system_prompt=_FAKE_PROMPT,
        )
        assert result == _FALLBACK_REPORT

    def test_fallback_has_all_required_sections(self) -> None:
        """The fallback report itself must contain all required sections."""
        for heading in REQUIRED_SECTIONS["en"]:
            assert heading in _FALLBACK_REPORT, f"Missing {heading} in fallback"


# ---------------------------------------------------------------------------
# Tests: ReportInput / DTO construction
# ---------------------------------------------------------------------------


class TestReportInput:
    def test_frozen(self) -> None:
        ri = _make_report_input()
        try:
            ri.n_runs = 99  # type: ignore[misc]
            raised = False
        except AttributeError:
            raised = True
        assert raised

    def test_default_limitations_available(self) -> None:
        assert len(DEFAULT_LIMITATIONS["en"]) >= 3
        assert len(DEFAULT_LIMITATIONS["ja"]) >= 3

    def test_required_sections_defined(self) -> None:
        assert len(REQUIRED_SECTIONS["en"]) == 5
        assert "## Summary" in REQUIRED_SECTIONS["en"]
        assert "## Limitations" in REQUIRED_SECTIONS["en"]
        assert len(REQUIRED_SECTIONS["ja"]) == 5


class TestActionExplanationEntry:
    def test_has_tp_type_and_action(self) -> None:
        entry = ActionExplanationEntry(
            tp_type="bench_streak",
            action="resist",
            explanation="Test.",
            label="analysis",
            confidence_note="",
        )
        assert entry.tp_type == "bench_streak"
        assert entry.action == "resist"


class TestPlayerImpactEntry:
    def test_has_diff_fields(self) -> None:
        entry = PlayerImpactEntry(
            player_name="Test",
            impact_score=0.5,
            form_diff=-0.1,
            fatigue_diff=0.05,
            understanding_diff=-0.2,
            trust_diff=-0.05,
        )
        assert entry.form_diff == -0.1
        assert entry.understanding_diff == -0.2
