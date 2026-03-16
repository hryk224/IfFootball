"""Tests for LLM report generation."""

from __future__ import annotations

import json

from iffootball.llm.report_generation import (
    CausalStepEntry,
    DEFAULT_LIMITATIONS,
    EvidenceEntry,
    PlayerImpactMeta,
    REQUIRED_SECTIONS,
    ActionExplanationEntry,
    PlayerImpactEntry,
    ReportInput,
    _build_payload,
    _build_retry_instruction,
    _FALLBACK_REPORT,
    _has_expected_hypothesis_labels,
    _has_no_internal_metadata_leak,
    _has_no_shared_reset_repetition,
    _has_causal_chain_coverage,
    _has_causal_chain_paragraph_format,
    _has_consistent_summary_directions,
    _has_no_multi_claim_sentences,
    _has_correct_summary_tradeoff,
    _has_no_summary_highlights_overuse,
    _has_sentence_level_labels,
    _has_valid_summary_length,
    _has_valid_key_differences_format,
    _normalize_signed_deltas_en,
    _section_label_detail,
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
        for heading in REQUIRED_SECTIONS:
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

    def test_required_sections_defined(self) -> None:
        assert len(REQUIRED_SECTIONS) == 5  # LLM-generated sections only
        assert "## Summary" in REQUIRED_SECTIONS
        assert "## Limitations" in REQUIRED_SECTIONS


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


# ---------------------------------------------------------------------------
# Sign/direction normalization tests
# ---------------------------------------------------------------------------


class TestNormalizeSignedDeltasEn:
    def test_decreased_by_negative_removes_sign(self) -> None:
        assert _normalize_signed_deltas_en("decreased by -35.3") == "decreased by 35.3"

    def test_increased_by_negative_flips_direction(self) -> None:
        assert _normalize_signed_deltas_en("increased by -24.0") == "decreased by 24.0"

    def test_dropped_by_negative_removes_sign(self) -> None:
        assert _normalize_signed_deltas_en("dropped by -2.5") == "dropped by 2.5"

    def test_normal_decreased_untouched(self) -> None:
        assert _normalize_signed_deltas_en("decreased by 35.3") == "decreased by 35.3"

    def test_normal_increased_untouched(self) -> None:
        assert _normalize_signed_deltas_en("increased by 24.0") == "increased by 24.0"

    def test_changed_by_negative_keeps_changed(self) -> None:
        assert _normalize_signed_deltas_en("changed by -5.0") == "changed by 5.0"

    def test_capitalized_direction(self) -> None:
        assert _normalize_signed_deltas_en("Increased by -3.0") == "Decreased by 3.0"

    def test_multiple_occurrences(self) -> None:
        text = "Form decreased by -0.11 and trust increased by -0.08."
        result = _normalize_signed_deltas_en(text)
        assert "decreased by 0.11" in result
        assert "decreased by 0.08" in result

    def test_no_match_returns_unchanged(self) -> None:
        text = "The score is 3.5 points."
        assert _normalize_signed_deltas_en(text) == text




# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------


class TestSentenceLevelLabels:
    def test_rejects_paragraph_with_single_label(self) -> None:
        report = (
            "## Summary\n\n"
            "The manager change had a positive impact. "
            "Points increased by 2.1. "
            "The team adapted quickly. [data]\n\n"
            "## Key Differences\n\n"
            "- Points increased by 2.1 [data]\n\n"
            "## Causal Chain\n\n"
            "The change led to adaptation. [analysis]\n\n"
            "## Player Impact\n\n"
            "Player A was most affected. [data]\n\n"
            "## Limitations\n\n"
            "- Known constraint."
        )
        # Summary has 3 sentences but only 1 label.
        assert _has_sentence_level_labels(report) is False

    def test_accepts_well_labelled_report(self) -> None:
        report = (
            "## Summary\n\n"
            "Points increased by 2.1. [data] "
            "The team adapted quickly. [analysis]\n\n"
            "## Key Differences\n\n"
            "- Points increased by 2.1 [data]\n\n"
            "## Causal Chain\n\n"
            "The change led to adaptation. [analysis]\n\n"
            "## Player Impact\n\n"
            "Player A form dropped by 0.15. [data]\n\n"
            "## Limitations\n\n"
            "- Known constraint."
        )
        assert _has_sentence_level_labels(report) is True

class TestInternalMetadataLeak:
    def test_rejects_field_names_in_text(self) -> None:
        report = (
            "## Summary\n\n"
            "Since causal_steps is not provided, we use action_explanations. [data]\n\n"
            "## Key Differences\n\n"
            "- Points diff [data]\n\n"
            "## Causal Chain\n\n"
            "No data. [analysis]\n\n"
            "## Player Impact\n\n"
            "No impact. [data]\n\n"
            "## Limitations\n\n"
            "- None."
        )
        assert _has_no_internal_metadata_leak(report) is False

    def test_accepts_clean_text(self) -> None:
        assert _has_no_internal_metadata_leak(_SAMPLE_REPORT) is True

    def test_rejects_display_hints_mention(self) -> None:
        report = "The display_hints indicate compact mode. [data]"
        assert _has_no_internal_metadata_leak(report) is False


class TestKeyDifferencesFormat:
    def test_rejects_key_value_style(self) -> None:
        report = (
            "## Key Differences\n\n"
            "- metric_name: total_points_mean, diff: 2.1, unit: points_mean, "
            "direction: increased, label: data\n\n"
            "## Summary\n\nOK. [data]"
        )
        assert _has_valid_key_differences_format(report) is False

    def test_accepts_natural_format(self) -> None:
        report = (
            "## Key Differences\n\n"
            "- Mean points increased by 2.1 points [data]\n"
            "- Adaptation progress increased by 24.0 events per run [data]\n\n"
            "## Summary\n\nOK. [data]"
        )
        assert _has_valid_key_differences_format(report) is True

class TestSharedResetRepetition:
    def test_rejects_per_player_repetition(self) -> None:
        meta = PlayerImpactMeta(shared_resets={"understanding": -0.25})
        ri = ReportInput(
            trigger_description="test",
            points_mean_a=10.0,
            points_mean_b=12.0,
            points_mean_diff=2.0,
            cascade_count_diff={},
            n_runs=10,
            player_impacts=[
                PlayerImpactEntry("A", 0.3, -0.1, 0.0, -0.25, -0.05),
                PlayerImpactEntry("B", 0.2, 0.05, 0.0, -0.25, 0.1),
            ],
            action_explanations=[],
            limitations=[],
            player_impact_meta=meta,
        )
        report = (
            "## Player Impact\n\n"
            "All players experienced a tactical understanding reset of -0.25.\n\n"
            "**A** — form dropped by 0.1. [data] "
            "Understanding decreased by 0.25. [data]\n\n"
            "**B** — trust increased by 0.1. [data] "
            "Understanding decreased by 0.25. [data]\n\n"
            "## Limitations\n\n- None."
        )
        assert _has_no_shared_reset_repetition(report, ri) is False

    def test_accepts_single_mention(self) -> None:
        meta = PlayerImpactMeta(shared_resets={"understanding": -0.25})
        ri = ReportInput(
            trigger_description="test",
            points_mean_a=10.0,
            points_mean_b=12.0,
            points_mean_diff=2.0,
            cascade_count_diff={},
            n_runs=10,
            player_impacts=[
                PlayerImpactEntry("A", 0.3, -0.1, 0.0, -0.25, -0.05),
                PlayerImpactEntry("B", 0.2, 0.05, 0.0, -0.25, 0.1),
            ],
            action_explanations=[],
            limitations=[],
            player_impact_meta=meta,
        )
        report = (
            "## Player Impact\n\n"
            "All players experienced a tactical understanding reset of -0.25.\n\n"
            "**A** — form dropped by 0.1. [data]\n\n"
            "**B** — trust increased by 0.1. [data]\n\n"
            "## Limitations\n\n- None."
        )
        assert _has_no_shared_reset_repetition(report, ri) is True


class TestHypothesisLabels:
    def test_rejects_speculative_wording_without_label(self) -> None:
        ri = _make_report_input()
        report = (
            "## Summary\n\n"
            "The team may struggle with adaptation. [analysis]\n\n"
            "## Key Differences\n\n- Points [data]\n\n"
            "## Causal Chain\n\nAdaptation. [analysis]\n\n"
            "## Player Impact\n\nImpact. [data]\n\n"
            "## Limitations\n\n- None."
        )
        # "may" is speculative but no [hypothesis] label anywhere.
        assert _has_expected_hypothesis_labels(report, ri) is False

    def test_accepts_speculative_with_hypothesis_label(self) -> None:
        ri = _make_report_input()
        report = (
            "## Summary\n\n"
            "The team may struggle with adaptation. [hypothesis]\n\n"
            "## Key Differences\n\n- Points [data]\n\n"
            "## Causal Chain\n\nAdaptation. [analysis]\n\n"
            "## Player Impact\n\nImpact. [data]\n\n"
            "## Limitations\n\n- None."
        )
        assert _has_expected_hypothesis_labels(report, ri) is True

    def test_rejects_missing_hypothesis_from_causal_steps(self) -> None:
        ri = ReportInput(
            trigger_description="test",
            points_mean_a=10.0,
            points_mean_b=12.0,
            points_mean_diff=2.0,
            cascade_count_diff={},
            n_runs=10,
            player_impacts=[],
            action_explanations=[],
            limitations=[],
            causal_steps=[
                CausalStepEntry(
                    step_id="cs-001",
                    cause="test",
                    effect="test",
                    affected_agent="A",
                    event_type="form_drop",
                    depth=3,
                    paragraph_label="hypothesis",
                    evidence_labels=("hypothesis",),
                    evidence=(
                        EvidenceEntry(
                            statement="test",
                            label="hypothesis",
                            source="rule_based_model",
                        ),
                    ),
                ),
            ],
        )
        report = (
            "## Summary\n\nOK. [data]\n\n"
            "## Key Differences\n\n- Points [data]\n\n"
            "## Causal Chain\n\nAdaptation happened. [analysis]\n\n"
            "## Player Impact\n\nNo impact. [data]\n\n"
            "## Limitations\n\n- None."
        )
        # causal_steps has hypothesis but report only has [analysis].
        assert _has_expected_hypothesis_labels(report, ri) is False


class TestGenerateReportRetry:
    def test_retries_once_on_validation_failure(self) -> None:
        """First response has metadata leak, second is clean."""
        call_count = 0
        bad_report = (
            "## Summary\n\n"
            "Since causal_steps is not provided, points decreased. [data]\n\n"
            "## Key Differences\n\n- Points: -1.2 [data]\n\n"
            "## Causal Chain\n\nAdaptation. [analysis]\n\n"
            "## Player Impact\n\nPlayer 7 form dropped. [data]\n\n"
            "## Limitations\n\n- Poisson model."
        )
        good_report = _SAMPLE_REPORT

        class RetryClient:
            def complete(self, messages: list[dict[str, str]]) -> str:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return bad_report
                return good_report

        ri = _make_report_input()
        result = generate_report(
            RetryClient(),  # type: ignore[arg-type]
            ri,
            system_prompt=_FAKE_PROMPT,
        )
        assert call_count == 2
        # Should use the retry result (good_report).
        assert "causal_steps" not in result

    def test_falls_back_after_retry_failure(self) -> None:
        """Both responses have metadata leak — falls back."""
        bad_report = (
            "## Summary\n\n"
            "The display_hints show something. [data]\n\n"
            "## Key Differences\n\n- Points: -1.2 [data]\n\n"
            "## Causal Chain\n\nChain. [analysis]\n\n"
            "## Player Impact\n\nImpact. [data]\n\n"
            "## Limitations\n\n- None."
        )

        class AlwaysBadClient:
            def complete(self, messages: list[dict[str, str]]) -> str:
                return bad_report

        ri = _make_report_input()
        result = generate_report(
            AlwaysBadClient(),  # type: ignore[arg-type]
            ri,
            system_prompt=_FAKE_PROMPT,
        )
        # Validator failed twice — must fall back, never return invalid report.
        assert "display_hints" not in result
        assert "Unable to generate" in result or "No data available" in result


# ---------------------------------------------------------------------------
# Player Impact sentence-level label tests
# ---------------------------------------------------------------------------


class TestPlayerImpactLabels:
    def test_split_sentences_pass(self) -> None:
        """Two separate sentences with labels each — should pass."""
        report = (
            "## Summary\n\n"
            "Points changed. [data]\n\n"
            "## Key Differences\n\n"
            "- Points: +2.1 [data]\n\n"
            "## Causal Chain\n\n"
            "Cause led to effect. [analysis]\n\n"
            "## Player Impact\n\n"
            "Juan Mata's form decreased by 0.11. [data] "
            "This suggests reduced suitability. [analysis]\n\n"
            "## Limitations\n\n"
            "- None."
        )
        assert _has_sentence_level_labels(report) is True

    def test_combined_sentence_detected_by_multi_claim_validator(self) -> None:
        """Two claims joined by 'while'/'indicating' — detected by multi-claim check."""
        report = (
            "## Summary\n\n"
            "Points changed. [data]\n\n"
            "## Key Differences\n\n"
            "- Points: +2.1 [data]\n\n"
            "## Causal Chain\n\n"
            "Cause led to effect. [analysis]\n\n"
            "## Player Impact\n\n"
            "Juan Mata's form decreased by 0.11 while trust also dropped by 0.08. [data]\n\n"
            "Ander Herrera's trust increased, indicating adaptation. [data]\n\n"
            "## Limitations\n\n"
            "- None."
        )
        # Sentence-level labels pass (each sentence has 1 label).
        assert _has_sentence_level_labels(report) is True
        # But multi-claim validator catches the combining words.
        assert _has_no_multi_claim_sentences(report) is False

    def test_multi_sentence_one_label_fails(self) -> None:
        """Paragraph with multiple sentences but only final label — should fail."""
        report = (
            "## Summary\n\n"
            "Points changed. [data]\n\n"
            "## Key Differences\n\n"
            "- Points: +2.1 [data]\n\n"
            "## Causal Chain\n\n"
            "Cause led to effect. [analysis]\n\n"
            "## Player Impact\n\n"
            "Juan Mata's form decreased by 0.11. "
            "His trust also dropped by 0.08. "
            "This suggests he struggled with the new system. [analysis]\n\n"
            "## Limitations\n\n"
            "- None."
        )
        assert _has_sentence_level_labels(report) is False

    def test_section_detail_identifies_player_impact(self) -> None:
        """section_label_detail should identify player_impact as problematic."""
        report = (
            "## Summary\n\n"
            "Points changed. [data]\n\n"
            "## Key Differences\n\n"
            "- Points: +2.1 [data]\n\n"
            "## Causal Chain\n\n"
            "Cause led to effect. [analysis]\n\n"
            "## Player Impact\n\n"
            "Juan Mata's form decreased by 0.11. "
            "His trust also dropped. [data]\n\n"
            "## Limitations\n\n"
            "- None."
        )
        detail = _section_label_detail(report)
        assert detail["player_impact"]["unlabelled"] > 0
        assert detail["summary"]["unlabelled"] == 0


# ---------------------------------------------------------------------------
# Section-specific retry instruction tests
# ---------------------------------------------------------------------------


class TestRetryInstruction:
    def test_en_includes_player_impact_section(self) -> None:
        issues = ["missing sentence-level labels"]
        section_detail = {
            "summary": {"labelled": 2, "unlabelled": 0},
            "player_impact": {"labelled": 1, "unlabelled": 2},
        }
        instruction = _build_retry_instruction(issues, section_detail)
        assert "Player Impact" in instruction
        assert "Summary" not in instruction  # summary has no unlabelled

    def test_no_section_detail_when_no_label_issue(self) -> None:
        issues = ["internal metadata leaked"]
        section_detail = {
            "player_impact": {"labelled": 1, "unlabelled": 2},
        }
        instruction = _build_retry_instruction(issues, section_detail)
        assert "Player Impact" not in instruction

    def test_multiple_sections_with_issues(self) -> None:
        issues = ["missing sentence-level labels"]
        section_detail = {
            "summary": {"labelled": 1, "unlabelled": 1},
            "player_impact": {"labelled": 1, "unlabelled": 1},
        }
        instruction = _build_retry_instruction(issues, section_detail)
        assert "Summary" in instruction
        assert "Player Impact" in instruction

    def test_multi_claim_retry_includes_split_guidance(self) -> None:
        issues = ["multi-claim sentences in player impact"]
        instruction = _build_retry_instruction(issues, {})
        assert "while" in instruction
        assert "separate" in instruction.lower()

# ---------------------------------------------------------------------------
# Multi-claim sentence validator tests
# ---------------------------------------------------------------------------


class TestMultiClaimSentences:
    def test_rejects_while_pattern(self) -> None:
        report = (
            "## Player Impact\n\n"
            "Form decreased by 0.11 while trust also dropped. [data]\n\n"
            "## Limitations\n\n- None."
        )
        assert _has_no_multi_claim_sentences(report) is False

    def test_rejects_indicating_pattern(self) -> None:
        report = (
            "## Player Impact\n\n"
            "Trust increased, indicating adaptation to the new system. [data]\n\n"
            "## Limitations\n\n- None."
        )
        assert _has_no_multi_claim_sentences(report) is False

    def test_rejects_suggesting_pattern(self) -> None:
        report = (
            "## Player Impact\n\n"
            "Form dropped significantly, suggesting poor fit. [analysis]\n\n"
            "## Limitations\n\n- None."
        )
        assert _has_no_multi_claim_sentences(report) is False

    def test_accepts_clean_sentences(self) -> None:
        report = (
            "## Player Impact\n\n"
            "Form decreased by 0.11. [data] "
            "This may indicate poor adaptation. [hypothesis]\n\n"
            "## Limitations\n\n- None."
        )
        assert _has_no_multi_claim_sentences(report) is True

    def test_no_player_impact_section_passes(self) -> None:
        report = "## Summary\n\nOK. [data]"
        assert _has_no_multi_claim_sentences(report) is True

    def test_is_valid_report_catches_multi_claim(self) -> None:
        """_is_valid_report integrates multi-claim check."""
        from iffootball.llm.report_generation import _is_valid_report

        report = (
            "## Summary\n\nOK. [data]\n\n"
            "## Key Differences\n\n- Points: +2.1 [data]\n\n"
            "## Causal Chain\n\nCause. [analysis]\n\n"
            "## Player Impact\n\n"
            "Form dropped while trust declined. [data]\n\n"
            "## Limitations\n\n- None."
        )
        ri = _make_report_input()
        valid, issues = _is_valid_report(report, ri)
        assert not valid
        assert "multi-claim sentences in player impact" in issues


# ---------------------------------------------------------------------------
# Summary direction consistency tests
# ---------------------------------------------------------------------------


def _make_report_input_with_highlights() -> ReportInput:
    """ReportInput with highlights that have direction info."""
    from iffootball.llm.report_generation import HighlightEntry

    return ReportInput(
        trigger_description="Manager change at week 29",
        points_mean_a=12.2,
        points_mean_b=14.3,
        points_mean_diff=2.1,
        cascade_count_diff={
            "adaptation_progress": 24.0,
            "tactical_confusion": -35.3,
            "form_drop": 8.4,
        },
        n_runs=20,
        player_impacts=[],
        action_explanations=[],
        limitations=[],
        highlights=[
            HighlightEntry(
                metric_name="total_points_mean",
                diff=2.1,
                label="data",
                statement="Points increased by 2.1.",
                unit="points_mean",
                direction="increased",
            ),
            HighlightEntry(
                metric_name="adaptation_progress",
                diff=24.0,
                label="data",
                statement="Adaptation progress increased by 24.0.",
                unit="events_per_run",
                direction="increased",
            ),
            HighlightEntry(
                metric_name="tactical_confusion",
                diff=35.3,
                label="data",
                statement="Tactical confusion decreased by 35.3.",
                unit="events_per_run",
                direction="decreased",
            ),
            HighlightEntry(
                metric_name="form_drop",
                diff=8.4,
                label="data",
                statement="Form drop increased by 8.4.",
                unit="events_per_run",
                direction="increased",
            ),
        ],
    )


class TestSummaryDirectionConsistency:
    def test_rejects_reversed_direction(self) -> None:
        """tactical_confusion direction is 'decreased' but summary says 'increased'."""
        ri = _make_report_input_with_highlights()
        report = (
            "## Summary\n\n"
            "Points increased by 2.1. [data] "
            "Tactical confusion increased significantly. [data]\n\n"
            "## Key Differences\n\n- Points [data]\n\n"
            "## Causal Chain\n\nCause. [analysis]\n\n"
            "## Player Impact\n\nImpact. [data]\n\n"
            "## Limitations\n\n- None."
        )
        assert _has_consistent_summary_directions(report, ri) is False

    def test_accepts_correct_direction(self) -> None:
        ri = _make_report_input_with_highlights()
        report = (
            "## Summary\n\n"
            "Points increased by 2.1. [data] "
            "Tactical confusion decreased by 35.3. [data]\n\n"
            "## Key Differences\n\n- Points [data]\n\n"
            "## Causal Chain\n\nCause. [analysis]\n\n"
            "## Player Impact\n\nImpact. [data]\n\n"
            "## Limitations\n\n- None."
        )
        assert _has_consistent_summary_directions(report, ri) is True

    def test_accepts_unmentioned_events(self) -> None:
        """Events not mentioned in summary are OK."""
        ri = _make_report_input_with_highlights()
        report = (
            "## Summary\n\n"
            "Points increased by 2.1. [data]\n\n"
            "## Key Differences\n\n- Points [data]\n\n"
            "## Causal Chain\n\nCause. [analysis]\n\n"
            "## Player Impact\n\nImpact. [data]\n\n"
            "## Limitations\n\n- None."
        )
        assert _has_consistent_summary_directions(report, ri) is True

    def test_no_highlights_passes(self) -> None:
        """No highlights in input — validator passes."""
        ri = _make_report_input()
        report = "## Summary\n\nPoints changed. [data]\n\n## Limitations\n\n- None."
        assert _has_consistent_summary_directions(report, ri) is True

    def test_is_valid_report_catches_direction_mismatch(self) -> None:
        from iffootball.llm.report_generation import _is_valid_report

        ri = _make_report_input_with_highlights()
        report = (
            "## Summary\n\n"
            "Points increased. [data] "
            "Tactical confusion increased sharply. [data]\n\n"
            "## Key Differences\n\n- Points increased by 2.1 [data]\n\n"
            "## Causal Chain\n\nCause led to effect. [analysis]\n\n"
            "## Player Impact\n\nImpact. [data]\n\n"
            "## Limitations\n\n- None."
        )
        valid, issues = _is_valid_report(report, ri)
        assert not valid
        assert "summary direction contradicts input data" in issues

    def test_rejects_second_mention_with_wrong_direction(self) -> None:
        """Event correct first time but reversed second time."""
        ri = _make_report_input_with_highlights()
        report = (
            "## Summary\n\n"
            "Tactical confusion decreased initially. [data] "
            "However tactical confusion increased later. [data]\n\n"
            "## Key Differences\n\n- Points [data]\n\n"
            "## Causal Chain\n\nCause. [analysis]\n\n"
            "## Player Impact\n\nImpact. [data]\n\n"
            "## Limitations\n\n- None."
        )
        assert _has_consistent_summary_directions(report, ri) is False

    def test_rejects_direction_word_before_event_name(self) -> None:
        """'Increased tactical confusion' — direction word before event."""
        ri = _make_report_input_with_highlights()
        report = (
            "## Summary\n\n"
            "Points increased by 2.1. [data] "
            "Increased tactical confusion disrupted the team. [analysis]\n\n"
            "## Key Differences\n\n- Points [data]\n\n"
            "## Causal Chain\n\nCause. [analysis]\n\n"
            "## Player Impact\n\nImpact. [data]\n\n"
            "## Limitations\n\n- None."
        )
        # tactical_confusion direction is "decreased" but "increased" appears
        # before the event name in the same sentence.
        assert _has_consistent_summary_directions(report, ri) is False

    def test_retry_instruction_includes_direction_fix(self) -> None:
        issues = ["summary direction contradicts input data"]
        instruction = _build_retry_instruction(issues, {})
        assert "direction" in instruction.lower()
        assert "Summary" in instruction


# ---------------------------------------------------------------------------
# Summary UX validators
# ---------------------------------------------------------------------------


class TestSummaryLength:
    def test_rejects_too_many_sentences(self) -> None:
        ri = _make_report_input_with_highlights()
        report = (
            "## Summary\n\n"
            "Trigger happened. [data] "
            "Points increased. [data] "
            "Adaptation rose. [data] "
            "Confusion decreased. [data] "
            "Form dropped. [data] "
            "This is a trade-off. [analysis]\n\n"
            "## Key Differences\n\n- Points [data]\n\n"
            "## Causal Chain\n\nCause. [analysis]\n\n"
            "## Player Impact\n\nImpact. [data]\n\n"
            "## Limitations\n\n- None."
        )
        # 6 sentences, default max is 4.
        assert _has_valid_summary_length(report, ri) is False

    def test_accepts_within_limit(self) -> None:
        ri = _make_report_input_with_highlights()
        report = (
            "## Summary\n\n"
            "Trigger happened. [data] "
            "Points increased by 2.1. [data] "
            "Form drops increased. [analysis] "
            "Net positive with costs. [analysis]\n\n"
            "## Key Differences\n\n- Points [data]\n\n"
            "## Causal Chain\n\nCause. [analysis]\n\n"
            "## Player Impact\n\nImpact. [data]\n\n"
            "## Limitations\n\n- None."
        )
        assert _has_valid_summary_length(report, ri) is True

class TestSummaryHighlightsOveruse:
    def test_rejects_three_or_more_highlights_mentioned(self) -> None:
        ri = _make_report_input_with_highlights()
        report = (
            "## Summary\n\n"
            "Points increased by 2.1. [data] "
            "Adaptation progress rose by 24.0. [data] "
            "Tactical confusion decreased by 35.3. [data] "
            "Form drop increased by 8.4. [data]\n\n"
            "## Key Differences\n\n- Points [data]\n\n"
            "## Causal Chain\n\nCause. [analysis]\n\n"
            "## Player Impact\n\nImpact. [data]\n\n"
            "## Limitations\n\n- None."
        )
        # 4 highlight metrics mentioned — too many.
        assert _has_no_summary_highlights_overuse(report, ri) is False

    def test_accepts_two_highlights_mentioned(self) -> None:
        ri = _make_report_input_with_highlights()
        report = (
            "## Summary\n\n"
            "Trigger happened. [data] "
            "Points increased by 2.1. [data] "
            "Form drop increased by 8.4. [analysis]\n\n"
            "## Key Differences\n\n- Points [data]\n\n"
            "## Causal Chain\n\nCause. [analysis]\n\n"
            "## Player Impact\n\nImpact. [data]\n\n"
            "## Limitations\n\n- None."
        )
        # Only points + form_drop = 2 metrics mentioned.
        assert _has_no_summary_highlights_overuse(report, ri) is True

    def test_no_highlights_passes(self) -> None:
        ri = _make_report_input()
        report = "## Summary\n\nPoints changed. [data]\n\n## Limitations\n\n- None."
        assert _has_no_summary_highlights_overuse(report, ri) is True

    def test_is_valid_report_catches_summary_issues(self) -> None:
        from iffootball.llm.report_generation import _is_valid_report

        ri = _make_report_input_with_highlights()
        report = (
            "## Summary\n\n"
            "Points increased. [data] "
            "Adaptation progress rose. [data] "
            "Tactical confusion decreased. [data] "
            "Form drop increased. [data]\n\n"
            "## Key Differences\n\n- Points increased by 2.1 [data]\n\n"
            "## Causal Chain\n\nCause. [analysis]\n\n"
            "## Player Impact\n\nImpact. [data]\n\n"
            "## Limitations\n\n- None."
        )
        valid, issues = _is_valid_report(report, ri)
        assert not valid
        assert "summary lists too many highlights" in issues

    def test_retry_instruction_for_summary_overuse(self) -> None:
        issues = ["summary lists too many highlights"]
        instruction = _build_retry_instruction(issues, {})
        assert "Key Differences" in instruction

    def test_retry_instruction_for_summary_length_default(self) -> None:
        issues = ["summary exceeds max sentences"]
        instruction = _build_retry_instruction(issues, {})
        assert "4 sentences" in instruction

    def test_retry_instruction_for_summary_length_compact(self) -> None:
        from iffootball.simulation.report_planner import DisplayHints

        ri = ReportInput(
            trigger_description="test",
            points_mean_a=0.0,
            points_mean_b=0.0,
            points_mean_diff=0.0,
            cascade_count_diff={},
            n_runs=1,
            player_impacts=[],
            action_explanations=[],
            limitations=[],
            display_hints=DisplayHints(
                section_order=(),
                expanded_step_ids=frozenset(),
                collapsed_step_ids=frozenset(),
                featured_players=(),
                show_limitations_info=False,
                summary_max_sentences=2,
            ),
        )
        issues = ["summary exceeds max sentences"]
        instruction = _build_retry_instruction(
            issues, {}, report_input=ri,
        )
        assert "2 sentences" in instruction
        assert "4 sentences" not in instruction


# ---------------------------------------------------------------------------
# Summary tradeoff metric validator
# ---------------------------------------------------------------------------


def _make_report_input_with_tradeoff() -> ReportInput:
    """ReportInput with display_hints specifying form_drop as tradeoff."""
    from iffootball.llm.report_generation import HighlightEntry
    from iffootball.simulation.report_planner import DisplayHints

    return ReportInput(
        trigger_description="Manager change at week 29",
        points_mean_a=12.2,
        points_mean_b=14.3,
        points_mean_diff=2.1,
        cascade_count_diff={},
        n_runs=20,
        player_impacts=[],
        action_explanations=[],
        limitations=[],
        display_hints=DisplayHints(
            section_order=("summary", "key_differences", "causal_chain", "player_impact", "limitations"),
            expanded_step_ids=frozenset(),
            collapsed_step_ids=frozenset(),
            featured_players=(),
            show_limitations_info=False,
            summary_max_sentences=4,
            summary_tradeoff_metric="form_drop",
        ),
        highlights=[
            HighlightEntry("total_points_mean", 2.1, "data", "", "points_mean", "increased"),
            HighlightEntry("adaptation_progress", 24.0, "data", "", "events_per_run", "increased"),
            HighlightEntry("tactical_confusion", 35.3, "data", "", "events_per_run", "decreased"),
            HighlightEntry("form_drop", 8.4, "data", "", "events_per_run", "increased"),
        ],
    )


class TestSummaryTradeoffMetric:
    def test_rejects_wrong_metric_in_tradeoff(self) -> None:
        """Designated tradeoff is form_drop but summary uses adaptation_progress."""
        ri = _make_report_input_with_tradeoff()
        report = (
            "## Summary\n\n"
            "Manager changed at week 29. [data] "
            "Points increased by 2.1. [data] "
            "Adaptation progress events increased by 24.0, indicating a cost. [analysis] "
            "Net positive with trade-offs. [analysis]\n\n"
            "## Limitations\n\n- None."
        )
        assert _has_correct_summary_tradeoff(report, ri) is False

    def test_accepts_correct_tradeoff_metric(self) -> None:
        ri = _make_report_input_with_tradeoff()
        report = (
            "## Summary\n\n"
            "Manager changed at week 29. [data] "
            "Points increased by 2.1. [data] "
            "Form drop events increased by 8.4, indicating a transition cost. [analysis] "
            "Net positive with trade-offs. [analysis]\n\n"
            "## Limitations\n\n- None."
        )
        assert _has_correct_summary_tradeoff(report, ri) is True

    def test_no_display_hints_passes(self) -> None:
        ri = _make_report_input()
        report = "## Summary\n\nOK. [data]\n\n## Limitations\n\n- None."
        assert _has_correct_summary_tradeoff(report, ri) is True

    def test_retry_includes_metric_name(self) -> None:
        ri = _make_report_input_with_tradeoff()
        issues = ["summary tradeoff uses wrong metric"]
        instruction = _build_retry_instruction(
            issues, {}, report_input=ri,
        )
        assert "form drop" in instruction.lower()


# ---------------------------------------------------------------------------
# Causal Chain coverage tests
# ---------------------------------------------------------------------------


def _make_report_input_with_causal_steps() -> ReportInput:
    """ReportInput with causal_steps for coverage testing."""
    from iffootball.llm.report_generation import CausalStepEntry, EvidenceEntry
    from iffootball.simulation.report_planner import DisplayHints

    return ReportInput(
        trigger_description="Manager change at week 29",
        points_mean_a=12.2,
        points_mean_b=14.3,
        points_mean_diff=2.1,
        cascade_count_diff={},
        n_runs=20,
        player_impacts=[],
        action_explanations=[],
        limitations=[],
        display_hints=DisplayHints(
            section_order=(
                "summary", "key_differences", "causal_chain",
                "player_impact", "limitations",
            ),
            expanded_step_ids=frozenset({"cs-001", "cs-002", "cs-003"}),
            collapsed_step_ids=frozenset(),
            featured_players=(),
            show_limitations_info=False,
        ),
        causal_steps=[
            CausalStepEntry(
                step_id="cs-001",
                cause="Manager change triggered reset",
                effect="Understanding dropped",
                affected_agent="Juan Mata",
                event_type="tactical_confusion",
                depth=1,
                paragraph_label="analysis",
                evidence_labels=("data",),
                evidence=(
                    EvidenceEntry("Understanding dropped", "data", "simulation_output"),
                ),
            ),
            CausalStepEntry(
                step_id="cs-002",
                cause="Confusion caused form drop",
                effect="Form declined",
                affected_agent="Juan Mata",
                event_type="form_drop",
                depth=2,
                paragraph_label="analysis",
                evidence_labels=("data",),
                evidence=(
                    EvidenceEntry("Form dropped by 0.11", "data", "simulation_output"),
                ),
            ),
            CausalStepEntry(
                step_id="cs-003",
                cause="Trust shifted to pressing players",
                effect="Trust gained",
                affected_agent="Ander Herrera",
                event_type="adaptation_progress",
                depth=2,
                paragraph_label="analysis",
                evidence_labels=("analysis",),
                evidence=(
                    EvidenceEntry("Trust +0.12", "analysis", "rule_based_model"),
                ),
            ),
        ],
    )


class TestCausalChainCoverage:
    def test_rejects_fallback_text_when_steps_provided(self) -> None:
        ri = _make_report_input_with_causal_steps()
        report = (
            "## Summary\n\nOK. [data]\n\n"
            "## Key Differences\n\n- Points [data]\n\n"
            "## Causal Chain\n\n"
            "No causal chain data is available for this scenario.\n\n"
            "## Player Impact\n\nImpact. [data]\n\n"
            "## Limitations\n\n- None."
        )
        assert _has_causal_chain_coverage(report, ri) is False

    def test_rejects_missing_agent(self) -> None:
        ri = _make_report_input_with_causal_steps()
        report = (
            "## Summary\n\nOK. [data]\n\n"
            "## Key Differences\n\n- Points [data]\n\n"
            "## Causal Chain\n\n"
            "The change caused tactical confusion for Juan Mata. [analysis]\n\n"
            "## Player Impact\n\nImpact. [data]\n\n"
            "## Limitations\n\n- None."
        )
        # Ander Herrera is missing.
        assert _has_causal_chain_coverage(report, ri) is False

    def test_accepts_all_agents_in_separate_paragraphs(self) -> None:
        ri = _make_report_input_with_causal_steps()
        report = (
            "## Summary\n\nOK. [data]\n\n"
            "## Key Differences\n\n- Points [data]\n\n"
            "## Causal Chain\n\n"
            "The change caused tactical confusion for Juan Mata. [analysis]\n\n"
            "Juan Mata's form declined by 0.11. [analysis]\n\n"
            "Ander Herrera gained trust. [analysis]\n\n"
            "## Player Impact\n\nImpact. [data]\n\n"
            "## Limitations\n\n- None."
        )
        assert _has_causal_chain_coverage(report, ri) is True

    def test_one_paragraph_passes_coverage_fails_format(self) -> None:
        """3 steps in 1 paragraph — coverage passes (sentences ok) but format fails."""
        ri = _make_report_input_with_causal_steps()
        report = (
            "## Summary\n\nOK. [data]\n\n"
            "## Key Differences\n\n- Points [data]\n\n"
            "## Causal Chain\n\n"
            "The change caused tactical confusion for Juan Mata. [analysis] "
            "Juan Mata's form declined by 0.11. [analysis] "
            "Ander Herrera gained trust. [analysis]\n\n"
            "## Player Impact\n\nImpact. [data]\n\n"
            "## Limitations\n\n- None."
        )
        # Content coverage passes (3 sentences, all agents).
        assert _has_causal_chain_coverage(report, ri) is True
        # Paragraph format fails (1 paragraph, need 3).
        assert _has_causal_chain_paragraph_format(report, ri) is False

    def test_is_valid_report_catches_paragraph_count(self) -> None:
        """_is_valid_report includes paragraph count as hard fail."""
        from iffootball.llm.report_generation import _is_valid_report

        ri = _make_report_input_with_causal_steps()
        report = (
            "## Summary\n\nOK. [data]\n\n"
            "## Key Differences\n\n- Points increased by 2.1 [data]\n\n"
            "## Causal Chain\n\n"
            "Confusion for Juan Mata. [analysis] "
            "Mata form dropped. [analysis] "
            "Ander Herrera gained trust. [analysis]\n\n"
            "## Player Impact\n\nImpact. [data]\n\n"
            "## Limitations\n\n- None."
        )
        valid, issues = _is_valid_report(report, ri)
        assert not valid
        assert "causal chain paragraph count too low" in issues

    def test_no_steps_passes(self) -> None:
        ri = _make_report_input()
        report = "## Causal Chain\n\nNo data. [analysis]\n\n## Limitations\n\n- None."
        assert _has_causal_chain_coverage(report, ri) is True

    def test_compact_mode_exempt(self) -> None:
        """Compact hides causal_chain — no coverage needed."""
        from iffootball.simulation.report_planner import DisplayHints

        ri = ReportInput(
            trigger_description="test",
            points_mean_a=0.0,
            points_mean_b=0.0,
            points_mean_diff=0.0,
            cascade_count_diff={},
            n_runs=1,
            player_impacts=[],
            action_explanations=[],
            limitations=[],
            display_hints=DisplayHints(
                section_order=("summary", "key_differences", "player_impact", "limitations"),
                expanded_step_ids=frozenset(),
                collapsed_step_ids=frozenset(),
                featured_players=(),
                show_limitations_info=False,
            ),
            causal_steps=[
                CausalStepEntry(
                    step_id="cs-001",
                    cause="test",
                    effect="test",
                    affected_agent="A",
                    event_type="form_drop",
                    depth=1,
                    paragraph_label="analysis",
                    evidence_labels=("data",),
                    evidence=(EvidenceEntry("test", "data", "simulation_output"),),
                ),
            ],
        )
        report = "## Summary\n\nOK. [data]\n\n## Limitations\n\n- None."
        assert _has_causal_chain_coverage(report, ri) is True

    def test_is_valid_report_catches_missing_steps(self) -> None:
        from iffootball.llm.report_generation import _is_valid_report

        ri = _make_report_input_with_causal_steps()
        report = (
            "## Summary\n\nOK. [data]\n\n"
            "## Key Differences\n\n- Points increased by 2.1 [data]\n\n"
            "## Causal Chain\n\n"
            "No causal chain data is available for this scenario.\n\n"
            "## Player Impact\n\nImpact. [data]\n\n"
            "## Limitations\n\n- None."
        )
        valid, issues = _is_valid_report(report, ri)
        assert not valid
        assert "causal chain steps missing" in issues

    def test_retry_includes_step_count_and_agents(self) -> None:
        ri = _make_report_input_with_causal_steps()
        issues = ["causal chain steps missing"]
        instruction = _build_retry_instruction(
            issues, {}, report_input=ri,
        )
        assert "3" in instruction  # 3 steps
        assert "Juan Mata" in instruction
        assert "Ander Herrera" in instruction

    def test_retry_includes_paragraph_count(self) -> None:
        ri = _make_report_input_with_causal_steps()
        issues = ["causal chain paragraph count too low"]
        instruction = _build_retry_instruction(
            issues, {}, report_input=ri,
        )
        assert "3 paragraphs" in instruction
        assert "blank line" in instruction

    def test_rejects_merged_same_agent_steps(self) -> None:
        """3 steps but only 2 paragraphs — one Juan Mata step was merged."""
        ri = _make_report_input_with_causal_steps()
        report = (
            "## Summary\n\nOK. [data]\n\n"
            "## Key Differences\n\n- Points [data]\n\n"
            "## Causal Chain\n\n"
            "Juan Mata's form declined due to the tactical reset. [analysis]\n\n"
            "Ander Herrera gained trust. [analysis]\n\n"
            "## Player Impact\n\nImpact. [data]\n\n"
            "## Limitations\n\n- None."
        )
        # 3 steps expected, but only 2 paragraphs. Both agents present but
        # cs-001 (tactical confusion for Mata) was merged into cs-002.
        assert _has_causal_chain_coverage(report, ri) is False

    def test_accepts_three_paragraphs_for_three_steps(self) -> None:
        """3 steps and 3 paragraphs — coverage met."""
        ri = _make_report_input_with_causal_steps()
        report = (
            "## Summary\n\nOK. [data]\n\n"
            "## Key Differences\n\n- Points [data]\n\n"
            "## Causal Chain\n\n"
            "The tactical reset caused understanding to drop for Juan Mata. [analysis]\n\n"
            "Juan Mata's form then declined by 0.11. [analysis]\n\n"
            "Ander Herrera gained trust as a pressing-oriented player. [analysis]\n\n"
            "## Player Impact\n\nImpact. [data]\n\n"
            "## Limitations\n\n- None."
        )
        assert _has_causal_chain_coverage(report, ri) is True
