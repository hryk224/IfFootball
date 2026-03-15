"""Natural language input structuring via LLM.

Converts user free-text scenario descriptions into structured trigger
parameters that can be used to construct ManagerChangeTrigger or
TransferInTrigger instances.

Pipeline:
  1. Load system prompt from prompts/input_structuring_v1.md.
  2. Wrap user text in a JSON user message.
  3. Send [system, user] to LLMClient.complete().
  4. Parse JSON response and validate required fields per trigger type.
  5. Return a StructuredInput result (success or failure).

Prompt injection mitigation:
  User text is embedded only in the user message as a JSON string value,
  never in the system prompt. The system prompt is loaded from a static
  file and never interpolated.

Note on TransferInTrigger:
  This function can parse transfer scenarios, but TransferInTrigger is
  not yet supported in the simulation engine (raises NotImplementedError).
  Callers must check engine support separately from parse_success.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from iffootball.llm.client import LLMClient

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_DEFAULT_PROMPT_PATH = (
    Path(__file__).parents[3] / "prompts" / "input_structuring_v1.md"
)


def _load_system_prompt(path: Path | None = None) -> str:
    """Load system prompt from file. Raises FileNotFoundError if missing."""
    resolved = path if path is not None else _DEFAULT_PROMPT_PATH
    return resolved.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_TRIGGER_TYPES = frozenset({"manager_change", "player_transfer_in"})

# Required param keys per trigger type (excluding trigger_type and team_name
# which are validated at the top level).
_REQUIRED_PARAMS: dict[str, tuple[str, ...]] = {
    "manager_change": (
        "outgoing_manager_name",
        "incoming_manager_name",
        "transition_type",
        "applied_at",
    ),
    "player_transfer_in": (
        "player_name",
        "expected_role",
        "applied_at",
    ),
}

_VALID_TRANSITION_TYPES = frozenset({"mid_season", "pre_season"})
_VALID_EXPECTED_ROLES = frozenset({"starter", "rotation", "squad"})


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructuredInput:
    """Result of structuring user input into trigger parameters.

    Attributes:
        trigger_type:  "manager_change" or "player_transfer_in",
                       or "" on parse failure.
        team_name:     Target team name, or "" on parse failure.
        params:        Trigger-specific parameter dict. Keys match the
                       trigger type schema. Values may be None for
                       optional fields the user did not specify.
        parse_success: True if the LLM output was valid and complete.
        error_message: Empty on success; describes the failure otherwise.
    """

    trigger_type: str
    team_name: str
    params: dict[str, Any]
    parse_success: bool
    error_message: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def structure_input(
    client: LLMClient,
    user_text: str,
    *,
    system_prompt: str | None = None,
) -> StructuredInput:
    """Convert user natural language input to structured trigger parameters.

    Args:
        client:        LLMClient implementation.
        user_text:     User's free-text scenario description.
        system_prompt: Override the loaded system prompt (tests only).

    Returns:
        StructuredInput with parsed parameters or error information.
    """
    prompt = (
        system_prompt if system_prompt is not None else _load_system_prompt()
    )

    user_payload = json.dumps({"user_text": user_text}, ensure_ascii=False)

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_payload},
    ]

    raw = client.complete(messages)
    return _parse_response(raw)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_response(raw: str) -> StructuredInput:
    """Parse and validate LLM JSON response into StructuredInput."""
    data = _parse_json(raw)
    if data is None:
        return _error("Failed to parse LLM response as JSON.")

    trigger_type = data.get("trigger_type")

    # LLM returned an explicit "not recognized" response.
    if trigger_type is None:
        error = data.get("error", "Trigger type not identified.")
        if not isinstance(error, str):
            error = "Trigger type not identified."
        return _error(error)

    if not isinstance(trigger_type, str) or trigger_type not in _VALID_TRIGGER_TYPES:
        return _error(f"Unknown trigger_type: {trigger_type!r}")

    # Validate team_name (required for all triggers).
    team_name = data.get("team_name")
    if not isinstance(team_name, str) or not team_name.strip():
        return _error("Missing or empty team_name.")

    # Validate trigger-specific required params.
    required_keys = _REQUIRED_PARAMS[trigger_type]
    params: dict[str, Any] = {}
    for key in required_keys:
        params[key] = data.get(key)

    # Apply defaults for enum fields when null (prompt instructs LLM to
    # default, but it may still return null).
    _apply_enum_defaults(trigger_type, params)

    # Validate enum fields.
    validation_error = _validate_enum_fields(trigger_type, params)
    if validation_error:
        return _error(validation_error)

    # Validate applied_at is int or None.
    applied_at = params.get("applied_at")
    if applied_at is not None and not isinstance(applied_at, int):
        return _error(f"Invalid applied_at: expected int or null, got {type(applied_at).__name__}")

    # Validate that at least the key identifying field is present.
    key_field_error = _validate_key_field(trigger_type, params)
    if key_field_error:
        return _error(key_field_error)

    return StructuredInput(
        trigger_type=trigger_type,
        team_name=team_name.strip(),
        params=params,
        parse_success=True,
        error_message="",
    )


def _apply_enum_defaults(
    trigger_type: str, params: dict[str, Any]
) -> None:
    """Fill in default values for null enum fields.

    The prompt instructs the LLM to default these, but it may still
    return null. We apply the same defaults here to ensure completeness.
    """
    if trigger_type == "manager_change":
        if params.get("transition_type") is None:
            params["transition_type"] = "mid_season"

    if trigger_type == "player_transfer_in":
        if params.get("expected_role") is None:
            params["expected_role"] = "starter"


def _validate_enum_fields(
    trigger_type: str, params: dict[str, Any]
) -> str:
    """Validate enum fields have allowed values. Returns error or empty."""
    if trigger_type == "manager_change":
        tt = params.get("transition_type")
        if tt not in _VALID_TRANSITION_TYPES:
            return f"Invalid transition_type: {tt!r}"

    if trigger_type == "player_transfer_in":
        role = params.get("expected_role")
        if role not in _VALID_EXPECTED_ROLES:
            return f"Invalid expected_role: {role!r}"

    return ""


def _validate_key_field(
    trigger_type: str, params: dict[str, Any]
) -> str:
    """Validate that the essential identifying field is non-empty."""
    if trigger_type == "manager_change":
        name = params.get("incoming_manager_name")
        if not isinstance(name, str) or not name.strip():
            return "Missing incoming_manager_name."

    if trigger_type == "player_transfer_in":
        name = params.get("player_name")
        if not isinstance(name, str) or not name.strip():
            return "Missing player_name."

    return ""


def _error(message: str) -> StructuredInput:
    """Create a failure StructuredInput."""
    return StructuredInput(
        trigger_type="",
        team_name="",
        params={},
        parse_success=False,
        error_message=message,
    )


def _parse_json(raw: str) -> dict[str, Any] | None:
    """Return parsed dict from a JSON string, or None on failure."""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
        return None
    except json.JSONDecodeError:
        return None
