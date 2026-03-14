"""LLM knowledge queries for manager and league hypothesis attributes.

All values returned by functions in this module are derived from LLM output
and must be treated as [hypothesis] labels, not facts. They supplement
StatsBomb-derived attributes where statistical derivation is impossible
(e.g., tactical style rigidity, perceived league physicality).

Pipeline:
  1. Load system prompt from prompts/knowledge_query_v1.md.
     Raises FileNotFoundError if missing — prompt files are required.
  2. Serialize query parameters as JSON into the user message.
  3. Send [system, user] to LLMClient.complete().
  4. Parse and validate JSON response.
  5. On invalid JSON or out-of-range values, return safe defaults
     (defined as module-level constants below).

Prompt injection mitigation:
  External inputs (manager_name, league_name) are embedded only in the
  user message as JSON values, never in the system prompt. The system
  prompt is loaded from a static file and never interpolated.
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

_DEFAULT_PROMPT_PATH = Path(__file__).parents[3] / "prompts" / "knowledge_query_v1.md"


def _load_system_prompt(path: Path | None = None) -> str:
    """Load system prompt from file. Raises FileNotFoundError if missing."""
    resolved = path if path is not None else _DEFAULT_PROMPT_PATH
    return resolved.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Valid ordinal level strings.
_VALID_LEVELS = frozenset({"high", "mid", "low"})

# Mapping from ordinal level string to style_stubbornness float.
_LEVEL_TO_STUBBORNNESS: dict[str, float] = {
    "high": 80.0,
    "mid": 50.0,
    "low": 20.0,
}

# Defaults used when LLM output cannot be parsed or validated.
_DEFAULT_STYLE_STUBBORNNESS: float = 50.0  # "mid"
_DEFAULT_PREFERRED_FORMATION: str | None = None
_DEFAULT_LEVEL: str = "mid"

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ManagerStyleResult:
    """LLM-derived manager style attributes — hypothesis labels.

    Both fields are hypotheses sourced from LLM output. They supplement
    StatsBomb-derived attributes and must not override fact-derived values
    where facts are available.

    Attributes:
        style_stubbornness:  Tactical rigidity as a float
                             (80.0 / 50.0 / 20.0 from high / mid / low).
        preferred_formation: LLM-assessed preferred formation, or None
                             if the LLM could not determine one or the
                             value was not in the provided formation_options.
                             Use as fallback only when StatsBomb lineups
                             are insufficient (e.g. tenure too short).
    """

    style_stubbornness: float
    preferred_formation: str | None


@dataclass(frozen=True)
class LeagueCharacteristicsResult:
    """LLM-derived league characteristics — hypothesis labels.

    All three fields are hypotheses. They are stored in LeagueContext and
    must be labelled [hypothesis] in any output that surfaces them to users.

    Attributes:
        pressing_level:     Perceived pressing intensity ("high"/"mid"/"low").
        physicality_level:  Perceived physicality ("high"/"mid"/"low").
        tactical_complexity: Perceived tactical sophistication
                             ("high"/"mid"/"low").
    """

    pressing_level: str
    physicality_level: str
    tactical_complexity: str


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------


def query_manager_style(
    client: LLMClient,
    manager_name: str,
    formation_options: list[str],
    system_prompt: str | None = None,
) -> ManagerStyleResult:
    """Query LLM for a manager's tactical style attributes.

    Returns [hypothesis] labels. Not to be used as facts.

    The system prompt is loaded from prompts/knowledge_query_v1.md by
    default. Pass system_prompt to override (useful in tests).

    Args:
        client:           LLMClient implementation.
        manager_name:     Manager name as recorded in StatsBomb data.
        formation_options: Valid formation strings the LLM may return
                           (e.g. ["4-3-3", "4-2-3-1", "other"]).
                           If empty, preferred_formation is always None.
        system_prompt:    Override the loaded system prompt (tests only).

    Returns:
        ManagerStyleResult with style_stubbornness and preferred_formation.
        Falls back to _DEFAULT_STYLE_STUBBORNNESS and
        _DEFAULT_PREFERRED_FORMATION on parse failure or invalid values.
    """
    prompt = system_prompt if system_prompt is not None else _load_system_prompt()
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "query_type": "manager_style",
                    "manager_name": manager_name,
                    "formation_options": formation_options,
                },
                ensure_ascii=False,
            ),
        },
    ]
    raw = client.complete(messages)
    return _parse_manager_style(raw, formation_options)


def query_league_characteristics(
    client: LLMClient,
    league_name: str,
    system_prompt: str | None = None,
) -> LeagueCharacteristicsResult:
    """Query LLM for a league's characteristic levels.

    Returns [hypothesis] labels. Not to be used as facts.

    Args:
        client:       LLMClient implementation.
        league_name:  Human-readable league name (e.g. "Premier League").
        system_prompt: Override the loaded system prompt (tests only).

    Returns:
        LeagueCharacteristicsResult with pressing_level, physicality_level,
        and tactical_complexity. Falls back to _DEFAULT_LEVEL on parse
        failure or invalid values.
    """
    prompt = system_prompt if system_prompt is not None else _load_system_prompt()
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "query_type": "league_characteristics",
                    "league_name": league_name,
                },
                ensure_ascii=False,
            ),
        },
    ]
    raw = client.complete(messages)
    return _parse_league_characteristics(raw)


# ---------------------------------------------------------------------------
# Internal parsers
# ---------------------------------------------------------------------------


def _parse_manager_style(
    raw: str,
    formation_options: list[str],
) -> ManagerStyleResult:
    """Parse and validate LLM response for a manager_style query."""
    data = _parse_json(raw)
    if data is None:
        return ManagerStyleResult(
            style_stubbornness=_DEFAULT_STYLE_STUBBORNNESS,
            preferred_formation=_DEFAULT_PREFERRED_FORMATION,
        )

    stubbornness_level = _extract_level(data, "style_stubbornness")
    style_stubbornness = _LEVEL_TO_STUBBORNNESS.get(
        stubbornness_level, _DEFAULT_STYLE_STUBBORNNESS
    )
    preferred_formation = _extract_formation(data, formation_options)

    return ManagerStyleResult(
        style_stubbornness=style_stubbornness,
        preferred_formation=preferred_formation,
    )


def _parse_league_characteristics(raw: str) -> LeagueCharacteristicsResult:
    """Parse and validate LLM response for a league_characteristics query."""
    data = _parse_json(raw)
    if data is None:
        return LeagueCharacteristicsResult(
            pressing_level=_DEFAULT_LEVEL,
            physicality_level=_DEFAULT_LEVEL,
            tactical_complexity=_DEFAULT_LEVEL,
        )

    return LeagueCharacteristicsResult(
        pressing_level=_extract_level(data, "pressing_level"),
        physicality_level=_extract_level(data, "physicality_level"),
        tactical_complexity=_extract_level(data, "tactical_complexity"),
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


def _extract_level(data: dict[str, Any], key: str) -> str:
    """Extract a high/mid/low level value; return _DEFAULT_LEVEL if invalid."""
    value = data.get(key)
    if isinstance(value, str) and value in _VALID_LEVELS:
        return value
    return _DEFAULT_LEVEL


def _extract_formation(
    data: dict[str, Any],
    formation_options: list[str],
) -> str | None:
    """Extract formation from data; return None if invalid or options empty."""
    if not formation_options:
        return None
    value = data.get("preferred_formation")
    if isinstance(value, str) and value in formation_options:
        return value
    return None
