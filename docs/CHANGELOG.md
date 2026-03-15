# Changelog

## Live Demo

Deployed the IfFootball Live Demo to Streamlit Community Cloud with guided scenario UI, LLM report generation, and Japanese language support.

**Live Demo:** https://iffootball-bfmlfowrz7fxf8j4fkuuqv.streamlit.app/

### Added

- Guided scenario UI with 3 preset cards (Van Gaal→Mourinho, Van Gaal→Klopp, Mourinho→Ranieri) and Custom scenario expander
- Candidate resolver (`candidates.py`) with week-aware manager lookup and cross-league support
- Per-scenario demo cache (`data/demo_cache/{team}_w{week}.db`) for instant startup on Streamlit Cloud
- Session state caching to prevent re-simulation on Streamlit rerun
- Summary card with headline, metrics, and points definition caption
- Player Impact with direction arrows (▲/▼/─), one-line reasons, and horizontal radar layout
- Japanese report generation prompt (`report_generation_ja_v1.md`) and Report Language selector
- Groq provider verified at 1.8s/en and 2.5s/ja report generation
- Per-provider model env vars (OPENAI_MODEL, ANTHROPIC_MODEL, GEMINI_MODEL, GROQ_MODEL); removed shared LLM_MODEL
- Streamlit Cloud secrets bridge for LLM provider resolution
- Post-process quality checks (player value validation, dangling sentence detection — warning-only)
- Demo screenshots in README

### Changed

- Output order: Summary → Player Impact → Detailed Analysis → Team Comparison (conclusion first)
- Report sections deduplicated: Summary shown at page top only, not repeated in Detailed Analysis
- Label system renamed from `fact` to `data` across all prompts, code, and documentation
- Radar chart figures shrunk to (4,4) with unified fonts and legend sizing
- Team Comparison centered at 1/3 page width
- Competition labels human-readable (Premier League 2015-16)
- Runs/Seed moved to Advanced Settings expander
- Report prompts hardened: sign convention, event name formatting, proper noun preservation, all-player coverage, causal chain flow, sentence completion rules

---

## OSS Release Preparation

Prepared the project for public release with documentation, safety checks, schema hardening, and UX improvements.

### Added

- `CONTRIBUTING.md` with development workflow, architecture principles, and supply chain policy
- `README.ja.md` — Full Japanese translation of README
- Schema version management (`db_meta` table with `schema_version` tracking; legacy DB detection)
- `cascade_runs` header table to distinguish saved-empty runs from never-saved
- `OPENAI_BASE_URL` and `OPENAI_MODEL` env var support for OpenAI-compatible endpoints
- Design memo for conversation memory (deferred to M7+)

### Changed

- README restructured: scenario quote + Quick Start (no API key needed) + real backtest output example at top; Full Setup separated from Quick Start
- CHANGELOG rewritten in English with public-facing terminology (internal milestone names removed)
- Disclaimer added to README, README.ja.md, and Streamlit UI (not a prediction engine; output liability limitation)
- LLM data transmission disclosure added to app, README, README.ja.md, .env.example
- StatsBomb Open Data license reference added to Data Source section
- `source provenance` → `source classification` (expression accuracy)
- DB schema hardened to version 2: CHECK constraints on player_agents (percentiles 0-100, dynamic state 0-1) and team_baselines (xG >= 0, possession 0-1); comparison_results metadata columns now NOT NULL
- Streamlit UI: human-readable competition labels, Advanced Settings expander for Runs/Seed, conclusion-first output order, data-driven Causal Chain when LLM unavailable
- Name-based identifier review: assessed as no action needed for StatsBomb Open Data scope; future ID-column migration documented

---

## MVP Finalization

Completed real LLM provider integration, transfer trigger UI, and end-to-end user acceptance testing.

### Added

- Multi-provider LLM client (`llm/providers/`)
  - OpenAI (default), Anthropic, Google Gemini, Groq
  - `LLM_PROVIDER` env var for explicit selection; auto-detect fallback by API key + SDK availability
  - `.env` loading via `python-dotenv`
- Transfer trigger as experimental UI path
  - Trigger type selector (Manager Change / Player Transfer) in sidebar
  - Transfer mode: team radar hidden, new signing info card displayed
- `.env.example` template for all provider API keys

### Changed

- `app.py` renders LLM-generated Markdown reports when a provider is configured; falls back to data-only structured report on failure or when unconfigured
- `pyproject.toml` adds `llm` optional dependency group (`openai` / `anthropic` / `google-genai` / `groq`) and `python-dotenv` as core dependency

---

## Backtesting

Executed a backtest scenario (Manchester United — Van Gaal dismissal, 2015-16 season, week 29) and recorded evaluation results.

### Added

- `docs/simulation-rules.md` — All TOML parameter definitions with rationale and source classification
- `scripts/backtest_van_gaal.py` — Backtest script: initialize → run_comparison → player_impact → JSON output

### Findings

- Points delta: +0.5 (B-A, 20 runs) — slight positive trend but statistically uncertain against std 3.5
- Cascade event over-firing: form_drop +29.9/run, tactical_confusion +36.0/run over 9 matches
- tactical_understanding uniformly -0.250 for all players (individual differences not reflected)

---

## Output Layer

Added visualization, LLM report generation, and Streamlit UI for end-to-end what-if simulation.

### Added

#### Trigger Enhancements

- `ManagerChangeTrigger.incoming_profile` — Optional tactical profile for the incoming manager; copies static attributes when provided, neutral defaults when None
- `TransferInTrigger.player` — PlayerAgent payload for incoming transfers; added to squad with role-based trust initialization (starter=0.7, rotation=0.5, squad=0.3) and deepcopy for branch isolation

#### Visualization

- Team comparison radar chart (`visualization/radar_chart.py` + `radar_data.py`)
  - 5 axes: xG/90 (simulation output), xGA/90 (fixed baseline), PPDA / Possession / Prog Passes (tactical estimates)
  - League-average-centered 0-1 normalization; PPDA and xGA inverted
- Player impact radar chart (`visualization/player_radar.py` + `player_impact.py`)
  - 4 axes: Form / Fatigue (inverted) / Tactical Understanding / Manager Trust
  - Impact score: mean absolute dynamic-state difference across runs; ranked by player_id for deterministic tiebreaking
- Tactical estimate module (`visualization/tactical_estimate.py`)
  - Manager profile → PPDA / Possession / Progressive Passes estimates with league-average regression and formation adjustment

#### LLM Output Layer

- Turning point action explanation (`llm/action_explanation.py` + `prompts/action_explanation_v1.md`)
  - data / analysis / hypothesis labels with source_types for data provenance
- Structured report generation (`llm/report_generation.py` + `prompts/report_generation_v1.md`)
  - 5 fixed sections (Summary / Key Differences / Causal Chain / Player Impact / Limitations)
  - Section heading validation with structured fallback on malformed LLM output
- Natural language input structuring (`llm/input_structuring.py` + `prompts/input_structuring_v1.md`)
  - Parses manager_change and player_transfer_in trigger types
  - Enum field defaults, applied_at type validation

#### UI

- Streamlit app (`app.py`)
  - Sidebar: competition/team/manager/trigger type/trigger week/N runs/seed
  - Main area: delta metrics, team radar, player impact radars, structured report

#### Storage

- `ComparisonMeta` / `ComparisonResultWithMeta` — Persists rng_seed, n_runs, trigger_summary, and created_at (UTC ISO 8601) alongside comparison results
- Documented cascade*events run_id naming convention (`{branch}*{index}`)

### Changed

#### Config Externalization

- Moved RuleBasedHandler action distributions from hardcoded values to `turning_points.toml` `[action_distribution]` section
  - 3 conditions (bench_streak_low_trust / low_understanding / default); all 3 actions required per distribution

#### Cascade Event Taxonomy

- Added 3 event types to `VALID_EVENT_TYPES`: `adaptation_progress`, `tactical_confusion` (now actively emitted), `squad_unrest`
- Engine: adapt → `adaptation_progress`; low_understanding + adapt → `tactical_confusion`; 2+ resist/week → `squad_unrest`

#### Consistency Model

- Extended `PlayerAgent.consistency` to position-group composite metric
  - GK/MF: pass_std, DF: def_std (Tackle+Interception), FW: xg_std
  - Common secondary: pass_std; composite weight 0.5/0.5 (provisional)
  - Within-group percentile ranking; zero-event matches included in variance calculation

#### Match Result

- Added `expected_goals_for` / `expected_goals_against` fields to `MatchResult`

#### Dependencies

- Added `matplotlib==3.10.8`, `streamlit==1.45.0`
- Downgraded `pandas` 3.0.1 → 2.3.3 (streamlit compatibility)

---

## Simulation Foundation

Connected data foundation components into an initialization pipeline and implemented the weekly simulation engine, Branch A/B comparison, and result persistence.

### Added

#### Initialization Pipeline

- Single `initialize()` entry point connecting collectors → converters → LLM → storage (`pipeline.py`)
  - 3-way data separation (target-team pre-trigger / league-wide pre-trigger / full-season) to structurally prevent future data leakage
  - `build_league_context()` computes league-average PPDA, xG, and progressive passes from StatsBomb data
  - `cultural_inertia` auto-updated from manager tenure length
  - LLM enrichment is optional (updates `style_stubbornness` only; `preferred_formation` respects StatsBomb fact values)

#### Domain Models

- `ManagerChangeTrigger` / `TransferInTrigger` domain models (`agents/trigger.py`)
  - `trigger_type` fixed via `field(init=False)`; `applied_at` semantics documented (injected after week N, effects from week N+1)

#### Simulation Rules Config

- TOML config files with typed frozen-dataclass loader (`config.py` + `config/simulation_rules/`)
  - `adaptation.toml` — fatigue, tactical understanding, trust, fatigue penalty weight
  - `turning_points.toml` — player TP thresholds (bench_streak / tactical_understanding / trust_low), manager TP thresholds (job_security / style_stubbornness)
  - `SimulationRules.load(config_dir)` with `__post_init__` value-range validation

#### Simulation Engine

- Poisson match result model (`simulation/match_result.py`)
  - `agent_state_factor` from starter form/fatigue (0.5-1.5 clamp)
  - Home advantage adjustment
  - Seeded `numpy.random.Generator` for reproducibility
- Formation parsing + lineup selection (`simulation/lineup_selection.py`)
  - `parse_formation("4-3-3")` → position slot conversion
  - `selection_score` from tactical fit, trust, form, fatigue penalty, understanding penalty (all config-driven)
- Weekly state updates (`simulation/state_update.py`)
  - fatigue, tactical_understanding (adaptation curve), manager_trust, job_security
- Turning point detection + RuleBasedHandler (`simulation/turning_point.py`)
  - `ActionDistribution` with normalization and validation
  - `TurningPointHandler` Protocol (swappable rule-based / future LLM-based boundary)
  - Player TPs (bench_streak / low_understanding), Manager TPs (job_security warning/critical)
- CascadeEvent model + tracker (`simulation/cascade_tracker.py`)
  - Depth limit and importance threshold filtering
  - `record_chained()` for automatic cause-chain tracking
- Weekly simulation loop (`simulation/engine.py`)
  - `Simulation.run()` — 10-step weekly loop over all fixtures
  - `SimulationResult` with deepcopy snapshots of final squad/manager state
- Branch A/B comparison (`simulation/comparison.py`)
  - `run_comparison()` — N parallel runs with `rng.spawn(2)` for independent reproducible streams
  - `AggregatedResult` / `DeltaMetrics` / `ComparisonResult`

#### Storage

- Extended SQLite storage (`storage/db.py`)
  - Added persistence for `LeagueContext`, `CascadeEvent`, `ComparisonResult`
  - `ComparisonResult` stored as JSON blobs (`run_results` excluded; restored as empty tuples on load)
  - `CascadeEvent` keyed by `(comparison_key, run_id, ordinal)`

#### Data Quality

- `PlayerAgent.consistency` derived from per-match xG variance (`converters/stats_to_attributes.py`)
  - Inverted percentile (low variance = high consistency)
  - Players with fewer than 5 xG-contributing matches retain 50.0 placeholder

### Changed

#### Scale Unification

- Unified `PlayerAgent` dynamic state attributes (`current_form` / `tactical_understanding` / `manager_trust`) from 0-100 to 0.0-1.0 scale
  - Config values and lineup selection scores aligned to 0-1

#### Config Separation

- Moved `home_advantage_factor` from `AdaptationConfig` to dedicated `MatchConfig` (`config/simulation_rules/match.toml`)

---

## Data Foundation

Built the data layer for initializing all domain objects from StatsBomb Open Data.

### Added

#### Data Collection

- StatsBomb Open Data retrieval layer (`collectors/statsbomb.py`)
  - `get_competitions()` / `get_matches()` / `get_events()` / `get_lineups()` via statsbombpy

#### Domain Models

- `PlayerAgent` (`agents/player.py`) — RoleFamily / BroadPosition enums, technical / adaptation / dynamic state fields
- `TeamBaseline` (`agents/team.py`) — StatsBomb metrics (xG/90, xGA/90, PPDA, progressive passes, possession) and league standing
- `ManagerAgent` (`agents/manager.py`) — StatsBomb-derived tactical attributes separated from LLM-derived hypothesis attributes
- `Fixture` / `FixtureList` / `OpponentStrength` (`agents/fixture.py`) — Frozen for safe Branch A/B sharing
- `LeagueContext` (`agents/league.py`) — Fact fields (StatsBomb) and hypothesis fields (LLM) clearly separated; frozen with `dataclasses.replace()` for updates

#### Converters

- Stats → `PlayerAgent` attributes (`converters/stats_to_attributes.py`) — Same-league percentile normalization with playing-time filter
- Team metrics → `TeamBaseline` (`converters/team_stats.py`) — xG/xGA, PPDA, progressive passes, possession, cultural_inertia
- Manager metrics → `ManagerAgent` (`converters/manager_stats.py`) — Tenure estimation, pressing intensity, possession preference, preferred formation from StatsBomb events
- `FixtureList` / `OpponentStrength` (`converters/fixture_stats.py`) — Elo rating calculation (initial 1500, K=20)

#### Storage

- SQLite persistence layer (`storage/db.py`)
  - UPSERT for all tables; `fixture_lists` header table distinguishes "saved empty" from "never saved"
  - Transactional DELETE+INSERT for fixture list re-saves

#### LLM Knowledge Query

- `LLMClient` Protocol (`llm/client.py`) — Single `complete()` method for provider-agnostic interface
- Knowledge query functions (`llm/knowledge_query.py`)
  - `query_manager_style()` — style_stubbornness + preferred_formation as hypothesis labels
  - `query_league_characteristics()` — pressing/physicality/tactical complexity levels
  - Safe defaults on invalid LLM output; system/user role separation for prompt injection mitigation
- System prompt managed as external file (`prompts/knowledge_query_v1.md`)
