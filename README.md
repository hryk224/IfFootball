# IfFootball

[日本語版はこちら](README.ja.md)

> _"What if Klopp had managed Chelsea from the start of the season?"_

Simulate how a different manager or squad change might have altered a football team's entire season. IfFootball uses StatsBomb Open Data, rule-based causal simulation, and optional LLM-generated analysis. **No API key required to run — LLM is optional.**

![IfFootball Demo - Preset cards and Summary](docs/images/demo-overview.png)
![IfFootball Demo - Player Impact and Detailed Analysis](docs/images/demo-detailed-analysis.png)

**[Live Demo](https://iffootball-bfmlfowrz7fxf8j4fkuuqv.streamlit.app/)** — Try it in your browser, no setup required.

## Quick Start

```bash
git clone https://github.com/hryk224/IfFootball.git
cd IfFootball
uv sync --extra dev
uv run python scripts/build_season_cache.py   # one-time setup (~5 min)
uv run streamlit run app.py
```

Select a team, choose a scenario (manager change, player add, or player remove), pick a candidate from the dropdown, and click **Run Scenario**. All candidates come from the pre-built season cache — no API calls at runtime.

No `.env` file, no API key, no LLM setup needed for the basic experience.

## Example Output

Season scenario: Chelsea with Klopp instead of Mourinho (Premier League 2015-16, 5 simulation runs):

```
Branch A (Mourinho):  64.6 points (mean)
Branch B (Klopp):     56.0 points (mean)
Delta (B - A):        -8.6 points
```

The simulation compares a full 38-match season under each manager. Tactical profile differences (pressing intensity, possession preference, formation) drive lineup selection and adaptation dynamics. Full analysis in the Streamlit UI includes player impact radar charts and structured reports.

## Disclaimer

IfFootball is a **what-if simulation tool**, not a prediction engine. Results represent theoretical outcomes under specified rule-based parameters, not predictions of actual events. All simulation parameters are provisional and documented with their rationale in [simulation-rules.md](docs/simulation-rules.md#known-limitations). The authors are not responsible for any decisions made based on this software or its outputs.

## Key Features

- **Manager Change** — Simulate a full season under a different manager with tactical profile differences, squad trust recalibration, and adaptation curves
- **Player Add** — Add a player from another team with role-based trust initialization
- **Player Remove** — Simulate a season without a specific squad member
- **A/B Comparison** — Poisson match model, weekly state updates (fatigue, trust, tactical understanding), turning point detection with cascade event tracking
- **Visualization** — Team and player radar charts comparing Branch A/B outcomes
- **LLM Reports** — Structured comparison reports with data / analysis / hypothesis labels and source classification
- **Streamlit UI** — Single-page app for end-to-end simulation

### How It Works

```
[One-time] build_season_cache.py
    StatsBomb Data --> Season Cache DB (all teams, full season)

[Runtime] Streamlit UI
    Team + Scenario Selection (from season cache)
        |
        v
    Simulation Engine (N runs x 2 branches, 38 matches each)
        |--- Branch A: baseline (real manager/squad)
        |--- Branch B: scenario applied (alt manager / +player / -player)
        |
        v
    Comparison & Visualization (radar charts, cascade events, reports)
```

## Data Source

IfFootball uses [StatsBomb Open Data](https://github.com/statsbomb/open-data) exclusively. All metrics and terminology follow StatsBomb definitions. No scraping is involved. Attribution and usage terms follow the [StatsBomb Open Data license](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf).

Currently supported:

| Competition    | Season  | Teams                                            |
| -------------- | ------- | ------------------------------------------------ |
| Premier League | 2015-16 | All 20 teams (season cache includes full league) |

## Full Setup

The Quick Start above covers the minimal path. This section adds development tools and LLM configuration.

```bash
git clone https://github.com/hryk224/IfFootball.git
cd IfFootball
cp .env.example .env
uv sync --extra dev
npm install                    # for Markdown formatting (optional)
```

**Requirements:** Python >= 3.11, [uv](https://docs.astral.sh/uv/), Node.js (optional, for `npm run format:md` only).

### LLM Setup (Optional)

LLM provider SDKs are included in core dependencies. To enable LLM-generated reports, configure an API key in `.env`:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...
GROQ_API_KEY=gsk_...
```

Supported providers: OpenAI, Anthropic, Google Gemini, Groq. Without LLM configuration, the app runs in data-only mode with structured reports generated from simulation data.

**Model override:** Each provider has its own model env var (`OPENAI_MODEL`, `ANTHROPIC_MODEL`, `GEMINI_MODEL`, `GROQ_MODEL`). If unset, provider defaults are used.

**OpenAI-compatible APIs:** Set `OPENAI_BASE_URL` to use any OpenAI-compatible endpoint (e.g., Azure OpenAI, local inference servers).

**Note:** When LLM is enabled, scenario data (team names, player names, simulation results) is sent to the configured provider. Data handling follows the provider's policy.

## Usage

### Streamlit UI

```bash
uv run streamlit run app.py
```

Select a team and scenario, then click "Run Scenario". Season cache must be built first (`scripts/build_season_cache.py`).

## Design Principles

- **Rule-based core** — Simulation logic is defined by config files, not by LLM. Reproducible with seeded RNG
- **StatsBomb alignment** — All metrics follow StatsBomb definitions. No custom metric inventions
- **LLM as explanation layer** — LLM is limited to knowledge queries, action explanations, and report generation. It never makes simulation decisions
- **Transparent parameters** — Every simulation parameter is documented with its definition, value, and rationale in [simulation-rules.md](docs/simulation-rules.md)
- **Data / Analysis / Hypothesis labels** — All outputs distinguish between simulation data, model-derived analysis, and speculative hypotheses

## Tests

```bash
uv run python -m pytest        # 826+ tests
uv run python -m ruff check .  # linter
uv run python -m mypy .        # type checker
```

## Documentation

- [Simulation Rules](docs/simulation-rules.md) — All configurable parameters with definitions and rationale
- [Contributing](CONTRIBUTING.md) — Development workflow and guidelines

## License

[MIT](LICENSE)

## Acknowledgments

- [StatsBomb](https://statsbomb.com/) for the Open Data initiative
- Built as a personal project exploring rule-based sports simulation with LLM integration
