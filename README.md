# IfFootball

[日本語版はこちら](README.ja.md)

Football "what-if" simulator powered by StatsBomb Open Data. Explore how managerial changes and player transfers might have altered a team's season through rule-based causal simulation and LLM-generated analysis.

## Overview

IfFootball runs parallel simulations — one where nothing changes (Branch A) and one where a trigger is applied (Branch B) — then compares the outcomes across N stochastic runs.

**Example scenario:** _"What if Manchester United dismissed Van Gaal after match week 29 of the 2015-16 season?"_

### Key Features

- **Manager Change** — Simulate mid-season dismissal with tactical profile reset, squad trust recalibration, and adaptation curves
- **Player Transfer** _(Experimental)_ — Add a player to the squad with role-based trust initialization
- **A/B Comparison** — Poisson match model, weekly state updates (fatigue, trust, tactical understanding), turning point detection with cascade event tracking
- **Visualization** — Team and player radar charts comparing Branch A/B outcomes
- **LLM Reports** — Structured comparison reports with fact / analysis / hypothesis labels and source provenance
- **Streamlit UI** — Single-page app for end-to-end simulation

### How It Works

```
User Input (team, manager, trigger week)
    |
    v
StatsBomb Data --> Agent Initialization (players, manager, team baseline)
    |
    v
Simulation Engine (N runs x 2 branches)
    |--- Branch A: no change
    |--- Branch B: trigger applied
    |
    v
Comparison & Visualization (radar charts, cascade events, reports)
```

## Data Source

IfFootball uses [StatsBomb Open Data](https://github.com/statsbomb/open-data) exclusively. All metrics and terminology follow StatsBomb definitions. No scraping is involved.

Currently supported competitions (from `config/targets.toml`):

| Competition    | Season  | Teams                                                                                              |
| -------------- | ------- | -------------------------------------------------------------------------------------------------- |
| Premier League | 2015-16 | Manchester United, Manchester City, Arsenal, Liverpool, Chelsea, Tottenham Hotspur, Leicester City |
| La Liga        | 2015-16 | Real Madrid, Barcelona, Atletico Madrid                                                            |

## Setup

### Requirements

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) (package manager)
- Node.js (for Markdown formatting only)

### Installation

```bash
git clone https://github.com/hryk224/IfFootball.git
cd IfFootball
cp .env.example .env
uv sync --extra dev
npm install
```

### LLM Setup (Optional)

To enable LLM-generated reports, install provider SDKs and configure an API key:

```bash
uv sync --extra dev --extra llm
```

Then edit `.env` with one of:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...
GROQ_API_KEY=gsk_...
```

Supported providers: OpenAI, Anthropic, Google Gemini, Groq. Without LLM configuration, the app runs in data-only mode with structured reports generated from simulation data.

## Usage

### Streamlit UI

```bash
uv run streamlit run app.py
```

Configure parameters in the sidebar (team, manager, trigger week, simulation settings) and click "Run Simulation".

### Backtest Script

```bash
uv run python scripts/backtest_van_gaal.py
```

Runs the Van Gaal dismissal scenario (Manchester United, week 29, 20 runs) and outputs results to `output/backtest_van_gaal/results.json`.

## Design Principles

- **Rule-based core** — Simulation logic is defined by config files, not by LLM. Reproducible with seeded RNG
- **StatsBomb alignment** — All metrics follow StatsBomb definitions. No custom metric inventions
- **LLM as explanation layer** — LLM is limited to knowledge queries, action explanations, and report generation. It never makes simulation decisions
- **Transparent parameters** — Every simulation parameter is documented with its definition, value, and rationale in [simulation-rules.md](docs/simulation-rules.md)
- **Fact / Analysis / Hypothesis labels** — All outputs distinguish between data-backed facts, model-derived analysis, and speculative hypotheses

## Tests

```bash
uv run python -m pytest        # 547+ tests
uv run python -m ruff check .  # linter
uv run python -m mypy .        # type checker
```

## Documentation

- [Simulation Rules](docs/simulation-rules.md) — All configurable parameters with definitions and rationale
- [Changelog](docs/CHANGELOG.md) — Per-milestone change history
- [Contributing](CONTRIBUTING.md) — Development workflow and guidelines

## License

[MIT](LICENSE)

## Acknowledgments

- [StatsBomb](https://statsbomb.com/) for the Open Data initiative
- Built as a personal project exploring rule-based sports simulation with LLM integration
