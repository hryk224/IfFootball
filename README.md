# IfFootball

[日本語版はこちら](README.ja.md)

> _"What if Van Gaal had been dismissed after match week 29?"_

Simulate how managerial changes and player transfers might have altered a football team's season. IfFootball uses StatsBomb Open Data, rule-based causal simulation, and optional LLM-generated analysis. **No API key required to run — LLM is optional.**

<!-- Screenshot placeholder: replace with actual UI screenshot -->
<!-- ![IfFootball UI](docs/images/screenshot.png) -->

## Quick Start

```bash
git clone https://github.com/hryk224/IfFootball.git
cd IfFootball
uv sync --extra dev
uv run streamlit run app.py
```

Select **Manchester United**, enter **Louis van Gaal** as the current manager, set trigger week to **29**, and click **Run Simulation**. Results appear in ~2-5 minutes (StatsBomb API fetch on first run).

No `.env` file, no API key, no LLM setup needed for the basic experience.

## Example Output

Backtest: Manchester United — Van Gaal dismissal at week 29 (Premier League 2015-16, 20 simulation runs):

```
Branch A (no change):  12.2 points (mean)  ±3.5
Branch B (dismissed):  12.8 points (mean)  ±3.2
Delta (B - A):         +0.5 points

Top 3 impacted players:
  1. Michael Carrick     (impact: 0.683) — fatigue +0.19, trust +0.13
  2. Jesse Lingard       (impact: 0.595) — form -0.09, fatigue -0.15
  3. Ander Herrera       (impact: 0.584) — form -0.12, understanding -0.25
```

The simulation shows a slight positive trend (+0.5 points) but within statistical uncertainty. All players experienced a tactical understanding reset (-0.25), reflecting the adaptation period after a managerial change. Full analysis in the Streamlit UI includes radar charts and structured reports.

## Disclaimer

IfFootball is a **what-if simulation tool**, not a prediction engine. Results represent theoretical outcomes under specified rule-based parameters, not predictions of actual events. All simulation parameters are provisional and documented with their rationale in [simulation-rules.md](docs/simulation-rules.md#known-limitations). The authors are not responsible for any decisions made based on this software or its outputs.

## Key Features

- **Manager Change** — Simulate mid-season dismissal with tactical profile reset, squad trust recalibration, and adaptation curves
- **Player Transfer** _(Experimental)_ — Add a player to the squad with role-based trust initialization
- **A/B Comparison** — Poisson match model, weekly state updates (fatigue, trust, tactical understanding), turning point detection with cascade event tracking
- **Visualization** — Team and player radar charts comparing Branch A/B outcomes
- **LLM Reports** — Structured comparison reports with data / analysis / hypothesis labels and source classification
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

IfFootball uses [StatsBomb Open Data](https://github.com/statsbomb/open-data) exclusively. All metrics and terminology follow StatsBomb definitions. No scraping is involved. Attribution and usage terms follow the [StatsBomb Open Data license](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf).

Currently supported competitions (from `config/targets.toml`):

| Competition    | Season  | Teams                                                                                              |
| -------------- | ------- | -------------------------------------------------------------------------------------------------- |
| Premier League | 2015-16 | Manchester United, Manchester City, Arsenal, Liverpool, Chelsea, Tottenham Hotspur, Leicester City |
| La Liga        | 2015-16 | Real Madrid, Barcelona, Atletico Madrid                                                            |

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

**Model override:** Each provider has its own model env var (`OPENAI_MODEL`, `ANTHROPIC_MODEL`, `GEMINI_MODEL`, `GROQ_MODEL`). If unset, provider defaults are used.

**OpenAI-compatible APIs:** Set `OPENAI_BASE_URL` to use any OpenAI-compatible endpoint (e.g., Azure OpenAI, local inference servers).

**Note:** When LLM is enabled, scenario data (team names, player names, simulation results) is sent to the configured provider. Data handling follows the provider's policy.

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
- **Data / Analysis / Hypothesis labels** — All outputs distinguish between simulation data, model-derived analysis, and speculative hypotheses

## Tests

```bash
uv run python -m pytest        # 554+ tests
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
