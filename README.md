# IfFootball

Football "what-if" simulator using StatsBomb data. Rule-based causal engine with LLM-generated reports.

## What It Does

- **Manager change simulation** — "What if the manager was dismissed at week N?" Compare Branch A (no change) vs Branch B (trigger applied) across N simulation runs
- **Transfer trigger** — "What if Player X joined the squad?" Add a player with role-based trust initialization
- **A/B comparison** — Poisson match model, weekly state updates (fatigue, trust, tactical understanding), turning point detection with cascade event tracking
- **Visualization** — Team and player radar charts comparing Branch A/B
- **Reports** — Structured comparison reports with fact/analysis/hypothesis labels
- **Streamlit UI** — Single-page app for end-to-end simulation

## Setup

```bash
cp .env.example .env  # configure API keys (see .env.example)
uv sync --extra dev
npm install
```

To enable LLM-generated reports (optional):

```bash
uv sync --extra dev --extra llm
```

Without `--extra llm`, the app runs in data-only mode.

## Usage

### Streamlit UI

```bash
streamlit run app.py
```

### Backtest Script

```bash
uv run python scripts/backtest_van_gaal.py
```

## Tests

```bash
uv run python -m pytest
uv run python -m ruff check .
uv run python -m mypy .
```

## Documentation

- [Simulation Rules](docs/simulation-rules.md) — All configurable parameters with definitions and rationale
- [Changelog](docs/CHANGELOG.md) — Per-milestone change history
