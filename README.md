# IfFootball

Football "what-if" simulator using StatsBomb data. Rule-based causal engine with LLM-generated reports.

## Setup

```bash
cp .env.example .env  # configure API keys (see .env.example)
uv sync --extra dev
npm install
```

## Tests

```bash
uv run python -m pytest
uv run python -m ruff check .
uv run python -m mypy .
```
