# Contributing

Thank you for your interest in contributing to IfFootball.

## Getting Started

```bash
git clone https://github.com/hryk224/IfFootball.git
cd IfFootball
uv sync --extra dev
npm install
```

LLM provider SDKs (OpenAI, Anthropic, Gemini, Groq) are included in core dependencies. Without an API key configured, the app runs in data-only mode.

## Development Workflow

1. Create a branch from `main`
2. Make your changes
3. Run checks before committing:

```bash
uv run python -m pytest
uv run python -m ruff check .
uv run python -m mypy .
npm run format:md
```

4. Write a commit message following [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):

```
<type>[optional scope]: <subject>
```

Types: `feat` / `fix` / `docs` / `style` / `refactor` / `perf` / `test` / `chore` / `build` / `ci` / `revert`

5. Open a pull request against `main`

## Architecture

- **Rule-based core** -- Simulation logic is defined by config files (`config/simulation_rules/`), not by LLM
- **StatsBomb alignment** -- All metrics follow StatsBomb definitions and terminology
- **LLM as explanation layer** -- LLM is limited to knowledge queries, action explanations, and report generation. It does not make simulation decisions
- **Reproducibility** -- Seeded RNG for deterministic results. Same seed = same output

See [docs/simulation-rules.md](docs/simulation-rules.md) for parameter definitions and rationale.

## Project Structure

```
src/iffootball/
  agents/         # Domain models (PlayerAgent, ManagerAgent, etc.)
  collectors/     # StatsBomb data retrieval
  converters/     # Stats -> agent attribute conversion
  llm/            # LLM client, providers, prompts
  simulation/     # Engine, comparison, cascade tracking
  storage/        # SQLite persistence
  visualization/  # Radar charts, player impact
config/           # TOML simulation rules
prompts/          # LLM system prompts
app.py            # Streamlit UI
```

## Adding a New LLM Provider

1. Create `src/iffootball/llm/providers/<name>_provider.py`
2. Implement the `LLMClient` protocol (single `complete()` method)
3. Register in `src/iffootball/llm/providers/__init__.py`
4. Add the SDK to `pyproject.toml` `[project].dependencies`

## Supply Chain Policy

- Pin all dependency versions exactly (no `^`, `~`, `>=`)
- New dependencies require maintainer approval
- See `CLAUDE.md` for full dependency management rules

## Data Sources

- **StatsBomb Open Data** is the only supported data source
- No scraping of any kind
- New data sources require explicit approval

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
