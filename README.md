# Trading Bot (Rebuild Track)

This repository is now set up as a clean research sandbox. The goal is to give you
good structure and guardrails while still leaving enough open surface area for you to
design and implement the hard parts.

## What is in place right now

- API ingestion primitives for CoinMarketCap listings (`handlers/`, `data/`)
- Cache and request limit management
- Baseline LSTM training entrypoint (`main.py`)
- Research scaffolding for:
  - strategies (`src/strategies/`)
  - backtesting (`src/backtest/`)
  - risk sizing (`src/risk/`)
- Modern tooling config in `pyproject.toml` (`ruff`, `pytest`, `mypy`, `pre-commit`)

## Environment and package management

This repo uses `pyproject.toml` as the single dependency source of truth.
Use [uv](https://github.com/astral-sh/uv) for fast installs and reproducible workflows.

### Quick start

1. Install `uv` (one-time)
2. Sync dependencies:
   - `make install-dev`
3. Set environment:
   - `cp .env.example .env`
   - add your `API_KEY`
4. Run baseline script:
   - `make run`

### Common commands

- `make test` - run tests
- `make lint` - run lint checks
- `make format` - run formatter
- `make typecheck` - run mypy
- `make precommit` - run all pre-commit hooks

## Upcoming Plans

Implementation roadmap of what may be implemented next for this trading system, broadly speaking

### Phase 1: Data and problem framing

- Switch from single snapshot listing data to historical OHLCV bars.
- Define one clear prediction target:
  - next return direction, or
  - volatility bucket, or
  - expected risk-adjusted return.
- Build reproducible datasets to `parquet` with clear train/val/test date ranges.

### Phase 2: Baselines before complexity

- Implement simple strategy baselines first:
  - buy-and-hold
  - moving average crossover
  - momentum/reversion rules
- Add realistic frictions:
  - fees
  - slippage
  - delayed signal execution
- Evaluate with:
  - drawdown
  - Sharpe/Sortino
  - turnover
  - hit rate

### Phase 3: Traditional ML then sequence models

- Starting with feature-based ML:
  - logistic/linear models
  - tree boosting
- Compare strategy baselines with walk-forward validation.
- Only then move to LSTM/sequence models if they show incremental value.

### Phase 4: AI/LLM integration (optional)

- LLMs are usually better for unstructured inputs (news, filings, social text) than
  raw price prediction.
- A practical path:
  - derive sentiment/event features from text
  - merge with market features
  - keep the execution/risk pipeline model-agnostic

## AI vs traditional ML guidance

- Prefer traditional ML first when:
  - data is tabular/time-series and limited
  - interpretability matters
  - you want faster iteration
- Use deep/AI-heavy approaches when:
  - you have enough quality data
  - baseline models saturate
  - you can justify extra complexity with out-of-sample gains

## Complexity options (good challenge, not overkill)

- Walk-forward optimization and rolling retraining
- Regime filters (trend vs mean reversion)
- Volatility-targeted position sizing
- Portfolio constraints (max exposure, correlation caps)
- Monte Carlo stress tests on trade sequences

can add options instruments later, Black-Scholes or local-vol extensions become very relevant.

## Notes and cautions

- Current `main.py` is intentionally a baseline example and not production-grade.
- Avoid leakage: split by time, never random-shuffle future into training.
- Keep experiment tracking from day one (config + metrics + seed).

## Project hygiene targets

- Keep `pyproject.toml` as the only dependency declaration.
- Keep runtime artifacts out of git (`crypto_cache*.json`, `request_log.json`).
- Add tests for every new component before model complexity.
