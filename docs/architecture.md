# System Architecture Specification

## Overview
The Precious Metals Research Pipeline is structured into decoupled, single-responsibility Python modules under `src/`.

### Core Layers
1. **Ingestion Layer (`src/ingestion/`):** Fetches raw time-series market data with retry logic and fallback synthetic generators.
2. **Validation Layer (`src/validation/`):** Validates OHLC relationships, jumps, and timestamps without discarding data.
3. **Storage Layer (`src/storage/`):** Manages Parquet, CSV, and JSON data persistence with deduplication.
4. **Analytics Layer (`src/indicators/`, `src/metals/`, `src/ratios/`, `src/macro/`, `src/regimes/`):** Computes quantitative technicals, ratio statistics, macro regimes, and market regimes.
5. **Machine Learning & Forecasting Layer (`src/patterns/`, `src/forecasting/`, `src/features/`, `src/models/`, `src/backtesting/`, `src/evaluation/`):** Historical analogue engine, forward probabilities, feature store, walk-forward models, backtester, and performance evaluation tracker.
6. **Reporting Layer (`src/scoring/`, `src/reporting/`, `src/pipeline/`):** Composite scoring, alert detector, daily Markdown/JSON report generators, and orchestrator runner.
