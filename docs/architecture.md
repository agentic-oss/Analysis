# Architecture Overview

The Precious Metals Pipeline is designed as a modular, decoupled data pipeline for processing gold, silver, currency, rate, commodity, equity, and volatility time-series data.

## Pipeline Layers

1. **Ingestion & Abstraction Layer (`src/ingestion/`)**:
   - `MarketDataProvider` abstract base class.
   - `YahooFinanceProvider` with exponential backoff and rate limit handling.
   - `MockMarketDataProvider` for offline fallback/testing.
   - `enrich_indian_conversions` for USD/oz to INR/10g and INR/kg conversions.

2. **Data Validation Layer (`src/validation/`)**:
   - Checks OHLC bounds, negative prices, missing/duplicate timestamps, and price jumps.
   - Classifies records as `valid`, `suspicious`, or `invalid`.

3. **Analysis & Indicator Layer (`src/indicators/`, `src/metals/`, `src/ratios/`)**:
   - Moving averages, momentum, volatility, volume, price structure.
   - Rolling cross-asset correlations.
   - Gold/Silver ratio relative value z-score and percentiles.

4. **Regime & Analogue Layer (`src/regimes/`, `src/patterns/`)**:
   - Macro and market trend/volatility regimes.
   - Multi-vector historical analogue similarity engine (Euclidean distance on standardized features, look-ahead bias free).

5. **Feature Store & Forecasting (`src/features/`, `src/models/`, `src/forecasting/`)**:
   - Feature store generation with `future_return_Nd` targets.
   - Walk-forward validation across Logistic Regression, Random Forest, and Gradient Boosting.
   - Probabilistic forward return signals.

6. **Scoring, Backtesting, Evaluation & Reporting (`src/scoring/`, `src/backtesting/`, `src/evaluation/`, `src/reporting/`)**:
   - Transparent composite 0-100 scores.
   - Strategy backtester with transaction costs.
   - Performance tracking of historical forecasts.
   - Daily Markdown (`reports/daily/YYYY-MM-DD.md`) and JSON report generation.
