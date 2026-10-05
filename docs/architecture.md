# Pipeline Architecture Overview

The system is structured as a modular, decoupled data pipeline:

1. **Ingestion Layer (`src/ingestion/`)**:
   - Fetches prices across metals, equities, currencies, yields, and commodities using `MarketDataProvider` implementations.
   - Standardizes timezones, symbols, and performs unit/currency conversions.

2. **Validation Layer (`src/validation/`)**:
   - Inspects incoming batches for missing fields, price anomalies, zero/negative values, and invalid OHLC relationships.

3. **Storage Layer (`src/storage/`)**:
   - Persists historical datasets in Parquet, CSV, and JSON formats without overwriting historical records.

4. **Technical & Macro Analysis Layer (`src/indicators/`, `src/macro/`, `src/metals/`, `src/ratios/`, `src/regimes/`)**:
   - Computes technical indicators, cross-market rolling correlations, ratio statistics, and detects macro/market regimes.

5. **Historical Analogue & ML Feature Layer (`src/patterns/`, `src/features/`, `src/scoring/`)**:
   - Constructs daily ML feature vectors with future targets (strictly separated).
   - Searches historical feature space for nearest matching market states.
   - Calculates component-wise and composite research scores (0-100).

6. **Forecasting, Evaluation, & Backtesting Layer (`src/forecasting/`, `src/models/`, `src/evaluation/`, `src/backtesting/`)**:
   - Runs baseline ML models using walk-forward validation.
   - Generates probabilistic return distributions and explainable confidence scores.
   - Evaluates past predictions against realized outcomes and tracks model performance.

7. **Reporting Layer (`src/reporting/`, `src/pipeline/`)**:
   - Outputs daily Markdown reports, JSON datasets, and alerts.
