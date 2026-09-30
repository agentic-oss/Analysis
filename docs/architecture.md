# Pipeline Architecture

The platform uses a modular architecture under `src/`:

- `ingestion/`: Provider abstraction (`MarketDataProvider`), Yahoo Finance implementation, and currency/unit converters.
- `validation/`: Batch validator classifying records into valid/suspicious/invalid and generating quality reports.
- `storage/`: Utilities for Parquet, CSV, and JSON data persistence.
- `indicators/`: Technical indicators and price structure metrics calculations.
- `metals/`, `ratios/`, `macro/`: Gold/Silver asset analysis, Gold/Silver ratio metrics, cross-market correlations, and calendar event tracking.
- `regimes/`: Explainable macro and market regime classifiers.
- `patterns/`: Historical analogue search engine and forward target distribution statistics.
- `features/`: Daily ML feature builder with strict look-ahead protection.
- `models/`: Walk-forward baseline forecasting models (Historical Base Rate, Logistic Regression, Random Forest, Gradient Boosting).
- `forecasting/`, `scoring/`: Probabilistic signal generation, 0-100 factor scoring, confidence framework, and alert triggers.
- `backtesting/`, `evaluation/`: Strategy backtest engine and forecast performance tracking against realized outcomes.
- `reporting/`, `pipeline/`: Markdown report generator, JSON report generator, and daily pipeline runner script.
