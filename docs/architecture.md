# System Architecture

The Gold & Silver Market Data, Deep Analysis, Historical Research, and Forecasting Pipeline consists of modular Python packages:

1. **Ingestion (`src/ingestion/`)**: Retries, rate limits, market data fetching.
2. **Validation (`src/validation/`)**: Data quality audits and zero/jump classifications.
3. **Storage (`src/storage/`)**: Parquet & JSON persistence layer.
4. **Indicators (`src/indicators/`)**: Technical indicators calculation.
5. **Metals & Macro (`src/metals/`, `src/regimes/`)**: Rolling driver correlations and macro regime detection.
6. **Ratios (`src/ratios/`)**: Gold/Silver ratio and Indian market price representations.
7. **Patterns & Forecasting (`src/patterns/`, `src/forecasting/`)**: Analogue search & forward probability distributions.
8. **Features & Models (`src/features/`, `src/models/`, `src/backtesting/`)**: Target-isolated ML dataset builder, walk-forward validation models, and backtester.
9. **Scoring, Evaluation & Reporting (`src/scoring/`, `src/evaluation/`, `src/reporting/`, `src/pipeline/`)**: Transparent scores, forecast calibration tracking, daily Markdown/JSON reports.
