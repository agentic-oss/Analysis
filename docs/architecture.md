# System Architecture

The pipeline follows a modular end-to-end quantitative architecture:

1. **Ingestion Layer (`src/ingestion`)**: Interacts with data providers (`MarketDataProvider` interface) with built-in retries, backoff, and unit converters.
2. **Validation Layer (`src/validation`)**: Validates OHLC relationships, bounds, and abnormal daily movements, writing output quality reports.
3. **Storage Layer (`src/storage`)**: Manages reading and writing Parquet, CSV, and JSON files across data subdirectories.
4. **Analytics & Indicators (`src/indicators`, `src/metals`, `src/ratios`)**: Computes technical indicators, multi-asset rolling correlations, and Gold/Silver relative value Z-scores.
5. **Regimes & Analogue Search (`src/macro`, `src/regimes`, `src/patterns`)**: Rule-based macro/market regime classification and weighted Euclidean distance analogue matching.
6. **Features & Modeling (`src/features`, `src/models`, `src/forecasting`, `src/backtesting`)**: ML feature store, expanding-window walk-forward model evaluation, confidence scoring, and event-driven backtesting.
7. **Evaluation, Alerts & Reports (`src/evaluation`, `src/scoring`, `src/reporting`)**: Out-of-sample forecast evaluation, alert generation, and daily Markdown/JSON research reports.
