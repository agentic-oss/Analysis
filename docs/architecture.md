# System Architecture

## Overview
The Precious Metals Research Database and Forecasting Pipeline is built using a modular, decoupled Python architecture. The system processes market observations through structured stages: Ingestion -> Validation -> Storage -> Indicator & Macro Analysis -> Regime Classification -> Analogue Search -> Scoring & Probabilistic Forecasting -> Reporting.

```
[ Data Ingestion ] ---> [ Data Validation ] ---> [ Storage Manager ]
                                                      |
    +-----------------+-------------------------------+
    |                 |                 |
[ Indicators ]  [ Macro Analysis ]  [ GSR & Ratios ]
    |                 |                 |
    +-----------------+-----------------+
                      |
           [ Regime Classification ]
                      |
           [ Historical Analogues ]
                      |
    +-----------------+-----------------+
    |                 |                 |
[ ML Features ]  [ Score/Confidence ] [ Forecast Engine ]
                      |                 |
            [ Daily Reports ]   [ Forecast Tracker ]
```

## Core Subsystems
1. **Ingestion & Provider Abstraction (`src/ingestion/`)**: Encapsulates data fetching via `MarketDataProvider` interface, including retries, rate-limiting, backoff, and synthetic offline fallback. Converts spot gold/silver prices into Indian units (`INR/10g` and `INR/kg`).
2. **Validation (`src/validation/`)**: Enforces OHLC sanity checks, detects price jumps (>15%), flags suspicious data, and produces daily quality logs (`data/quality/YYYY-MM-DD.json`).
3. **Storage (`src/storage/`)**: Manages immutable raw storage and processed time-series files in Parquet, CSV, and JSON formats.
4. **Analytics Engines (`src/indicators/`, `src/macro/`, `src/metals/`, `src/ratios/`)**: Computes technical indicators, cross-asset correlations, macro summaries, and Gold/Silver ratio metrics.
5. **Pattern Engine (`src/patterns/`)**: Matches today's observation vector against historical states without future information leakage.
6. **Probabilistic Forecasting & Scoring (`src/scoring/`, `src/forecasting/`)**: Builds transparent 0-100 pillar scores, composite confidence scores, alerts, and 1D–60D empirical forward statistics.
7. **Reporting & Evaluation (`src/reporting/`, `src/evaluation/`, `src/pipeline/`)**: Generates Markdown research reports, daily JSON analysis, and updates forecast accuracy tracking.
