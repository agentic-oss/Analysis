# Pipeline System Architecture

## Architecture Overview

The system operates as an end-to-end modular pipeline executed daily:

1. **Ingestion Layer (`src/ingestion/`)**: Fetches market data using `YFinanceProvider` with retries and exponential backoff, or `SyntheticProvider` for deterministic fallback testing.
2. **Validation Layer (`src/validation/`)**: Validates OHLC relationships, checks price jumps, non-positive values, and generates `data/quality/YYYY-MM-DD.json`.
3. **Storage Layer (`src/storage/`)**: Manages raw, processed, feature, and report files in Parquet, CSV, and JSON formats.
4. **Technical Analysis (`src/indicators/`)**: Computes SMAs (5, 10, 20, 50, 100, 200), EMAs (9, 20, 50, 200), RSI, MACD, ATR, Bollinger Bands, Volume metrics, and Price structure signals.
5. **Macro Engine (`src/macro/`)**: Evaluates DXY, Treasury yields, Oil, Equities, and Volatility to classify macro regimes (Inflationary, Disinflationary, Risk-On, Risk-Off, Monetary Stance).
6. **Metals & Relative Value (`src/metals/`, `src/ratios/`)**: Analyzes Gold/Silver ratio (z-score, percentiles, mean-reversion signals) and cross-asset correlations.
7. **Regime & Pattern Matching (`src/regimes/`, `src/patterns/`)**: Identifies trend/volatility market regimes and finds top N historical analogues.
8. **Feature Store & Scoring (`src/features/`, `src/scoring/`)**: Constructs ML feature datasets and computes transparent 0–100 multi-factor composite scores.
9. **Forecasting & Modeling (`src/models/`, `src/forecasting/`)**: Train walk-forward models and blend with analogue conditional stats for probabilistic signals.
10. **Backtesting & Evaluation (`src/backtesting/`, `src/evaluation/`)**: Backtests signal strategies accounting for transaction costs/slippage and evaluates historical forecast accuracy.
11. **Reporting & Alerts (`src/reporting/`)**: Generates Markdown reports, daily JSON outputs, and alert JSONs.
