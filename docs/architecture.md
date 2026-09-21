# Architecture Overview

The **Gold & Silver Market Data, Deep Analysis, Historical Research, and Forecasting Pipeline** is an automated end-to-end quantitative research platform designed for precious metals research with an Indian-market focus.

## System Architecture

```
[ Data Ingestion Layer ] ──► [ Data Validation Layer ] ──► [ Data Storage Layer ]
        │                             │                            │
        ▼                             ▼                            ▼
[ Technical Indicators ]   [ Macro & Correlation ]   [ Relative Value / Ratios ]
        │                             │                            │
        └─────────────────────────────┼────────────────────────────┘
                                      │
                                      ▼
                        [ Regime & Analogue Search ]
                                      │
                                      ▼
                       [ Probabilistic Forecasting ]
                                      │
                                      ▼
                   [ ML Feature Store & Baseline Models ]
                                      │
                                      ▼
                      [ Backtest & Scoring Engine ]
                                      │
                                      ▼
                         [ Daily Research Reports ]
```

## Modular Structure

- `src/ingestion/`: Market data providers (Yahoo, Mock fallbacks), currency/unit conversions (USD/oz to INR/10g, INR/kg).
- `src/validation/`: Batch QA validator checking OHLC geometry, negative/zero prices, jumps, volume, timestamps.
- `src/storage/`: Multi-format storage manager (Parquet, CSV, JSON).
- `src/indicators/`: Full technical indicator suite (MAs, EMAs, RSI, MACD, Stoch RSI, ATR, Bollinger Bands, Volume, Distances).
- `src/macro/`: Cross-market rolling correlations (DXY, Treasuries, Oil, Equities, VIX).
- `src/metals/`: Gold and Silver specific structural analysis.
- `src/ratios/`: Gold/Silver ratio relative value, Z-score, percentiles, mean reversion analysis.
- `src/regimes/`: Transparent rule-based Macro Regime and Market Regime classifiers.
- `src/patterns/`: Historical analogue search engine using normalized feature state vector similarity.
- `src/forecasting/`: Multi-horizon probabilistic research signal generator.
- `src/features/`: ML-ready daily feature store with look-ahead protection and separate target datasets.
- `src/models/`: Baseline machine learning models (Historical conditional probability, Logistic Regression, Random Forest, Gradient Boosting) using walk-forward validation.
- `src/evaluation/`: Historical prediction outcome logging, realized return matching, performance drift monitoring.
- `src/backtesting/`: Reusable signal backtesting engine with transaction friction and position sizing.
- `src/scoring/`: Transparent 0-100 multi-factor composite scoring and anomaly threshold alerts.
- `src/reporting/`: Automated daily Markdown report and JSON analysis generator.
- `src/pipeline/runner.py`: Command-line orchestrator for daily automated execution.
