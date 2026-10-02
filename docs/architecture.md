# System Architecture & Pipeline Flow

## High-Level Architecture

The Precious Metals Market Data, Research, and Forecasting Pipeline is built with a modular, package-oriented Python architecture.

```
                  ┌──────────────────────┐
                  │ Market Data Sources  │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  MarketDataProvider  │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Data Validation &    │
                  │ Unit/FX Conversion   │
                  └──────────┬───────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
┌───────────────────────┐         ┌───────────────────────┐
│ Technical Indicators  │         │   Macro & Relative    │
│    (SMA/RSI/MACD)     │         │   Value Ratio Analysis│
└───────────┬───────────┘         └───────────┬───────────┘
            │                                 │
            └────────────────┬────────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Macro & Market       │
                  │ Regime Classifiers   │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Historical Analogue  │
                  │ & Feature Store      │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Probabilistic Signal │
                  │ Models & Scoring     │
                  └──────────┬───────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
┌───────────────────────┐         ┌───────────────────────┐
│ Daily Markdown Report │         │ Machine-Readable JSON │
│ reports/daily/*.md    │         │ data/analysis/*.json  │
└───────────────────────┘         └───────────────────────┘
```

## Modular Package Breakdown

* `src/ingestion`: Data provider abstractions and yfinance provider implementation with exponential retries and FX conversions.
* `src/validation`: Quality validator checking missing dates, duplicates, OHLC integrity, negative/zero prices, and stale values.
* `src/storage`: Central storage manager handling Parquet, JSON, and CSV persistences.
* `src/indicators`: Technical analysis calculations (SMA, EMA, RSI, MACD, ATR, Bollinger Bands, Volume, Price Structure).
* `src/metals`: Gold and Silver multi-factor correlation engines across 20, 60, 120, 252 days.
* `src/ratios`: Gold/Silver ratio relative value, Z-scores, and mean reversion behavior.
* `src/regimes`: Transparent rule-based Macro Regime and Market Regime models.
* `src/patterns`: Multi-metric historical analogue similarity search without look-ahead bias.
* `src/features`: ML-ready daily feature store with target horizon separation.
* `src/scoring`: Transparent 0-100 composite scoring engine and alert engine.
* `src/forecasting`: Multi-horizon probabilistic signals and confidence framework.
* `src/models`: Baseline time-series models (Conditional Probability, Logistic Regression, Random Forest) with walk-forward validation.
* `src/backtesting`: Strategy backtester with transaction costs, slippage, and position sizing.
* `src/evaluation`: Historical forecast performance evaluator comparing predictions to actual outcomes.
* `src/reporting`: Markdown and JSON daily research report builder.
* `src/pipeline`: End-to-end pipeline runner (`runner.py`).
