# System Architecture Overview

The system is structured as a modular daily data processing, analysis, machine learning feature store, and research reporting pipeline for Gold and Silver.

```
+------------------+     +-------------------+     +------------------+
| Market Data      | --> | Data Validation   | --> | Data Storage     |
| Providers        |     | & Quality Engine  |     | (Parquet/CSV)    |
+------------------+     +-------------------+     +------------------+
                                                            |
                                                            v
+------------------+     +-------------------+     +------------------+
| Historical       | <-- | Feature Store     | <-- | Technical, Ratio |
| Analogue Engine  |     | & Target Builder  |     | & Macro Engines  |
+------------------+     +-------------------+     +------------------+
         |                                                  |
         v                                                  v
+------------------+     +-------------------+     +------------------+
| Probabilistic    |     | Composite Scoring |     | Reports & Alerts |
| Signals & ML     | --> | & Alert Engine    | --> | (Markdown/JSON)  |
+------------------+     +-------------------+     +------------------+
```

## Core Modules

* `src/ingestion/`: Ingests market data with retries, backoff, and synthetic offline fallback.
* `src/validation/`: Classifies OHLC observations as `valid`, `suspicious`, or `invalid`.
* `src/storage/`: Parquet and CSV/JSON storage manager.
* `src/indicators/`: Full suite of technical indicators (MAs, Momentum, Volatility, Volume).
* `src/metals/`: Indian market unit converter (INR/10g, INR/kg, USD/INR conversion tracking).
* `src/ratios/`: Gold/Silver ratio, relative value spread, and cross-market correlation engine.
* `src/macro/` & `src/regimes/`: Rule-based macro regime and trend/volatility market regime classifiers.
* `src/patterns/`: Multi-factor historical analogue similarity search and forward return statistical engine.
* `src/features/`: ML daily feature store with look-ahead protection and event features.
* `src/scoring/` & `src/forecasting/`: Composite factor scoring, alerts, and probabilistic return signals.
* `src/models/` & `src/backtesting/`: Walk-forward validation models, trading execution simulator, and evaluation tracker.
* `src/reporting/` & `src/pipeline/`: Markdown research report generator, JSON builder, and automated runner script.
