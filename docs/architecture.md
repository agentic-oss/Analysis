# Architecture & Design

## System Architecture Overview

The Precious Metals Market Data, Deep Analysis, Research, and Forecasting Pipeline is built using a modular, decoupled architecture in Python 3.12.

```
[ Data Sources ] -> [ Data Ingestion ] -> [ Data Validation ] -> [ Storage Engine ]
                                                                      |
    [ ML Feature Store ] <- [ Technical & Macro Analytics ] <---------+
            |
            v
 [ Historical Analogue Engine ] -> [ Forward Probability Engine ]
            |
            v
 [ Scoring & Alerts ] -> [ Daily Reports (MD/JSON) ] -> [ Model Performance Monitoring ]
```

## Key Principles

1. **Provider Abstraction:** `MarketDataProvider` decouples ingestion logic from concrete market data providers (e.g., yfinance, mock/synthetic fallback).
2. **Temporal Integrity:** Strict prevention of look-ahead bias across all feature calculations, analogue searches, and machine learning targets.
3. **Immutability & Retention:** Raw snapshot files are saved per day in `data/raw/YYYY/MM/YYYY-MM-DD/`, while continuous processed time-series are stored as append-only Parquet files in `data/processed/`.
4. **Explainable Intelligence:** Multi-factor composite scoring and rule-based regime classifications are fully transparent with individual component sub-scores stored alongside final metrics.
