# Architecture Overview

The Precious Metals Market Intelligence Pipeline is built using a modular, decoupled software architecture designed for data integrity, strict temporal correctness (avoidance of look-ahead bias), and reproducibility.

## High-Level Architecture Flow

```
[ Data Sources ] (Yahoo Finance, Macro APIs, Synthetic Fallbacks)
       │
       ▼
[ Ingestion Layer ] (src/ingestion/ - MarketDataProvider & Abstraction)
       │
       ▼
[ Validation & Quality Layer ] (src/validation/ - DataValidator & Status Classification)
       │
       ▼
[ Storage Manager ] (src/storage/ - Scalable Raw/Processed Parquet & JSON Storage)
       │
       ├───────────────────────────────┬──────────────────────────────┐
       ▼                               ▼                              ▼
[ Technical Indicators ]     [ Gold/Silver Ratio ]        [ Macro Correlations ]
(src/indicators/)            (src/ratios/)                (src/macro/)
       │                               │                              │
       └───────────────────────────────┼──────────────────────────────┘
                                       ▼
                         [ Regime Classification ]
                         (src/regimes/ - Macro & Market)
                                       │
                                       ▼
                       [ Historical Analogue Engine ]
                       (src/patterns/ - Look-ahead-free Similarity)
                                       │
                                       ▼
                         [ Daily ML Feature Store ]
                         (src/features/ - Date T Features vs Future Targets)
                                       │
       ┌───────────────────────────────┴──────────────────────────────┐
       ▼                                                              ▼
[ Baseline ML Models ]                                    [ Probabilistic Engine ]
(src/models/ - Direction/Return)                          (src/forecasting/ - Conditional Stats)
       │                                                              │
       └───────────────────────────────┬──────────────────────────────┘
                                       ▼
                        [ Forecast Tracking & Eval ]
                        (src/evaluation/ - Actual vs Predicted)
                                       │
       ┌───────────────────────────────┴──────────────────────────────┐
       ▼                                                              ▼
[ Backtesting Engine ]                                    [ Multi-Factor Scoring ]
(src/backtesting/ - Cost/Slippage)                        (src/scoring/ - Transparent 0-100)
                                                                      │
                                                                      ▼
                                                          [ Alert Detector & Reports ]
                                                          (src/reporting/ - MD & JSON)
```

## System Principles

1. **Strict Temporal Separation**: Features on date T strictly exclude any information occurring after date T. Target metrics (`future_return_1d` to `future_return_60d`) are stored separately.
2. **Provider Abstraction**: Data ingestion is decoupled from specific APIs using the `MarketDataProvider` base class, supporting seamless provider replacement.
3. **Data Retention**: Raw observations are preserved permanently under `data/raw/YYYY/MM/YYYY-MM-DD/`. No historical dataset is ever overwritten.
4. **Explainability**: Black-box predictions are avoided. All composite scores, confidence ratings, and regime classifications provide explainable component breakdowns.
