# System Architecture

The pipeline uses a modular, decoupled architecture where each layer serves a distinct responsibility.

```
[ Data Sources (yfinance / Synthetic) ]
                  │
                  ▼
         [ Ingestion Layer ] (Rate limits, retries, unit conversions USD->INR)
                  │
                  ▼
         [ Validation Layer ] (OHLC checks, abnormal jump classification)
                  │
                  ▼
         [ Storage Layer ] (Parquet / CSV / JSON structured files)
                  │
                  ├──► [ Indicators & Macro Analysis ]
                  ├──► [ Gold/Silver Ratio Analysis ]
                  ├──► [ Regime Classification ]
                  ├──► [ Historical Analogue Engine ]
                  │
                  ▼
         [ Feature Store Builder ]
                  │
                  ▼
         [ Forecasting Models & Backtester ]
                  │
                  ▼
         [ Reporting & Alert Generator ] (Markdown & JSON Outputs)
```
