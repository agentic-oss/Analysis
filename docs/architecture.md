# System Architecture

The Gold & Silver Market Data, Deep Analysis, Historical Research, and Forecasting Pipeline is built around a modular Python architecture designed for reproducibility, scalability, and strict temporal correctness.

```
+-----------------------------------------------------------------------------------+
|                                  DATA SOURCES                                     |
|           (yfinance / Yahoo Finance, FX Rates, Macro & Economic Inputs)           |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                           INGESTION & DATA VALIDATION                             |
|          - MarketDataProvider abstraction with retries & exponential backoff    |
|          - OHLC integrity checks, price jump classification                       |
|          - Quality output: data/quality/YYYY-MM-DD.json                           |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        STORAGE & CURRENCY ENRICHMENT                              |
|          - USD/oz -> INR/10g & INR/kg conversions with timestamped metadata      |
|          - Raw data: data/raw/YYYY/MM/YYYY-MM-DD/                                 |
|          - Processed Parquet storage: data/processed/{category}/                  |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                    INDICATORS, MACRO & REGIME ANALYSIS                            |
|          - Moving Averages, RSI, MACD, ATR, Bollinger Bands, OBV                   |
|          - Rolling Cross-Market Correlations (DXY, Yields, Oil, Equities, VIX)     |
|          - Rule-based Macro Regime & Market Regime Classification                 |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                HISTORICAL ANALOGUE ENGINE & FEATURE STORE                         |
|          - Multi-factor Euclidean similarity search over historical states        |
|          - Conditional forward statistics (1D, 3D, 5D, 10D, 20D, 60D)             |
|          - Temporal look-ahead protection & feature dataset generation            |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                 TRANSPARENT SCORING, MODELS & EVALUATION                          |
|          - Multi-subscore composite score (Technical, Macro, Momentum, etc.)      |
|          - Time-Series Walk-Forward validation & Backtesting                       |
|          - Predicted vs Actual outcome performance evaluation                       |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                     REPORTING, ALERTS & AUTOMATION                                |
|          - Daily Markdown research report: reports/daily/YYYY-MM-DD.md            |
|          - Machine-readable daily JSON: data/analysis/daily/YYYY-MM-DD.json       |
|          - Alerts output: data/analysis/alerts/YYYY-MM-DD.json                    |
|          - GitHub Actions automated daily execution                               |
+-----------------------------------------------------------------------------------+
```
