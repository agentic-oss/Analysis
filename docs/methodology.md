# Research & Forecasting Methodology

## Core Philosophy

The primary objective of this system is to serve as an objective, continuously growing precious-metals research database rather than an unverified prediction engine.

All model predictions, forward returns, and historical analogue statistics are explicitly defined as **probabilistic research signals**.

## Information Flow Hierarchy

```
Observed Market Data (OHLCV, FX, Rates)
  ↓
Derived Technical & Macro Indicators (MAs, RSI, Correlations)
  ↓
Macro & Market Regime Classification (Rule-Based Framework)
  ↓
Historical Analogue Search (Look-Ahead-Bias-Free Vector Matching)
  ↓
Statistical & ML Model Inference (Expanding-Window Validation)
  ↓
Probabilistic Forward Research Signal (Conditional Distributions)
```

## Temporal Integrity & Prevention of Look-Ahead Bias

To prevent data leakage:
1. All feature data on date T is computed exclusively using information available on or before date T.
2. Target variables (`future_return_1d` to `future_return_60d`) represent strictly future shifts:
   $$\text{future\_return\_}h\text{d}_t = \frac{P_{t+h} - P_t}{P_t}$$
3. When searching for historical analogues on date T, the target date itself is excluded from the candidate search set.
