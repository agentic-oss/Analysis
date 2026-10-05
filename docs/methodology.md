# Research Methodology & Research Disclaimer

## Research Principles

1. **Probabilistic Research Signals**:
   - Price predictions are never framed as guarantees or definitive claims.
   - All forecasts represent historical conditional distributions under similar technical, macro, and relative value conditions.

2. **Temporal Integrity & Look-Ahead Bias Prevention**:
   - Feature vectors available on date T only use information available up to date T.
   - Target metrics (`future_return_1d`, `future_return_10d`, etc.) are computed separately and never fed back into historical feature transformations.

3. **Multi-Factor Explainability**:
   - High composite scores require alignment across technical, macro, momentum, volatility, and relative value components.
   - Explanations for confidence scores are explicitly enumerated in daily reports and JSON outputs.
