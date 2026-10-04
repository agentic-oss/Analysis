# Methodology & Philosophy

## Research Philosophy

The primary purpose of this system is to provide rigorous, quantitative historical research and probabilistic signals.

Key Principles:
1. **No Certainty Claims**: Financial markets are inherently stochastic. Predictions are framed strictly as historical conditional probability distributions.
2. **Temporal Integrity**: Features on date $T$ strictly contain information available by date $T$. Target variables $T+h$ are calculated without leaking into feature rows.
3. **Transparent Explainability**: Sub-scores and macro regimes are rule-based and fully explainable rather than black-box models.
4. **Continuously Growing Database**: Forecasts generated on date $T$ are stored alongside actual realized outcomes to continuously evaluate and improve baseline models over time.
