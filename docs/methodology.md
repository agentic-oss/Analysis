# Research & Forecasting Methodology

## Probabilistic Signal Principle

All market forecasts in this repository are strictly expressed as **probabilistic research signals** derived from historical analogue distributions and walk-forward statistical models.

**No model output represents guaranteed future prices.**

## Temporal Correctness & Look-Ahead Protection

1. **Information Isolation**: Any observation or signal at date $T$ uses only information available up to date $T$.
2. **Shifted Targets**: Target variables ($1\text{D}, 3\text{D}, 5\text{D}, 10\text{D}, 20\text{D}, 60\text{D}$ forward returns) are strictly stored in separate target columns shifted forward in time.
3. **Historical Analogue Filtering**: When searching for historical analogues for date $T$, candidate days are strictly limited to historical records prior to $T$.

## Transparent Scoring System

All composite scores (0–100 scale) are fully explainable and derived from weighted component scores:

* **Technical Score (25%)**: Moving average alignment, breakout/breakdown signals.
* **Macro Score (20%)**: Risk stance (VIX/Equities) and Monetary stance (Yields/DXY).
* **Momentum Score (15%)**: RSI 14 and MACD histogram state.
* **Volatility Score (10%)**: Rolling 20-day annualized volatility.
* **Relative Value Score (15%)**: Gold/Silver ratio Z-score and mean-reversion potential.
* **Historical Pattern Score (15%)**: Win rate among closest historical analogues.
