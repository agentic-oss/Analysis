# Probabilistic Forecasting Models

The system evaluates probabilistic forward-looking research signals across multiple time horizons: 1D, 3D, 5D, 10D, 20D, and 60D.

## Model Framework

### 1. Baseline Machine Learning Models (`src/models/baseline.py`)
- **Linear Regression**: Predicts expected forward return percentage.
- **Random Forest Regressor / Classifier**: Captures non-linear feature interactions for return direction and expected gain.
- **Logistic Regression & Gradient Boosting Classifier**: Predicts directional probability ($\Pr(\text{Return} > 0)$).

### 2. Probabilistic Forward Engine (`src/forecasting/probabilistic.py`)
Calculates empirical conditional probability distributions by isolating historical days matching current momentum (RSI) and regime conditions:
- Positive Return Probability
- Negative Return Probability
- Expected & Median Return
- Expected Volatility & Historical Gain/Loss Boundaries

## Model Agreement & Signal Confidence

Model confidence (0–100) is determined dynamically based on:
1. Historical sample size in matched regime.
2. Degree of consensus across independent model algorithms.
3. Macro regime alignment.
4. Technical trend confirmation.
