# Forecasting Engine & Model Validation

## Forecast Horizons

Probabilistic research signals are generated for six discrete trading horizons:
* 1 Trading Day (1D)
* 3 Trading Days (3D)
* 5 Trading Days (5D / 1 Week)
* 10 Trading Days (10D / 2 Weeks)
* 20 Trading Days (20D / 1 Month)
* 60 Trading Days (60D / 1 Quarter)

## Model Ensembling & Walk-Forward Validation

For each horizon $h$, the system evaluates:
1. **Historical Conditional Baseline**: Directional probability based on historical analogue frequency.
2. **Logistic Regression**: Linear classifier trained on technical, ratio, and macro features up to date $T-1$.
3. **Random Forest Classifier**: Non-linear ensemble model trained with 50 estimators on historical records up to $T-1$.
4. **Ridge Regression**: Regularized linear model predicting expected return percentage.

No random shuffling is ever used during cross-validation.

## Confidence Scoring Framework

Each forecast receives a confidence score (0–100) based on:
1. **Sample Size**: Penalty for small historical analogue sample count (< 50).
2. **Model Agreement**: Boost for unanimous agreement (e.g. 3/3 models agreeing on direction).
3. **Score Alignment**: Boost when composite score is far from neutral 50.
4. **Volatility Penalty**: Penalty when operating in a high volatility regime.
