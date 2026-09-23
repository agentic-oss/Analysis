import pytest
import os
import pandas as pd
import numpy as np
from src.forecasting.engine import ForecasterPipeline
from src.backtesting.engine import StrategyBacktester
from src.evaluation.tracker import ForecastEvaluator

def test_forecaster_pipeline():
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    prices = 2000.0 + np.cumsum(np.random.normal(0, 5, 100))
    df = pd.DataFrame({
        "timestamp": dates.strftime("%Y-%m-%d"),
        "close": prices,
        "rsi_14": np.random.uniform(30, 70, 100),
        "volatility_20d": np.random.uniform(10, 20, 100)
    })
    for h in [1, 5]:
        df[f"target_future_return_{h}d"] = (df["close"].shift(-h) - df["close"]) / df["close"] * 100.0
        df[f"target_future_direction_{h}d"] = (df[f"target_future_return_{h}d"] > 0).astype(int)

    pipeline = ForecasterPipeline(horizons=[1, 5])
    forecasts = pipeline.generate_all_forecasts(df, feature_cols=["rsi_14", "volatility_20d"])
    assert "1d" in forecasts
    assert "probability_positive" in forecasts["1d"]
    assert "expected_return" in forecasts["1d"]

def test_strategy_backtester():
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    # Guaranteed clear upward trend
    prices = 2000.0 + np.linspace(0, 500, 100)
    df = pd.DataFrame({
        "timestamp": dates.strftime("%Y-%m-%d"),
        "close": prices
    })
    signals = pd.Series([1] * 100, index=df.index) # Always long

    backtester = StrategyBacktester(initial_capital=100000.0)
    res = backtester.run_backtest(df, signals)
    assert res["final_equity"] > 100000.0
    assert "sharpe_ratio" in res
    assert "max_drawdown_pct" in res

def test_forecast_evaluator(tmp_path):
    evaluator = ForecastEvaluator(models_dir=str(tmp_path))
    preds = [
        {"prediction_date": "2025-01-01", "instrument": "GOLD", "horizon": "1d", "predicted_direction": 1, "predicted_return": 0.5, "actual_return": 1.0},
        {"prediction_date": "2025-01-02", "instrument": "GOLD", "horizon": "1d", "predicted_direction": 1, "predicted_return": 0.5, "actual_return": -0.8}
    ]
    res = evaluator.evaluate_predictions(preds)
    assert res["overall_directional_accuracy_pct"] == 50.0
    assert os.path.exists(os.path.join(str(tmp_path), "model-performance.json"))
