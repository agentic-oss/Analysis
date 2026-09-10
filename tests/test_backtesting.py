import pytest
import pandas as pd
import numpy as np
from src.backtesting.engine import ForecastingModelPipeline, BacktestEngine

def test_forecasting_model_pipeline():
    np.random.seed(42)
    n = 150
    dates = pd.date_range("2023-01-01", periods=n, freq="B").strftime("%Y-%m-%d")
    df = pd.DataFrame({
        "date": dates,
        "rsi_14": np.random.uniform(30, 70, n),
        "dist_sma_20": np.random.normal(0, 0.02, n),
        "volatility_20d": np.random.uniform(0.10, 0.25, n),
        "future_return_5d": np.random.normal(0.002, 0.015, n)
    })

    pipeline = ForecastingModelPipeline(feature_cols=["rsi_14", "dist_sma_20", "volatility_20d"])
    res = pipeline.evaluate_walk_forward(df, target_col="future_return_5d", min_train_size=100, step_size=10)

    assert res["status"] == "success"
    assert "classification_metrics" in res
    assert "accuracy" in res["classification_metrics"]
    assert "regression_metrics" in res

def test_backtest_engine():
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="B").strftime("%Y-%m-%d")
    np.random.seed(42)
    prices = 2000.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, n)))
    signals = np.random.choice([1, 0, -1], size=n)

    df = pd.DataFrame({
        "date": dates,
        "close": prices,
        "signal": signals
    })

    engine = BacktestEngine(transaction_cost_bps=5.0, slippage_bps=2.0)
    res = engine.run_signal_backtest(df, signal_col="signal", price_col="close")

    assert "total_return" in res
    assert "sharpe_ratio" in res
    assert "max_drawdown" in res
    assert res["transaction_costs_applied_bps"] == 7.0
