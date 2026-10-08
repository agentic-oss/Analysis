import pytest
import pandas as pd
import numpy as np
from src.models.baseline_models import BaselineModels
from src.backtesting.engine import BacktestEngine
from src.evaluation.tracking import ForecastTracker

def test_baseline_models():
    dates = pd.date_range("2020-01-01", periods=300, freq="B").strftime("%Y-%m-%d")
    df = pd.DataFrame({
        "date": dates,
        "rsi_14": np.random.uniform(30, 70, 300),
        "dist_sma_200": np.random.uniform(-0.05, 0.05, 300),
        "future_direction_5d": np.random.choice([0, 1], 300)
    })

    res = BaselineModels.train_evaluate_walk_forward(
        df, feature_cols=["rsi_14", "dist_sma_200"], target_col="future_direction_5d", train_window_min=200, step_size=20
    )
    assert "accuracy" in res
    assert 0.0 <= res["accuracy"] <= 1.0

def test_backtest_engine():
    dates = pd.date_range("2026-01-01", periods=50, freq="B").strftime("%Y-%m-%d")
    df = pd.DataFrame({
        "date": dates,
        "close": np.linspace(100, 150, 50),
        "composite_score": [70.0 if i % 2 == 0 else 30.0 for i in range(50)]
    })

    bt = BacktestEngine.run_signal_backtest(df, signal_col="composite_score")
    assert "final_capital" in bt
    assert "cumulative_return_pct" in bt
    assert "total_trades" in bt

def test_forecast_tracker():
    prices = pd.DataFrame({
        "date": ["2026-03-01", "2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05", "2026-03-06"],
        "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    })
    preds = [{
        "date": "2026-03-01",
        "symbol": "GOLD",
        "horizons": {
            "5d": {"probability_positive_return_pct": 60.0, "expected_median_return_pct": 2.0}
        }
    }]

    eval_res = ForecastTracker.evaluate_past_predictions(preds, prices)
    assert eval_res["total_evaluated_forecasts"] == 1
    assert eval_res["directional_accuracy_pct"] == 100.0
