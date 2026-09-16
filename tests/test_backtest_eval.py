import numpy as np
import pandas as pd
import pytest
from src.backtesting.engine import BacktestEngine
from src.evaluation.tracker import ForecastEvaluator

def test_backtest_engine():
    dates = pd.date_range("2025-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "close": 2000.0 + np.cumsum(np.random.randn(100) * 5),
        "signal": np.random.choice([-1.0, 0.0, 1.0], 100)
    })

    bt = BacktestEngine(initial_capital=100000.0)
    res = bt.run_backtest(df, signal_col="signal", price_col="close")

    assert res["status"] == "success"
    assert "sharpe_ratio" in res
    assert "max_drawdown" in res

def test_forecast_evaluator(tmp_path):
    pred_dir = tmp_path / "predictions"
    model_path = tmp_path / "model_perf.json"

    evaluator = ForecastEvaluator(predictions_dir=str(pred_dir), model_metrics_path=str(model_path))
    evaluator.log_daily_prediction("2025-01-01", "GOLD", 5, 1, 0.02, 75.0)

    # Provide market data to evaluate prediction
    dates = pd.date_range("2025-01-01", periods=10, freq="D")
    market_df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "close": [2000, 2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090]
    })

    res = evaluator.evaluate_past_predictions({"GOLD": market_df})
    assert res["evaluated_predictions_count"] == 1
    assert res["overall_accuracy"] == 1.0
