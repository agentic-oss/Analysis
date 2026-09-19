import pytest
import os
import pandas as pd
import numpy as np
from src.models.baseline_models import BaselineModels
from src.evaluation.evaluator import ForecastEvaluator
from src.backtesting.backtest_engine import BacktestEngine
from src.pipeline.runner import DailyPipelineRunner

def test_baseline_models():
    np.random.seed(42)
    df = pd.DataFrame({
        "feat1": np.random.randn(100),
        "feat2": np.random.randn(100),
        "future_return_5d": np.random.randn(100) * 2,
        "future_direction_5d": np.random.choice([0, 1], size=100)
    })
    res = BaselineModels.train_and_predict(df, ["feat1", "feat2"], horizon=5)
    assert res["status"] == "success"
    assert "logistic_accuracy" in res
    assert "latest_predicted_direction" in res

def test_backtest_engine():
    price_df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=50, freq="B").strftime("%Y-%m-%d"),
        "close": 100 + np.cumsum(np.random.randn(50))
    })
    signals = pd.Series(np.random.choice([-1, 0, 1], size=50))
    bt_res = BacktestEngine.run_signal_backtest(price_df, signals)
    assert "total_return_pct" in bt_res
    assert "sharpe_ratio" in bt_res

def test_daily_pipeline_runner(tmp_path):
    runner = DailyPipelineRunner()
    target_date = "2026-03-31"
    res = runner.run_pipeline(target_date, use_mock=True)

    assert res["date"] == target_date
    assert "gold" in res
    assert "silver" in res

    # Verify outputs created on disk
    assert os.path.exists(f"data/analysis/daily/{target_date}.json")
    assert os.path.exists(f"reports/daily/{target_date}.md")
