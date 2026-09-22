"""
Tests for Backtesting engine, Forecast evaluator, and Alert engine.
"""

import os
import pytest
import pandas as pd
import numpy as np
from src.backtesting.engine import StrategyBacktester
from src.evaluation.tracker import ForecastEvaluator
from src.scoring.alerts import AlertEngine


def test_backtesting_and_eval(tmp_path):
    dates = pd.date_range("2025-01-01", periods=100, freq="D").strftime("%Y-%m-%d")
    prices = 2000.0 + np.cumsum(np.random.normal(1.0, 5.0, 100))
    df = pd.DataFrame({"date": dates, "close": prices})
    signals = pd.Series(np.random.choice([-1, 0, 1], size=100))

    backtester = StrategyBacktester(initial_capital=100000.0)
    res = backtester.run_backtest(df, signals)

    assert "final_capital" in res
    assert "sharpe_ratio" in res
    assert "win_rate_pct" in res

    # Test Forecast Evaluator
    evaluator = ForecastEvaluator(predictions_dir=str(tmp_path / "predictions"), models_dir=str(tmp_path / "models"))
    pred_data = {
        "prediction_date": dates[10],
        "horizon_signals": {
            "5d": {
                "horizon_days": 5,
                "research_bias": "Bullish",
                "historical_mean_return_pct": 1.2,
            }
        }
    }
    evaluator.log_prediction(pred_data)
    eval_res = evaluator.evaluate_all_predictions(df)

    assert eval_res["total_evaluated_forecasts"] == 1

    # Test Alert Engine
    row = pd.Series({
        "rsi_14": 75.0,
        "gold_silver_ratio_zscore": 2.5,
        "golden_cross": True,
    })
    alert_engine = AlertEngine()
    alerts_out = alert_engine.detect_alerts("2026-03-01", row, output_dir=str(tmp_path / "alerts"))

    assert alerts_out["alert_count"] == 3
