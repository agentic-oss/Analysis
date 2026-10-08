import pytest
import pandas as pd
from src.models.baseline_models import BaselineModels
from src.backtesting.engine import BacktestEngine

def test_models_and_backtesting():
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=100).strftime("%Y-%m-%d"),
        "close": [100 + i for i in range(100)],
        "composite_score": [65.0 if i % 2 == 0 else 35.0 for i in range(100)],
        "rsi_14": [50.0] * 100,
        "future_direction_5d": [1] * 100
    })

    bt = BacktestEngine.run_signal_backtest(df)
    assert "cumulative_return_pct" in bt
