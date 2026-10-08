import pytest
import pandas as pd
import numpy as np
from src.indicators.technical import TechnicalIndicators
from src.ratios.gold_silver_ratio import GoldSilverRatio

def test_indicators_math():
    close_vals = np.linspace(100, 200, 50)
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=50).strftime("%Y-%m-%d"),
        "open": close_vals - 1,
        "high": close_vals + 2,
        "low": close_vals - 2,
        "close": close_vals,
        "volume": [1000] * 50
    })
    ind_df = TechnicalIndicators.add_all_indicators(df)
    assert ind_df["sma_20"].iloc[-1] == pytest.approx(close_vals[-20:].mean())
    assert ind_df["rsi_14"].iloc[-1] == 100.0  # Monotonically increasing close

def test_ratio_calculations():
    gold = pd.DataFrame({"date": ["2026-03-20"], "close": [2000.0]})
    silver = pd.DataFrame({"date": ["2026-03-20"], "close": [25.0]})
    ratio_df = GoldSilverRatio.calculate_ratio(gold, silver)
    assert ratio_df["gold_silver_ratio"].iloc[0] == 80.0
