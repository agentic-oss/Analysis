import pytest
import numpy as np
import pandas as pd
from src.indicators.technical import TechnicalIndicators

def test_technical_indicators_calculation():
    dates = pd.date_range("2025-01-01", periods=250, freq="B")
    np.random.seed(42)
    close = 2000.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.01, size=len(dates))))

    df = pd.DataFrame({
        "Date": dates,
        "Open": close * 0.99,
        "High": close * 1.01,
        "Low": close * 0.98,
        "Close": close,
        "Volume": np.random.randint(10000, 50000, size=len(dates)),
    })

    res = TechnicalIndicators.calculate_all(df)

    assert "sma_20" in res.columns
    assert "sma_200" in res.columns
    assert "rsi_14" in res.columns
    assert "macd" in res.columns
    assert "atr_14" in res.columns
    assert "bb_upper" in res.columns
    assert "signal_golden_cross" in res.columns
    assert not res["rsi_14"].dropna().empty
