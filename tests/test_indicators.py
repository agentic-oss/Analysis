import pytest
import pandas as pd
import numpy as np
from src.indicators.technical import TechnicalIndicators

def test_technical_indicators_calculation():
    dates = pd.date_range("2026-01-01", periods=250)
    prices = 2000.0 + np.cumsum(np.random.normal(0.5, 5.0, 250))
    df = pd.DataFrame({
        "date": dates,
        "open": prices * 0.99,
        "high": prices * 1.01,
        "low": prices * 0.98,
        "close": prices,
        "volume": 10000
    })

    res = TechnicalIndicators.calculate_all(df)
    assert "sma_20" in res.columns
    assert "sma_200" in res.columns
    assert "rsi_14" in res.columns
    assert "macd" in res.columns
    assert "atr_14" in res.columns
    assert "bb_width" in res.columns
    assert "golden_cross" in res.columns
