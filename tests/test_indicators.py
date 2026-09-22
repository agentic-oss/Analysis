"""
Tests for technical indicators and price structure calculations.
"""

import pytest
import pandas as pd
import numpy as np
from src.indicators.technical import TechnicalIndicators


def test_technical_indicators_calculation():
    # Create sample prices with known trend
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=260, freq="D").strftime("%Y-%m-%d")
    close_prices = 100.0 + np.cumsum(np.random.normal(0.5, 1.0, 260))
    open_prices = close_prices - 0.2
    high_prices = np.maximum(open_prices, close_prices) + 0.5
    low_prices = np.minimum(open_prices, close_prices) - 0.5
    volume = np.random.randint(1000, 5000, 260)

    df = pd.DataFrame({
        "date": dates,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volume
    })

    result = TechnicalIndicators.calculate_all(df)

    assert "sma_20" in result.columns
    assert "sma_200" in result.columns
    assert "ema_9" in result.columns
    assert "rsi_14" in result.columns
    assert "macd" in result.columns
    assert "atr_14" in result.columns
    assert "bollinger_bandwidth" in result.columns
    assert "dist_sma_200_pct" in result.columns
    assert "golden_cross" in result.columns

    # Verify RSI range between 0 and 100
    rsi = result["rsi_14"].dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()

    # Verify SMA 20 non-null towards end
    assert not pd.isna(result["sma_20"].iloc[-1])
