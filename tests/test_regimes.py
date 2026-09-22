"""
Tests for macro and market regime classification engines.
"""

import pytest
import pandas as pd
import numpy as np
from src.regimes.macro import MacroRegimeDetector
from src.regimes.market import MarketRegimeDetector


def test_macro_regime_classification():
    data = pd.DataFrame([{
        "dxy_return_20d": 0.03,
        "us10y_change_20d": 0.20,
        "oil_return_20d": 0.10,
        "sp500_return_20d": -0.04,
        "vix_close": 22.0
    }])

    res = MacroRegimeDetector.classify_df(data)
    assert res.loc[0, "macro_regime"] == "Stagflationary"


def test_market_regime_classification():
    df = pd.DataFrame({
        "close": [100, 110, 120, 130],
        "sma_20": [95, 105, 115, 125],
        "sma_50": [90, 100, 110, 120],
        "sma_200": [80, 90, 100, 110],
        "rsi_14": [60, 65, 70, 75],
        "volatility_20d": [0.10, 0.12, 0.15, 0.25]
    })

    res = MarketRegimeDetector.classify_instrument(df)
    assert "trend_regime" in res.columns
    assert "volatility_regime" in res.columns
    assert res.loc[3, "trend_regime"] == "Strong Bullish Trend"
