import pandas as pd
import pytest
from src.ratios.relative_value import RelativeValueAnalytics
from src.regimes.detector import RegimeDetector

def test_gold_silver_ratio():
    dates = pd.date_range("2025-01-01", periods=100, freq="D")
    gold_df = pd.DataFrame({"date": dates, "close": [2000.0 + i for i in range(100)]})
    silver_df = pd.DataFrame({"date": dates, "close": [25.0 + (i * 0.05) for i in range(100)]})

    res = RelativeValueAnalytics.calculate_gold_silver_ratio(gold_df, silver_df)
    assert "gold_silver_ratio" in res.columns
    assert "gs_ratio_zscore" in res.columns
    assert not res.empty

def test_regime_detector():
    df = pd.DataFrame({
        "close": [2000.0],
        "sma_20": [1980.0],
        "sma_50": [1950.0],
        "sma_200": [1900.0],
        "volatility_20d": [0.15]
    })
    regime = RegimeDetector.detect_market_regime(df)
    assert "Strong Bullish" in regime

    macro_data = {
        "VIX": pd.DataFrame({"close": [28.0]}),
        "SP500": pd.DataFrame({"close": [5000.0, 4900.0]})
    }
    macro_regime = RegimeDetector.detect_macro_regime(macro_data)
    assert macro_regime["sentiment"] == "Risk-Off"
