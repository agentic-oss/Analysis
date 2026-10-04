import pytest
import pandas as pd
import numpy as np
from src.indicators.technical import calculate_technical_indicators
from src.macro.correlations import calculate_rolling_correlations, analyze_cross_market_relationships
from src.ratios.gold_silver import analyze_gold_silver_ratio
from src.regimes.detector import RegimeClassifier

def test_technical_indicators():
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": np.linspace(100, 200, 100),
        "high": np.linspace(102, 202, 100),
        "low": np.linspace(98, 198, 100),
        "close": np.linspace(101, 201, 100),
        "volume": np.random.randint(100, 1000, 100)
    })
    config = {"indicators": {"sma_periods": [20, 50], "ema_periods": [9, 20]}}
    res = calculate_technical_indicators(df, config)

    assert "sma_20" in res.columns
    assert "ema_9" in res.columns
    assert "rsi_14" in res.columns
    assert "macd" in res.columns
    assert "atr_14" in res.columns
    assert "volatility_20d" in res.columns
    assert "bb_upper" in res.columns
    assert "obv" in res.columns
    assert "dist_52w_high" in res.columns

def test_gold_silver_ratio():
    gold_df = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "close": [2000.0, 2020.0, 2040.0]
    })
    silver_df = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "close": [25.0, 25.25, 25.5]
    })
    ratio_df = analyze_gold_silver_ratio(gold_df, silver_df)
    assert not ratio_df.empty
    assert "gold_silver_ratio" in ratio_df.columns
    assert ratio_df.iloc[0]["gold_silver_ratio"] == 80.0

def test_regime_classifier():
    gold_df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=30, freq="B").strftime("%Y-%m-%d"),
        "close": np.linspace(2000, 2200, 30),
        "sma_20": np.linspace(1980, 2180, 30),
        "sma_50": np.linspace(1950, 2150, 30),
        "sma_200": np.linspace(1900, 2100, 30),
        "volatility_20d": [0.15] * 30
    })
    market_regime = RegimeClassifier.classify_market_regime(gold_df)
    assert market_regime["trend_regime"] == "Strong Bullish Trend"

    sp500_df = pd.DataFrame({"close": [4000 + i*10 for i in range(30)]})
    oil_df = pd.DataFrame({"close": [70 + i*0.5 for i in range(30)]})
    macro = RegimeClassifier.classify_macro_regime(
        dxy_df=None, us10y_df=None, tip_df=None, oil_df=oil_df, sp500_df=sp500_df, vix_df=None
    )
    assert "macro_regime" in macro
    assert macro["macro_regime"] in ["Inflationary", "Risk-On", "Neutral"]
