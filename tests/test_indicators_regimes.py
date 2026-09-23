import pytest
import pandas as pd
import numpy as np
from src.indicators.technical import calculate_technical_indicators
from src.macro.analyzer import MetalsMacroAnalyzer
from src.ratios.relative_value import RelativeValueAnalyzer
from src.regimes.classifier import RegimeClassifier

def test_technical_indicators():
    dates = pd.date_range("2024-01-01", periods=250, freq="B")
    prices = 2000.0 + np.cumsum(np.random.normal(0.5, 10, 250))
    df = pd.DataFrame({
        "timestamp": dates.strftime("%Y-%m-%d"),
        "open": prices - 2.0,
        "high": prices + 5.0,
        "low": prices - 5.0,
        "close": prices,
        "volume": 10000
    })

    ind_df = calculate_technical_indicators(df)
    assert "sma_20" in ind_df.columns
    assert "rsi_14" in ind_df.columns
    assert "macd" in ind_df.columns
    assert "volatility_20d" in ind_df.columns
    assert not ind_df["sma_20"].isna().all()

def test_macro_analyzer():
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    gold_df = pd.DataFrame({
        "timestamp": dates.strftime("%Y-%m-%d"),
        "close": 2000.0 + np.cumsum(np.random.normal(0, 5, 100))
    })
    dxy_df = pd.DataFrame({
        "timestamp": dates.strftime("%Y-%m-%d"),
        "close": 104.0 + np.cumsum(np.random.normal(0, 0.2, 100))
    })

    analyzer = MetalsMacroAnalyzer(windows=[20, 60])
    summary = analyzer.get_latest_correlation_summary(gold_df, {"DXY": dxy_df})
    assert "DXY" in summary
    assert "20d" in summary["DXY"]

def test_relative_value_analyzer():
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    gold_df = pd.DataFrame({"timestamp": dates.strftime("%Y-%m-%d"), "close": 2000.0})
    silver_df = pd.DataFrame({"timestamp": dates.strftime("%Y-%m-%d"), "close": 25.0})

    rv = RelativeValueAnalyzer()
    ratio_df = rv.calculate_ratio_series(gold_df, silver_df)
    assert not ratio_df.empty
    assert (ratio_df["gold_silver_ratio"] == 80.0).all()

    summary = rv.analyze_ratio_extremes(ratio_df)
    assert summary["current_ratio"] == 80.0

def test_regime_classifier():
    classifier = RegimeClassifier()
    macro_res = classifier.classify_macro_regime(rates_change_60d=0.5, cpi_yoy=4.0, vix_level=25.0)
    assert macro_res["primary_macro_regime"] in ["Stagflationary", "Inflationary", "Risk-Off"]

    row = pd.Series({"close": 2100.0, "sma_20": 2050.0, "sma_50": 2000.0, "sma_200": 1900.0, "rsi_14": 65.0, "volatility_20d": 15.0})
    market_res = classifier.classify_market_regime(row)
    assert market_res["trend_regime"] == "Strong Bullish Trend"
