import pytest
import pandas as pd
import numpy as np
from src.indicators.technical import TechnicalIndicators
from src.metals.analytics import GoldAnalyzer, SilverAnalyzer, RelativeValueAnalyzer

def test_technical_indicators_calculation():
    dates = pd.date_range("2023-01-01", periods=100, freq="B").strftime("%Y-%m-%d")
    np.random.seed(42)
    prices = 2000.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))
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
    assert "ema_20" in res.columns
    assert "rsi_14" in res.columns
    assert "macd" in res.columns
    assert "atr_14" in res.columns
    assert "volatility_20d" in res.columns
    assert "bollinger_upper" in res.columns
    assert "dist_sma_200" in res.columns

def test_gold_silver_analyzers():
    dates = pd.date_range("2023-01-01", periods=100, freq="B").strftime("%Y-%m-%d")
    g_prices = 2000.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))
    s_prices = 25.0 * np.exp(np.cumsum(np.random.normal(0, 0.015, 100)))

    gold_df = pd.DataFrame({"date": dates, "open": g_prices, "high": g_prices*1.01, "low": g_prices*0.99, "close": g_prices, "volume": 5000})
    silver_df = pd.DataFrame({"date": dates, "open": s_prices, "high": s_prices*1.01, "low": s_prices*0.99, "close": s_prices, "volume": 5000})

    g_analyzer = GoldAnalyzer()
    g_res = g_analyzer.analyze(gold_df)
    assert g_res["symbol"] == "GOLD"
    assert "rsi_14" in g_res

    s_analyzer = SilverAnalyzer()
    s_res = s_analyzer.analyze(silver_df, gold_df=gold_df)
    assert s_res["symbol"] == "SILVER"
    assert "gold_silver_ratio" in s_res
    assert "current_ratio" in s_res["gold_silver_ratio"]

    rv_analyzer = RelativeValueAnalyzer()
    rv_res = rv_analyzer.analyze(gold_df, silver_df)
    assert "gold_silver_ratio" in rv_res
    assert "ratio_zscore_60" in rv_res
