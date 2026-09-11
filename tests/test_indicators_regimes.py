"""
Unit tests for technical analysis, Indian market conversions, ratios, and regime engines.
"""

import pytest
import pandas as pd
import numpy as np
from src.indicators.technical import TechnicalAnalysisEngine
from src.metals.indian_market import IndianMarketConverter
from src.ratios.ratio_engine import RatioAnalysisEngine
from src.regimes.macro_regime import MacroRegimeClassifier, MarketRegimeClassifier


def test_technical_analysis_engine():
    dates = pd.date_range("2025-01-01", periods=250, freq="B")
    prices = 2000.0 + np.cumsum(np.random.normal(0.5, 10.0, 250))
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": prices,
        "high": prices + 5.0,
        "low": prices - 5.0,
        "close": prices,
        "volume": 10000,
    })

    engine = TechnicalAnalysisEngine()
    res = engine.compute_indicators(df)

    assert "sma_20" in res.columns
    assert "rsi_14" in res.columns
    assert "macd" in res.columns
    assert "atr" in res.columns
    assert "bollinger_upper" in res.columns
    assert "dist_sma_200_pct" in res.columns


def test_indian_market_converter():
    converter = IndianMarketConverter(custom_duty_pct=0.06)
    # Gold USD 2000/oz, USDINR 83.0 -> 2000/31.1034768 * 10 * 83.0 * 1.06 ~ 56534.81
    gold_inr = converter.convert_gold_usd_to_inr_10g(2000.0, 83.0)
    assert 55000 < gold_inr < 58000

    # Silver USD 25/oz, USDINR 83.0 -> 25/31.1034768 * 1000 * 83.0 * 1.06 ~ 70701.38
    silver_inr = converter.convert_silver_usd_to_inr_kg(25.0, 83.0)
    assert 68000 < silver_inr < 73000


def test_ratio_analysis_engine():
    gold_df = pd.DataFrame({"date": ["2026-01-01", "2026-01-02"], "close": [2000.0, 2020.0]})
    silver_df = pd.DataFrame({"date": ["2026-01-01", "2026-01-02"], "close": [25.0, 25.0]})

    engine = RatioAnalysisEngine()
    res = engine.calculate_ratio_metrics(gold_df, silver_df)

    assert "gold_silver_ratio" in res.columns
    assert res["gold_silver_ratio"].iloc[0] == 80.0
    assert res["gold_silver_ratio"].iloc[1] == 80.8


def test_macro_regime_classifier():
    macro_clf = MacroRegimeClassifier()
    regime = macro_clf.classify_macro_regime(
        dxy_change_20d=0.02,
        yield_10y_change_20d=0.25,
        real_yield_change_20d=0.20,
        vix_level=15.0,
        oil_return_20d=0.10,
        sp500_return_20d=-0.02,
    )
    assert regime["primary_regime"] == "Stagflationary"
    assert regime["monetary_regime"] == "Tightening"

    mkt_clf = MarketRegimeClassifier()
    mkt_regime = mkt_clf.classify_market_regime(
        close=2050.0, sma_20=2020.0, sma_50=2000.0, sma_200=1900.0, rsi=65.0, volatility_20d=0.15
    )
    assert mkt_regime["trend_regime"] == "Strong Bullish Trend"
