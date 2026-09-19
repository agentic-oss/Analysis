import pytest
import pandas as pd
import numpy as np
from src.indicators.technical import TechnicalIndicators
from src.macro.macro_analyzer import MacroAnalyzer
from src.metals.metal_analyzer import MetalAnalyzer
from src.ratios.ratio_analyzer import RatioAnalyzer
from src.regimes.regime_detector import RegimeDetector

def test_technical_indicators():
    dates = pd.date_range("2026-01-01", periods=60, freq="B").strftime("%Y-%m-%d")
    np.random.seed(42)
    prices = 2000 + np.cumsum(np.random.randn(60) * 10)
    df = pd.DataFrame({
        "date": dates,
        "open": prices,
        "high": prices + 5,
        "low": prices - 5,
        "close": prices,
        "volume": 1000
    })

    config = {"indicators": {"sma_windows": [20, 50]}}
    res = TechnicalIndicators.calculate_all(df, config)

    assert "sma_20" in res.columns
    assert "rsi_14" in res.columns
    assert "macd" in res.columns
    assert "atr_14" in res.columns
    assert len(res) == 60

def test_ratio_analyzer():
    gold_df = pd.DataFrame({
        "date": ["2026-03-30", "2026-03-31"],
        "close": [2000.0, 2020.0],
        "return_1d": [0.0, 1.0],
        "return_5d": [0.0, 2.0],
        "return_20d": [0.0, 5.0]
    })
    silver_df = pd.DataFrame({
        "date": ["2026-03-30", "2026-03-31"],
        "close": [25.0, 25.25],
        "return_1d": [0.0, 1.0],
        "return_5d": [0.0, 2.0],
        "return_20d": [0.0, 5.0]
    })

    ratio_res = RatioAnalyzer.calculate_gold_silver_ratio(gold_df, silver_df)
    assert ratio_res["current_ratio"] == 80.0 # 2020 / 25.25

def test_regime_detector():
    macro_data = {
        "dxy_return_20d": -2.5,
        "us10y_yield": 3.8,
        "us10y_change_20d": -0.4,
        "vix": 28.0,
        "sp500_return_20d": -4.0,
        "oil_return_20d": 6.0
    }
    regime = RegimeDetector.detect_macro_regime(macro_data)
    assert regime["primary_regime"] in ["Risk-off", "Stagflationary", "Easing"]
