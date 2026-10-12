import pytest
import pandas as pd
import numpy as np
from src.regimes.classifier import MacroRegimeClassifier, MarketRegimeClassifier
from src.patterns.analogue import HistoricalAnalogueEngine

def test_macro_regime_classifier():
    classifier = MacroRegimeClassifier()
    res = classifier.classify_macro_regime(
        dxy_return_20d=0.02,
        us10y_change_20d=0.25,
        oil_return_20d=-0.05,
        sp500_return_20d=-0.03,
        vix_level=25.0
    )
    assert res["regime"] in MacroRegimeClassifier.REGIMES
    assert "reasons" in res
    assert len(res["reasons"]) > 0

def test_market_regime_classifier():
    classifier = MarketRegimeClassifier()
    res = classifier.classify_market_regime(
        close_price=2100.0,
        sma_20=2050.0,
        sma_50=2000.0,
        sma_200=1900.0,
        rsi_14=65.0,
        hist_vol_20=0.18
    )
    assert res["trend"] == "Strong Bullish Trend"
    assert res["volatility"] == "Normal Volatility"

def test_historical_analogue_engine():
    # Construct dummy historical dataset with known future returns
    dates = pd.date_range("2024-01-01", periods=50, freq="B").astype(str)
    np.random.seed(42)
    hist_df = pd.DataFrame({
        "date": dates,
        "rsi_14": np.random.uniform(40, 60, 50),
        "dist_sma_20": np.random.uniform(-2, 2, 50),
        "dist_sma_50": np.random.uniform(-3, 3, 50),
        "hist_vol_20": np.random.uniform(0.1, 0.2, 50),
        "gold_silver_ratio": np.random.uniform(80, 85, 50),
        "dxy_return_20d": np.random.uniform(-0.02, 0.02, 50),
        "us10y_change_20d": np.random.uniform(-0.1, 0.1, 50),
        "future_return_1d": np.random.normal(0.001, 0.01, 50),
        "future_return_5d": np.random.normal(0.005, 0.02, 50),
        "future_return_10d": np.random.normal(0.01, 0.03, 50)
    })

    current_state = {
        "rsi_14": 50.0,
        "dist_sma_20": 0.0,
        "dist_sma_50": 0.0,
        "hist_vol_20": 0.15,
        "gold_silver_ratio": 82.5,
        "dxy_return_20d": 0.0,
        "us10y_change_20d": 0.0
    }

    engine = HistoricalAnalogueEngine(top_k=5)
    out = engine.find_analogues(current_state, hist_df)

    assert out["top_k"] == 5
    assert len(out["analogues"]) == 5
    assert "1d" in out["forward_statistics"]
    assert "prob_positive_pct" in out["forward_statistics"]["1d"]
