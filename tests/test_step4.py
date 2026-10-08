import pytest
import pandas as pd
import numpy as np
from src.macro.macro_analysis import MacroAnalyzer
from src.regimes.macro_regimes import MacroRegimeDetector
from src.regimes.market_regimes import MarketRegimeDetector
from src.patterns.analogues import AnalogueEngine

def test_macro_analysis_and_regimes():
    dates = pd.date_range("2026-01-01", periods=30, freq="B").strftime("%Y-%m-%d")
    gold_df = pd.DataFrame({"date": dates, "close": np.linspace(2000, 2100, 30)})
    dxy_df = pd.DataFrame({"date": dates, "close": np.linspace(104, 102, 30)})
    sp500_df = pd.DataFrame({"date": dates, "close": np.linspace(5000, 5200, 30)})
    vix_df = pd.DataFrame({"date": dates, "close": [15.0] * 30})

    macro_dfs = {"DXY": dxy_df, "SP500": sp500_df, "VIX": vix_df}

    corrs = MacroAnalyzer.calculate_rolling_correlations(gold_df, macro_dfs, windows=[20])
    assert "DXY" in corrs
    assert "corr_20d" in corrs["DXY"]

    macro_regime = MacroRegimeDetector.detect_macro_regime(macro_dfs)
    assert "primary_regime" in macro_regime
    assert macro_regime["primary_regime"] in ["Risk-On", "Neutral"]

    mkt_regime = MarketRegimeDetector.detect_market_regime(gold_df, "GOLD")
    assert "trend_regime" in mkt_regime

def test_analogue_engine():
    dates = pd.date_range("2020-01-01", periods=100, freq="B").strftime("%Y-%m-%d")
    hist_df = pd.DataFrame({
        "date": dates,
        "rsi_14": np.random.uniform(30, 70, 100),
        "stoch_rsi": np.random.uniform(0, 1, 100),
        "macd_hist": np.random.uniform(-5, 5, 100),
        "volatility_20d": np.random.uniform(0.1, 0.3, 100),
        "dist_sma_20": np.random.uniform(-0.05, 0.05, 100),
        "dist_sma_50": np.random.uniform(-0.05, 0.05, 100),
        "dist_sma_200": np.random.uniform(-0.05, 0.05, 100),
        "future_return_5d": np.random.uniform(-0.02, 0.02, 100),
        "future_return_20d": np.random.uniform(-0.05, 0.05, 100),
    })

    curr_row = hist_df.iloc[-1]
    res = AnalogueEngine.find_analogues(hist_df, curr_row, top_n=5)
    assert "num_analogues_found" in res
    assert res["num_analogues_found"] > 0
    assert "forward_statistics" in res
