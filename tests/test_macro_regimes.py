import pytest
import pandas as pd
from src.macro.macro_analysis import MacroAnalyzer
from src.regimes.macro_regimes import MacroRegimeDetector

def test_macro_and_regimes():
    dates = pd.date_range("2026-01-01", periods=25).strftime("%Y-%m-%d")
    vix = pd.DataFrame({"date": dates, "close": [30.0] * 25})
    sp500 = pd.DataFrame({"date": dates, "close": [5000 - i * 10 for i in range(25)]})

    macro_dfs = {"VIX": vix, "SP500": sp500}
    regime = MacroRegimeDetector.detect_macro_regime(macro_dfs)
    assert regime["primary_regime"] == "Risk-Off"
