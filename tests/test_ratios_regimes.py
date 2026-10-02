import pytest
import pandas as pd
from src.ratios.ratio_analysis import GoldSilverRatioAnalyzer
from src.regimes.macro_regime import MacroRegimeClassifier
from src.regimes.market_regime import MarketRegimeClassifier

def test_gold_silver_ratio():
    dates = pd.date_range("2026-01-01", periods=100)
    gold = pd.Series(2500.0, index=dates)
    silver = pd.Series(30.0, index=dates)

    analyzer = GoldSilverRatioAnalyzer()
    ratio_df = analyzer.analyze_ratio(gold, silver)
    summary = analyzer.get_latest_summary(ratio_df)

    assert round(summary["current_ratio"], 2) == round(2500.0 / 30.0, 2)
    assert "relative_value_regime" in summary

def test_macro_market_regimes():
    macro = MacroRegimeClassifier().classify_regime({"vix_level": 30.0, "sp500_return_20d": -0.08})
    assert macro["risk_regime"] == "Risk-Off"

    mkt = MarketRegimeClassifier().classify_market_regime(pd.DataFrame([{
        "close": 2600, "sma_20": 2550, "sma_50": 2500, "sma_200": 2400, "volatility_20d": 0.15, "rsi_14": 60
    }]))
    assert mkt["trend_regime"] == "Strong Bullish Trend"
