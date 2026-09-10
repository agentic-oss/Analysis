import pytest
from src.regimes.classifier import MacroRegimeClassifier, MarketRegimeClassifier

def test_macro_regime_classifier():
    res = MacroRegimeClassifier.classify_macro_regime(
        us10y_change_20d=0.30,
        dxy_return_20d=0.01,
        oil_return_20d=0.08,
        sp500_return_20d=-0.04,
        vix_level=22.0
    )
    assert "primary_macro_regime" in res
    assert "regime_scores" in res
    assert res["regime_scores"]["Stagflationary"] > 0

def test_market_regime_classifier():
    res = MarketRegimeClassifier.classify_market_regime(
        price=2100.0,
        sma_20=2050.0,
        sma_50=2000.0,
        sma_200=1900.0,
        volatility_20d=0.18,
        rsi_14=62.0,
        macro_regime="Risk-Off"
    )
    assert res["trend_regime"] == "Strong Bullish Trend"
    assert res["volatility_regime"] == "Moderate Volatility"
    assert "Risk-Off" in res["combined_label"]
