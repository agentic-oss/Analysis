"""
Macro Regime Detection and Market Regime Classification Engine.
Provides rule-based, transparent, and explainable macro & market regime models.
"""

from typing import Dict, Any, List
import pandas as pd


class MacroRegimeClassifier:
    """Rule-based transparent Macro Regime Classifier."""

    def classify_macro_regime(
        self,
        dxy_change_20d: float,
        yield_10y_change_20d: float,
        real_yield_change_20d: float,
        vix_level: float,
        oil_return_20d: float,
        sp500_return_20d: float,
    ) -> Dict[str, Any]:
        """
        Classifies current macro environment into explainable regime labels:
        Inflationary, Disinflationary, Deflationary, Risk-on, Risk-off, Tightening, Easing, Stagflationary, Neutral.
        """
        reasons = []

        # Risk Regime
        if sp500_return_20d > 0.02 and vix_level < 18.0:
            risk_regime = "Risk-on"
            reasons.append("S&P 500 rising with low equity volatility")
        elif sp500_return_20d < -0.03 or vix_level > 22.0:
            risk_regime = "Risk-off"
            reasons.append("S&P 500 declining or high VIX spike")
        else:
            risk_regime = "Neutral"
            reasons.append("Moderate equity returns and normal volatility")

        # Monetary Policy & Rate Regime
        if real_yield_change_20d > 0.15 or yield_10y_change_20d > 0.20:
            monetary_regime = "Tightening"
            reasons.append("Rising real yields / 10Y Treasury yields")
        elif real_yield_change_20d < -0.15 or yield_10y_change_20d < -0.20:
            monetary_regime = "Easing"
            reasons.append("Falling real yields / 10Y Treasury yields")
        else:
            monetary_regime = "Neutral"

        # Inflation / Economic Growth Regime
        if oil_return_20d > 0.08 and real_yield_change_20d > 0.0:
            if sp500_return_20d < -0.01:
                inflation_regime = "Stagflationary"
                reasons.append("Rising energy costs with weak equity growth and firm yields")
            else:
                inflation_regime = "Inflationary"
                reasons.append("Rising commodity prices and yields")
        elif oil_return_20d < -0.08 and dxy_change_20d > 0.01:
            inflation_regime = "Disinflationary"
            reasons.append("Falling energy prices and strong USD")
        else:
            inflation_regime = "Neutral"

        # Composite Primary Regime
        if inflation_regime != "Neutral":
            primary_regime = inflation_regime
        elif risk_regime != "Neutral":
            primary_regime = risk_regime
        elif monetary_regime != "Neutral":
            primary_regime = monetary_regime
        else:
            primary_regime = "Neutral"

        return {
            "primary_regime": primary_regime,
            "risk_regime": risk_regime,
            "monetary_regime": monetary_regime,
            "inflation_regime": inflation_regime,
            "components": {
                "dxy_change_20d": dxy_change_20d,
                "yield_10y_change_20d": yield_10y_change_20d,
                "real_yield_change_20d": real_yield_change_20d,
                "vix_level": vix_level,
                "oil_return_20d": oil_return_20d,
                "sp500_return_20d": sp500_return_20d,
            },
            "reasons": reasons,
        }


class MarketRegimeClassifier:
    """Classifies gold and silver specific trend and volatility market regimes."""

    def classify_market_regime(
        self,
        close: float,
        sma_20: float,
        sma_50: float,
        sma_200: float,
        rsi: float,
        volatility_20d: float,
    ) -> Dict[str, Any]:
        """Classifies trend and volatility dimensions independently."""
        # Trend Regime
        if close > sma_20 and sma_20 > sma_50 and sma_50 > sma_200:
            trend_regime = "Strong Bullish Trend"
        elif close > sma_50 and sma_50 > sma_200:
            trend_regime = "Bullish Trend"
        elif close < sma_20 and sma_20 < sma_50 and sma_50 < sma_200:
            trend_regime = "Strong Bearish Trend"
        elif close < sma_50 and sma_50 < sma_200:
            trend_regime = "Bearish Trend"
        else:
            trend_regime = "Range"

        # Volatility Regime
        if volatility_20d > 0.22:
            vol_regime = "High Volatility"
        elif volatility_20d < 0.10:
            vol_regime = "Low Volatility"
        else:
            vol_regime = "Normal Volatility"

        return {
            "trend_regime": trend_regime,
            "volatility_regime": vol_regime,
            "rsi": rsi,
            "dist_sma_200_pct": ((close - sma_200) / (sma_200 + 1e-9)) * 100.0 if sma_200 else 0.0,
        }
