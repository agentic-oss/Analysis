import pandas as pd
from typing import Dict, Any, Optional

class MacroRegimeClassifier:
    """Classifies global macroeconomic regime in an explainable rule-based manner."""

    @staticmethod
    def classify_macro_regime(
        us10y_change_20d: float = 0.0,
        dxy_return_20d: float = 0.0,
        oil_return_20d: float = 0.0,
        sp500_return_20d: float = 0.0,
        vix_level: float = 18.0
    ) -> Dict[str, Any]:
        """Rules-based composite classification of macro regime."""
        scores = {
            "Inflationary": 0,
            "Disinflationary": 0,
            "Deflationary": 0,
            "Risk-On": 0,
            "Risk-Off": 0,
            "Tightening": 0,
            "Easing": 0,
            "Stagflationary": 0
        }

        # Energy & Yield rules
        if oil_return_20d > 0.05:
            scores["Inflationary"] += 2
            scores["Stagflationary"] += 1
        elif oil_return_20d < -0.05:
            scores["Disinflationary"] += 1
            scores["Deflationary"] += 1

        if us10y_change_20d > 0.20: # 20 bps rise
            scores["Tightening"] += 2
            scores["Inflationary"] += 1
        elif us10y_change_20d < -0.20:
            scores["Easing"] += 2
            scores["Disinflationary"] += 1

        # Equities & Volatility rules
        if sp500_return_20d > 0.02 and vix_level < 20.0:
            scores["Risk-On"] += 2
        elif sp500_return_20d < -0.03 or vix_level > 25.0:
            scores["Risk-Off"] += 2

        # Stagflation: High inflation indicators + falling equities
        if oil_return_20d > 0.03 and sp500_return_20d < -0.02:
            scores["Stagflationary"] += 2

        primary_regime = max(scores, key=scores.get)
        if scores[primary_regime] == 0:
            primary_regime = "Neutral"

        return {
            "primary_macro_regime": primary_regime,
            "regime_scores": scores,
            "inputs": {
                "us10y_change_20d": us10y_change_20d,
                "dxy_return_20d": dxy_return_20d,
                "oil_return_20d": oil_return_20d,
                "sp500_return_20d": sp500_return_20d,
                "vix_level": vix_level
            }
        }


class MarketRegimeClassifier:
    """Classifies asset-specific market regime combining Trend, Volatility, and Macro components."""

    @staticmethod
    def classify_market_regime(
        price: float,
        sma_20: float,
        sma_50: float,
        sma_200: float,
        volatility_20d: float,
        rsi_14: float,
        macro_regime: str
    ) -> Dict[str, Any]:
        # Trend Component
        if price > sma_20 > sma_50 > sma_200:
            trend = "Strong Bullish Trend"
        elif price > sma_50 and sma_20 > sma_50:
            trend = "Bullish Trend"
        elif price < sma_20 < sma_50 < sma_200:
            trend = "Strong Bearish Trend"
        elif price < sma_50 and sma_20 < sma_50:
            trend = "Bearish Trend"
        else:
            trend = "Range"

        # Volatility Component
        if volatility_20d > 0.25:
            vol_regime = "High Volatility"
        elif volatility_20d < 0.12:
            vol_regime = "Low Volatility"
        else:
            vol_regime = "Moderate Volatility"

        # Momentum Component
        if rsi_14 > 70:
            momentum = "Overbought"
        elif rsi_14 < 30:
            momentum = "Oversold"
        else:
            momentum = "Neutral"

        return {
            "trend_regime": trend,
            "volatility_regime": vol_regime,
            "momentum_regime": momentum,
            "macro_context": macro_regime,
            "combined_label": f"{trend} | {vol_regime} | Macro: {macro_regime}"
        }
