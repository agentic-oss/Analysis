import pandas as pd
import numpy as np
from typing import Dict, Any

class MacroRegimeClassifier:
    """
    Transparent, explainable rule-based macro regime classifier.
    Combines interest rates, real yields, DXY, inflation expectations, equity market volatility, and crude oil.
    """

    REGIMES = [
        "Inflationary",
        "Disinflationary",
        "Deflationary",
        "Risk-On",
        "Risk-Off",
        "Tightening",
        "Easing",
        "Stagflationary",
        "Neutral"
    ]

    def classify_macro_regime(
        self,
        dxy_return_20d: float,
        us10y_change_20d: float,
        oil_return_20d: float,
        sp500_return_20d: float,
        vix_level: float,
        real_yield_change_20d: float = 0.0
    ) -> Dict[str, Any]:
        """Classifies current macro regime with full component transparency."""

        components = {
            "dxy_trend": "Strengthening" if dxy_return_20d > 0.01 else ("Weakening" if dxy_return_20d < -0.01 else "Neutral"),
            "rate_trend": "Rising" if us10y_change_20d > 0.15 else ("Falling" if us10y_change_20d < -0.15 else "Neutral"),
            "oil_trend": "Rising" if oil_return_20d > 0.03 else ("Falling" if oil_return_20d < -0.03 else "Neutral"),
            "equity_trend": "Bullish" if sp500_return_20d > 0.02 else ("Bearish" if sp500_return_20d < -0.02 else "Neutral"),
            "volatility_env": "High Vol / Risk Off" if vix_level > 22.0 else ("Low Vol / Risk On" if vix_level < 15.0 else "Normal Vol")
        }

        # Rule evaluation
        regime = "Neutral"
        reasons = []

        if components["volatility_env"] == "High Vol / Risk Off" or (sp500_return_20d < -0.04 and dxy_return_20d > 0):
            regime = "Risk-Off"
            reasons.append("Elevated equity volatility and broad market risk aversion")

        elif components["oil_trend"] == "Rising" and components["rate_trend"] == "Rising":
            if components["equity_trend"] == "Bearish":
                regime = "Stagflationary"
                reasons.append("Rising energy prices, rising yields, and falling equity markets")
            else:
                regime = "Inflationary"
                reasons.append("Surging commodity prices and rising bond yields")

        elif components["rate_trend"] == "Rising" and components["dxy_trend"] == "Strengthening":
            regime = "Tightening"
            reasons.append("Tight monetary environment with rising yields and firm USD")

        elif components["rate_trend"] == "Falling" and components["dxy_trend"] == "Weakening":
            regime = "Easing"
            reasons.append("Accommodative monetary environment with falling yields and soft USD")

        elif components["equity_trend"] == "Bullish" and components["volatility_env"] == "Low Vol / Risk On":
            regime = "Risk-On"
            reasons.append("Strong equity momentum and low volatility regime")

        elif components["oil_trend"] == "Falling" and components["rate_trend"] == "Falling":
            regime = "Disinflationary"
            reasons.append("Cooling commodity prices and declining interest rates")

        return {
            "regime": regime,
            "components": components,
            "reasons": reasons,
            "inputs": {
                "dxy_return_20d": round(dxy_return_20d, 4),
                "us10y_change_20d": round(us10y_change_20d, 4),
                "oil_return_20d": round(oil_return_20d, 4),
                "sp500_return_20d": round(sp500_return_20d, 4),
                "vix_level": round(vix_level, 2)
            }
        }


class MarketRegimeClassifier:
    """Classifies gold/silver specific market regimes (Trend x Volatility)."""

    def classify_market_regime(
        self,
        close_price: float,
        sma_20: float,
        sma_50: float,
        sma_200: float,
        rsi_14: float,
        hist_vol_20: float
    ) -> Dict[str, Any]:
        """Determines multi-dimensional market regime for a single instrument."""

        # Trend component
        if close_price > sma_20 > sma_50 > sma_200:
            trend = "Strong Bullish Trend"
        elif close_price > sma_50 and close_price > sma_200:
            trend = "Bullish Trend"
        elif close_price < sma_20 < sma_50 < sma_200:
            trend = "Strong Bearish Trend"
        elif close_price < sma_50 and close_price < sma_200:
            trend = "Bearish Trend"
        else:
            trend = "Range / Sideways"

        # Volatility component
        if hist_vol_20 >= 0.25:
            vol_env = "High Volatility"
        elif hist_vol_20 <= 0.12:
            vol_env = "Low Volatility"
        else:
            vol_env = "Normal Volatility"

        # Momentum state
        if rsi_14 >= 70:
            mom_state = "Overbought"
        elif rsi_14 <= 30:
            mom_state = "Oversold"
        else:
            mom_state = "Neutral Momentum"

        composite_label = f"{trend} | {vol_env} | {mom_state}"

        return {
            "composite_regime": composite_label,
            "trend": trend,
            "volatility": vol_env,
            "momentum": mom_state,
            "metrics": {
                "rsi_14": round(rsi_14, 1),
                "hist_vol_20": round(hist_vol_20, 4)
            }
        }
