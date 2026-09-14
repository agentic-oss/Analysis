"""
Macro Regime Detection Engine.
Transparent, rule-based macro regime classification (Inflationary, Disinflationary, Deflationary,
Risk-on, Risk-off, Tightening, Easing, Stagflationary, Neutral).
Stores all individual components for explainability.
"""
import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MacroRegimeDetector:
    """
    Transparent rule-based Macro Regime Model incorporating:
      - DXY (US Dollar Index)
      - Interest Rates (US 10Y Yield)
      - Inflation / Commodity proxy (Crude Oil WTI)
      - Equity performance (S&P 500)
      - Volatility (VIX)
    """

    @staticmethod
    def detect_regime(
        dxy_df: pd.DataFrame,
        us10y_df: pd.DataFrame,
        oil_df: pd.DataFrame,
        sp500_df: pd.DataFrame,
        vix_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detects macro regime based on recent 20-day returns/changes of macro drivers.
        Returns regime classification and explainable score components.
        """
        components = {
            "dxy_trend": "neutral",
            "rates_trend": "neutral",
            "oil_trend": "neutral",
            "equity_trend": "neutral",
            "vix_level": "normal"
        }

        dxy_20d_ret = 0.0
        us10y_20d_change = 0.0
        oil_20d_ret = 0.0
        sp500_20d_ret = 0.0
        latest_vix = 15.0

        if not dxy_df.empty and "close" in dxy_df.columns and len(dxy_df) >= 20:
            dxy_20d_ret = (dxy_df["close"].iloc[-1] - dxy_df["close"].iloc[-20]) / dxy_df["close"].iloc[-20]
            components["dxy_trend"] = "strong_dollar" if dxy_20d_ret > 0.015 else ("weak_dollar" if dxy_20d_ret < -0.015 else "neutral")

        if not us10y_df.empty and "close" in us10y_df.columns and len(us10y_df) >= 20:
            us10y_20d_change = us10y_df["close"].iloc[-1] - us10y_df["close"].iloc[-20]
            components["rates_trend"] = "rising_rates" if us10y_20d_change > 0.20 else ("falling_rates" if us10y_20d_change < -0.20 else "neutral")

        if not oil_df.empty and "close" in oil_df.columns and len(oil_df) >= 20:
            oil_20d_ret = (oil_df["close"].iloc[-1] - oil_df["close"].iloc[-20]) / oil_df["close"].iloc[-20]
            components["oil_trend"] = "rising_commodities" if oil_20d_ret > 0.04 else ("falling_commodities" if oil_20d_ret < -0.04 else "neutral")

        if not sp500_df.empty and "close" in sp500_df.columns and len(sp500_df) >= 20:
            sp500_20d_ret = (sp500_df["close"].iloc[-1] - sp500_df["close"].iloc[-20]) / sp500_df["close"].iloc[-20]
            components["equity_trend"] = "bullish_equities" if sp500_20d_ret > 0.02 else ("bearish_equities" if sp500_20d_ret < -0.02 else "neutral")

        if not vix_df.empty and "close" in vix_df.columns:
            latest_vix = float(vix_df["close"].iloc[-1])
            components["vix_level"] = "high_fear" if latest_vix > 25.0 else ("low_fear" if latest_vix < 14.0 else "normal")

        # Rule evaluation for macro regime
        regime = "Neutral"

        if components["vix_level"] == "high_fear" or components["equity_trend"] == "bearish_equities":
            if components["oil_trend"] == "rising_commodities":
                regime = "Stagflationary"
            else:
                regime = "Risk-Off"
        elif components["rates_trend"] == "rising_rates" and components["oil_trend"] == "rising_commodities":
            regime = "Inflationary"
        elif components["rates_trend"] == "rising_rates" and components["dxy_trend"] == "strong_dollar":
            regime = "Tightening"
        elif components["rates_trend"] == "falling_rates" and components["equity_trend"] == "bullish_equities":
            if components["oil_trend"] == "falling_commodities":
                regime = "Disinflationary"
            else:
                regime = "Easing"
        elif components["equity_trend"] == "bullish_equities" and components["vix_level"] == "low_fear":
            regime = "Risk-On"

        return {
            "macro_regime": regime,
            "components": components,
            "metrics": {
                "dxy_20d_return": float(dxy_20d_ret),
                "us10y_20d_change": float(us10y_20d_change),
                "oil_20d_return": float(oil_20d_ret),
                "sp500_20d_return": float(sp500_20d_ret),
                "vix_level": float(latest_vix)
            }
        }
