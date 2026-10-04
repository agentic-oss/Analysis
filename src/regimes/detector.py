import pandas as pd
import numpy as np
from typing import Dict, Any, List

class RegimeClassifier:
    """Classifies transparent, explainable macro and instrument market regimes."""

    @staticmethod
    def classify_macro_regime(
        dxy_df: pd.DataFrame,
        us10y_df: pd.DataFrame,
        tip_df: pd.DataFrame,
        oil_df: pd.DataFrame,
        sp500_df: pd.DataFrame,
        vix_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Combine interest rates, real yield proxy, equities, energy, and volatility into an explainable macro regime."""
        components = {
            "dxy_return_20d": 0.0,
            "us10y_change_20d": 0.0,
            "sp500_return_20d": 0.0,
            "oil_return_20d": 0.0,
            "vix_level": 20.0
        }

        if dxy_df is not None and not dxy_df.empty and len(dxy_df) >= 20:
            components["dxy_return_20d"] = float(dxy_df["close"].pct_change(20).iloc[-1] * 100)

        if us10y_df is not None and not us10y_df.empty and len(us10y_df) >= 20:
            components["us10y_change_20d"] = float(us10y_df["close"].diff(20).iloc[-1])

        if sp500_df is not None and not sp500_df.empty and len(sp500_df) >= 20:
            components["sp500_return_20d"] = float(sp500_df["close"].pct_change(20).iloc[-1] * 100)

        if oil_df is not None and not oil_df.empty and len(oil_df) >= 20:
            components["oil_return_20d"] = float(oil_df["close"].pct_change(20).iloc[-1] * 100)

        if vix_df is not None and not vix_df.empty:
            components["vix_level"] = float(vix_df["close"].iloc[-1])

        # Rule-based classification
        oil_ret = components["oil_return_20d"]
        sp_ret = components["sp500_return_20d"]
        vix = components["vix_level"]
        rates_change = components["us10y_change_20d"]

        if oil_ret > 5.0 and sp_ret < -2.0:
            regime = "Stagflationary"
        elif oil_ret > 5.0 and sp_ret >= 0:
            regime = "Inflationary"
        elif vix > 25.0 or sp_ret < -4.0:
            regime = "Risk-Off"
        elif sp_ret > 3.0 and vix < 18.0:
            regime = "Risk-On"
        elif rates_change > 0.3:
            regime = "Tightening"
        elif rates_change < -0.3:
            regime = "Easing"
        elif oil_ret < -5.0 and sp_ret < 0:
            regime = "Disinflationary"
        else:
            regime = "Neutral"

        return {
            "macro_regime": regime,
            "explainable_components": components
        }

    @staticmethod
    def classify_market_regime(
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Classify instrument market regime using Trend state (20/50/200 MAs) and Volatility state (ATR / Volatility)."""
        if df.empty or len(df) < 20:
            return {
                "trend_regime": "Range",
                "volatility_regime": "Normal Volatility",
                "composite_market_regime": "Range - Normal Volatility"
            }

        latest = df.iloc[-1]
        close = latest["close"]
        sma_20 = latest.get("sma_20", close)
        sma_50 = latest.get("sma_50", close)
        sma_200 = latest.get("sma_200", close)
        vol_20d = latest.get("volatility_20d", 0.15)

        # Trend Regime
        if close > sma_20 and sma_20 > sma_50 and sma_50 > sma_200:
            trend = "Strong Bullish Trend"
        elif close > sma_50:
            trend = "Bullish Trend"
        elif close < sma_20 and sma_20 < sma_50 and sma_50 < sma_200:
            trend = "Strong Bearish Trend"
        elif close < sma_50:
            trend = "Bearish Trend"
        else:
            trend = "Range"

        # Volatility Regime
        if vol_20d > 0.25:
            vol = "High Volatility"
        elif vol_20d < 0.12:
            vol = "Low Volatility"
        else:
            vol = "Normal Volatility"

        return {
            "trend_regime": trend,
            "volatility_regime": vol,
            "composite_market_regime": f"{trend} | {vol}"
        }
