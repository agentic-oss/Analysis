"""
Market Regime Classification Engine for Precious Metals.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any


class MarketRegimeDetector:
    """
    Detects metal market trend regimes (Strong Bullish, Bullish, Range, Bearish, Strong Bearish)
    and volatility regimes (High Volatility, Normal Volatility, Low Volatility).
    """

    @staticmethod
    def detect_market_regime(df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        if df.empty or len(df) < 20:
            return {
                "symbol": symbol,
                "trend_regime": "Neutral",
                "volatility_regime": "Normal Volatility",
                "composite_regime": "Neutral | Normal Volatility",
            }

        latest = df.iloc[-1]
        close = float(latest["close"])
        sma20 = float(latest.get("sma_20", close))
        sma50 = float(latest.get("sma_50", close))
        sma200 = float(latest.get("sma_200", close))
        rsi = float(latest.get("rsi_14", 50.0))
        vol20 = float(latest.get("volatility_20d", 0.15))

        # Trend classification
        if close > sma20 > sma50 > sma200 and rsi > 55:
            trend_regime = "Strong Bullish Trend"
        elif close > sma50:
            trend_regime = "Bullish Trend"
        elif close < sma20 < sma50 < sma200 and rsi < 45:
            trend_regime = "Strong Bearish Trend"
        elif close < sma50:
            trend_regime = "Bearish Trend"
        else:
            trend_regime = "Range / Sideways"

        # Volatility classification
        if vol20 > 0.25:
            vol_regime = "High Volatility"
        elif vol20 < 0.12:
            vol_regime = "Low Volatility"
        else:
            vol_regime = "Normal Volatility"

        composite = f"{trend_regime} | {vol_regime}"

        return {
            "symbol": symbol,
            "trend_regime": trend_regime,
            "volatility_regime": vol_regime,
            "composite_regime": composite,
            "rsi": rsi,
            "volatility_20d": vol20,
        }
