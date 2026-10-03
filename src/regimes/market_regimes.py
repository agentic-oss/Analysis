import pandas as pd
import numpy as np
from typing import Dict, Any, List


class AssetMarketRegimeModel:
    """Classifies market regime for specific assets (e.g. Gold, Silver)."""

    @staticmethod
    def classify_market_regime(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Classifies trend regime and volatility regime simultaneously.
        Trend: Strong Bullish Trend, Bullish Trend, Range, Bearish Trend, Strong Bearish Trend
        Volatility: High Volatility, Low Volatility, Normal Volatility
        """
        if df is None or df.empty or len(df) < 50:
            return {
                "trend_regime": "Range",
                "volatility_regime": "Normal Volatility",
                "trend_strength": 0.0,
                "distance_200dma_pct": 0.0,
                "rsi": 50.0,
            }

        latest = df.iloc[-1]
        c = float(latest["Close"])
        sma20 = float(latest.get("sma_20", c))
        sma50 = float(latest.get("sma_50", c))
        sma200 = float(latest.get("sma_200", c))
        rsi = float(latest.get("rsi_14", 50.0))
        vol20 = float(latest.get("volatility_20d", 0.15))
        vol252 = float(latest.get("volatility_252d", 0.15))

        dist_200 = (c - sma200) / (sma200 + 1e-9)

        # Trend classification
        if c > sma20 > sma50 > sma200 and rsi > 55:
            trend_regime = "Strong Bullish Trend"
        elif c > sma50 and sma50 > sma200:
            trend_regime = "Bullish Trend"
        elif c < sma20 < sma50 < sma200 and rsi < 45:
            trend_regime = "Strong Bearish Trend"
        elif c < sma50 and sma50 < sma200:
            trend_regime = "Bearish Trend"
        else:
            trend_regime = "Range"

        # Volatility classification
        if vol20 > vol252 * 1.3:
            vol_regime = "High Volatility"
        elif vol20 < vol252 * 0.7:
            vol_regime = "Low Volatility"
        else:
            vol_regime = "Normal Volatility"

        return {
            "trend_regime": trend_regime,
            "volatility_regime": vol_regime,
            "distance_200dma_pct": round(dist_200 * 100, 2),
            "rsi": round(rsi, 1),
            "volatility_20d_ann_pct": round(vol20 * 100, 2),
        }
