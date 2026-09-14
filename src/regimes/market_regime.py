"""
Precious Metals Market Regime Detection Module.
Classifies Gold and Silver markets into distinct Trend and Volatility regimes.
"""
import logging
from typing import Dict, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """
    Classifies Gold & Silver market regimes into independent Trend & Volatility dimensions:
      - Trend: Strong Bullish, Bullish, Range, Bearish, Strong Bearish
      - Volatility: High Volatility, Normal Volatility, Low Volatility
    Returns composite label and breakdown.
    """

    @staticmethod
    def classify_market_regime(df_metal: pd.DataFrame) -> Dict[str, Any]:
        """
        Classifies market regime based on technical indicators (SMA_20, SMA_50, SMA_200, RSI, Volatility).
        `df_metal` must contain calculated technical indicator columns.
        """
        if df_metal.empty or "close" not in df_metal.columns or len(df_metal) < 20:
            return {
                "trend_regime": "Range",
                "volatility_regime": "Normal Volatility",
                "composite_regime": "Range / Normal Volatility",
                "metrics": {}
            }

        last_row = df_metal.iloc[-1]
        close = float(last_row["close"])
        sma_20 = float(last_row.get("sma_20", close))
        sma_50 = float(last_row.get("sma_50", close))
        sma_200 = float(last_row.get("sma_200", close))
        rsi_14 = float(last_row.get("rsi_14", 50.0))

        # 1. Trend Classification
        if close > sma_20 > sma_50 > sma_200 and rsi_14 > 55:
            trend = "Strong Bullish Trend"
        elif close > sma_50 and sma_20 > sma_50:
            trend = "Bullish Trend"
        elif close < sma_20 < sma_50 < sma_200 and rsi_14 < 45:
            trend = "Strong Bearish Trend"
        elif close < sma_50 and sma_20 < sma_50:
            trend = "Bearish Trend"
        else:
            trend = "Range"

        # 2. Volatility Classification
        vol_20d = float(last_row.get("volatility_20d", 0.15))
        hist_vol_mean = df_metal["volatility_20d"].dropna().rolling(window=120, min_periods=20).mean().iloc[-1] if "volatility_20d" in df_metal.columns else vol_20d
        if np.isnan(hist_vol_mean) or hist_vol_mean == 0:
            hist_vol_mean = vol_20d

        if vol_20d > 1.3 * hist_vol_mean:
            volatility = "High Volatility"
        elif vol_20d < 0.7 * hist_vol_mean:
            volatility = "Low Volatility"
        else:
            volatility = "Normal Volatility"

        composite = f"{trend} / {volatility}"

        return {
            "trend_regime": trend,
            "volatility_regime": volatility,
            "composite_regime": composite,
            "metrics": {
                "close": close,
                "sma_200": sma_200,
                "rsi_14": rsi_14,
                "volatility_20d": vol_20d,
                "hist_vol_mean": float(hist_vol_mean)
            }
        }
