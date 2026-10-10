import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class MarketRegime:
    """
    Classifies precious metal market state along trend, volatility, and momentum dimensions.
    """

    @staticmethod
    def classify_metal_regime(df_tech: pd.DataFrame) -> Dict[str, Any]:
        if df_tech.empty or "Close" not in df_tech.columns:
            return {}

        row = df_tech.iloc[-1]
        close = row["Close"]
        sma_20 = row.get("sma_20", close)
        sma_50 = row.get("sma_50", close)
        sma_200 = row.get("sma_200", close)
        vol_20d = row.get("volatility_20d", 0.15)
        rsi = row.get("rsi_14", 50.0)

        # Trend classification
        if close > sma_20 > sma_50 > sma_200:
            trend = "Strong Bullish Trend"
        elif close > sma_50:
            trend = "Bullish Trend"
        elif close < sma_20 < sma_50 < sma_200:
            trend = "Strong Bearish Trend"
        elif close < sma_50:
            trend = "Bearish Trend"
        else:
            trend = "Range-Bound / Neutral"

        # Volatility classification
        if vol_20d > 0.25:
            volatility = "High Volatility"
        elif vol_20d < 0.12:
            volatility = "Low Volatility"
        else:
            volatility = "Normal Volatility"

        # Momentum state
        if rsi >= 70:
            momentum_state = "Overbought"
        elif rsi <= 30:
            momentum_state = "Oversold"
        else:
            momentum_state = "Neutral Momentum"

        return {
            "trend": trend,
            "volatility": volatility,
            "momentum_state": momentum_state,
            "composite_regime": f"{trend} | {volatility} | {momentum_state}"
        }
