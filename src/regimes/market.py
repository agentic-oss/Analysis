"""Metal-specific market regime classifier (Trend and Volatility)."""

import pandas as pd
import numpy as np
from typing import Dict, Any


class MarketRegimeClassifier:
    """Classifies gold and silver market regimes across Trend, Volatility, and Momentum dimensions."""

    @staticmethod
    def classify_market(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyzes price dataframe with technicals to evaluate market regime."""
        if df.empty or len(df) < 20:
            return {
                "trend_regime": "Neutral",
                "volatility_regime": "Normal Volatility",
                "composite_label": "Neutral / Normal Volatility"
            }

        latest = df.iloc[-1]
        close = float(latest.get('close', 0.0))
        sma_20 = float(latest.get('sma_20', close))
        sma_50 = float(latest.get('sma_50', close))
        sma_200 = float(latest.get('sma_200', close))
        volatility_20d = float(latest.get('volatility_20d', 0.15))

        # Trend Classification
        if close > sma_20 > sma_50 > sma_200:
            trend = "Strong Bullish Trend"
        elif close > sma_50:
            trend = "Bullish Trend"
        elif close < sma_20 < sma_50 < sma_200:
            trend = "Strong Bearish Trend"
        elif close < sma_50:
            trend = "Bearish Trend"
        else:
            trend = "Range"

        # Volatility Classification
        if volatility_20d > 0.25:
            vol = "High Volatility"
        elif volatility_20d < 0.10:
            vol = "Low Volatility"
        else:
            vol = "Normal Volatility"

        composite = f"{trend} | {vol}"

        return {
            "trend_regime": trend,
            "volatility_regime": vol,
            "volatility_20d_pct": round(volatility_20d * 100, 2),
            "dist_sma_50_pct": round(float(latest.get('dist_sma_50_pct', 0.0)), 2),
            "dist_sma_200_pct": round(float(latest.get('dist_sma_200_pct', 0.0)), 2),
            "composite_label": composite
        }
