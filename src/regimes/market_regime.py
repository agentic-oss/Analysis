import pandas as pd
import numpy as np
from typing import Dict, List, Any


class MarketRegimeDetector:
    """Independent multi-factor market regime detector for Gold and Silver."""

    @staticmethod
    def detect_market_regime(df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        if df is None or df.empty:
            return {"symbol": symbol, "trend_regime": "Neutral", "volatility_regime": "Normal Volatility"}

        latest = df.iloc[-1]
        close = float(latest["close"])
        sma20 = float(latest.get("sma_20", close))
        sma50 = float(latest.get("sma_50", close))
        sma200 = float(latest.get("sma_200", close))
        rsi = float(latest.get("rsi_14", 50.0))
        vol_20d = float(latest.get("volatility_20d", 15.0))

        # 1. Trend Regime Classification
        if close > sma20 > sma50 > sma200 and rsi > 55:
            trend_regime = "Strong Bullish Trend"
        elif close > sma50 and rsi > 50:
            trend_regime = "Bullish Trend"
        elif close < sma20 < sma50 < sma200 and rsi < 45:
            trend_regime = "Strong Bearish Trend"
        elif close < sma50 and rsi < 50:
            trend_regime = "Bearish Trend"
        else:
            trend_regime = "Range"

        # 2. Volatility Regime Classification
        hist_vol_mean = df["volatility_20d"].rolling(60, min_periods=5).mean().iloc[-1] if "volatility_20d" in df.columns else 15.0
        if pd.isna(hist_vol_mean):
            hist_vol_mean = 15.0

        if vol_20d > hist_vol_mean * 1.3:
            volatility_regime = "High Volatility"
        elif vol_20d < hist_vol_mean * 0.7:
            volatility_regime = "Low Volatility"
        else:
            volatility_regime = "Normal Volatility"

        # Composite multi-tag summary
        multi_tags = {
            "trend": trend_regime,
            "volatility": volatility_regime,
            "rsi_state": "Overbought" if rsi >= 70 else ("Oversold" if rsi <= 30 else "Neutral")
        }

        return {
            "symbol": symbol,
            "trend_regime": trend_regime,
            "volatility_regime": volatility_regime,
            "multi_tags": multi_tags,
            "rsi": round(rsi, 2),
            "volatility_20d": round(vol_20d, 2),
            "dist_sma_200_pct": round(float(latest.get("dist_sma_200", 0.0)), 2)
        }
