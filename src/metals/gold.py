import pandas as pd
import numpy as np
from typing import Dict, List, Any


class GoldAnalyzer:
    """Dedicated analysis module for Gold market structure, key levels, and macro sensitivities."""

    @staticmethod
    def analyze(
        gold_df: pd.DataFrame,
        macro_dict: Dict[str, pd.DataFrame] = None
    ) -> Dict[str, Any]:
        if gold_df is None or gold_df.empty:
            return {}

        latest = gold_df.iloc[-1]

        # Key Support and Resistance Calculation
        close_price = float(latest["close"])
        sma20 = float(latest.get("sma_20", close_price))
        sma50 = float(latest.get("sma_50", close_price))
        sma200 = float(latest.get("sma_200", close_price))

        recent_20 = gold_df.tail(20)
        high_20d = float(recent_20["high"].max()) if "high" in recent_20.columns else close_price
        low_20d = float(recent_20["low"].min()) if "low" in recent_20.columns else close_price

        supports = sorted([p for p in [sma20, sma50, sma200, low_20d] if p < close_price], reverse=True)
        resistances = sorted([p for p in [sma20, sma50, sma200, high_20d] if p > close_price])

        key_support = supports[0] if supports else close_price * 0.98
        key_resistance = resistances[0] if resistances else close_price * 1.02

        # Research Bias / Trend
        rsi = float(latest.get("rsi_14", 50.0))
        macd_hist = float(latest.get("macd_hist", 0.0))
        dist_200 = float(latest.get("dist_sma_200", 0.0))

        if close_price > sma50 and rsi > 50 and macd_hist > 0:
            trend = "Bullish"
            research_bias = "Bullish Bias"
        elif close_price < sma50 and rsi < 50 and macd_hist < 0:
            trend = "Bearish"
            research_bias = "Bearish Bias"
        else:
            trend = "Neutral / Ranging"
            research_bias = "Neutral Bias"

        vol_20d = float(latest.get("volatility_20d", 15.0))
        vol_state = "High Volatility" if vol_20d > 20.0 else ("Low Volatility" if vol_20d < 10.0 else "Normal Volatility")

        analysis = {
            "symbol": "GOLD",
            "price": round(close_price, 2),
            "price_inr_10g": round(float(latest["price_inr"]), 2) if "price_inr" in latest and not pd.isna(latest["price_inr"]) else None,
            "daily_move_pct": round(float(latest.get("return_1d", 0.0) or 0.0) * 100, 2),
            "weekly_move_pct": round(float(latest.get("return_5d", 0.0) or 0.0) * 100, 2),
            "monthly_move_pct": round(float(latest.get("return_20d", 0.0) or 0.0) * 100, 2),
            "trend": trend,
            "volatility": vol_state,
            "volatility_20d_ann": round(vol_20d, 2),
            "research_bias": research_bias,
            "technical_indicators": {
                "rsi_14": round(rsi, 2),
                "macd": round(float(latest.get("macd", 0.0)), 2),
                "macd_signal": round(float(latest.get("macd_signal", 0.0)), 2),
                "macd_hist": round(macd_hist, 2),
                "sma_20": round(sma20, 2),
                "sma_50": round(sma50, 2),
                "sma_200": round(sma200, 2),
                "dist_sma_200_pct": round(dist_200, 2)
            },
            "key_levels": {
                "key_support": round(key_support, 2),
                "key_resistance": round(key_resistance, 2),
                "high_20d": round(high_20d, 2),
                "low_20d": round(low_20d, 2)
            }
        }

        return analysis
