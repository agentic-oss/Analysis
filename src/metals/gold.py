"""
Gold-specific and Silver-specific Analysis Modules.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any


class GoldAnalyzer:
    """
    Dedicated analysis engine for Gold market structures.
    """

    @staticmethod
    def analyze_gold(gold_df: pd.DataFrame, usd_inr_rate: float = 83.0) -> Dict[str, Any]:
        if gold_df.empty:
            return {}

        latest = gold_df.iloc[-1]
        price_usd_oz = float(latest["close"])

        # Unit & Currency Conversions:
        # 1 troy oz = 31.1034768 grams. 10g Gold = (Price per oz / 31.1034768) * 10 * USDINR
        price_inr_10g = (price_usd_oz / 31.1034768) * 10.0 * usd_inr_rate

        # Historical returns
        ret_1d = float((gold_df["close"].pct_change(1).iloc[-1]) * 100) if len(gold_df) > 1 else 0.0
        ret_5d = float((gold_df["close"].pct_change(5).iloc[-1]) * 100) if len(gold_df) > 5 else 0.0
        ret_20d = float((gold_df["close"].pct_change(20).iloc[-1]) * 100) if len(gold_df) > 20 else 0.0
        ret_252d = float((gold_df["close"].pct_change(252).iloc[-1]) * 100) if len(gold_df) > 252 else 0.0

        # Drawdown
        cummax = gold_df["close"].cummax()
        drawdown = (gold_df["close"] - cummax) / cummax
        current_drawdown = float(drawdown.iloc[-1] * 100)
        max_drawdown = float(drawdown.min() * 100)

        return {
            "symbol": "GOLD",
            "date": str(latest["date"]),
            "price_usd_oz": price_usd_oz,
            "price_inr_10g": price_inr_10g,
            "conversion_usdinr": usd_inr_rate,
            "return_1d_pct": ret_1d,
            "return_5d_pct": ret_5d,
            "return_20d_pct": ret_20d,
            "return_1y_pct": ret_252d,
            "current_drawdown_pct": current_drawdown,
            "max_drawdown_pct": max_drawdown,
            "rsi_14": float(latest.get("rsi_14", 50.0)),
            "sma_50": float(latest.get("sma_50", price_usd_oz)),
            "sma_200": float(latest.get("sma_200", price_usd_oz)),
            "dist_sma_200_pct": float(latest.get("dist_sma_200", 0.0) * 100),
            "golden_cross": bool(latest.get("golden_cross", False)),
            "death_cross": bool(latest.get("death_cross", False)),
        }


class SilverAnalyzer:
    """
    Dedicated analysis engine for Silver market structures.
    """

    @staticmethod
    def analyze_silver(silver_df: pd.DataFrame, usd_inr_rate: float = 83.0) -> Dict[str, Any]:
        if silver_df.empty:
            return {}

        latest = silver_df.iloc[-1]
        price_usd_oz = float(latest["close"])

        # Unit & Currency Conversions:
        # 1 troy oz = 0.0311034768 kg. 1 kg Silver = (Price per oz / 31.1034768) * 1000 * USDINR
        price_inr_kg = (price_usd_oz / 31.1034768) * 1000.0 * usd_inr_rate

        ret_1d = float((silver_df["close"].pct_change(1).iloc[-1]) * 100) if len(silver_df) > 1 else 0.0
        ret_5d = float((silver_df["close"].pct_change(5).iloc[-1]) * 100) if len(silver_df) > 5 else 0.0
        ret_20d = float((silver_df["close"].pct_change(20).iloc[-1]) * 100) if len(silver_df) > 20 else 0.0
        ret_252d = float((silver_df["close"].pct_change(252).iloc[-1]) * 100) if len(silver_df) > 252 else 0.0

        cummax = silver_df["close"].cummax()
        drawdown = (silver_df["close"] - cummax) / cummax
        current_drawdown = float(drawdown.iloc[-1] * 100)
        max_drawdown = float(drawdown.min() * 100)

        return {
            "symbol": "SILVER",
            "date": str(latest["date"]),
            "price_usd_oz": price_usd_oz,
            "price_inr_kg": price_inr_kg,
            "conversion_usdinr": usd_inr_rate,
            "return_1d_pct": ret_1d,
            "return_5d_pct": ret_5d,
            "return_20d_pct": ret_20d,
            "return_1y_pct": ret_252d,
            "current_drawdown_pct": current_drawdown,
            "max_drawdown_pct": max_drawdown,
            "rsi_14": float(latest.get("rsi_14", 50.0)),
            "sma_50": float(latest.get("sma_50", price_usd_oz)),
            "sma_200": float(latest.get("sma_200", price_usd_oz)),
            "dist_sma_200_pct": float(latest.get("dist_sma_200", 0.0) * 100),
            "golden_cross": bool(latest.get("golden_cross", False)),
            "death_cross": bool(latest.get("death_cross", False)),
        }
