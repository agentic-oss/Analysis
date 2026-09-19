import pandas as pd
import numpy as np
from typing import Dict, Any, List

class MetalAnalyzer:
    """Dedicated analysis for individual precious metals (Gold / Silver)."""

    @staticmethod
    def analyze_metal(
        metal_df: pd.DataFrame,
        symbol: str,
        macro_df: pd.DataFrame
    ) -> Dict[str, Any]:
        if metal_df.empty:
            return {}

        latest = metal_df.iloc[-1]
        price = float(latest.get("close", 0.0))

        # Compute key statistics
        analysis = {
            "symbol": symbol,
            "current_price": price,
            "date": str(latest.get("date", "")),
            "daily_return_pct": float(latest.get("return_1d", 0.0)),
            "weekly_return_pct": float(latest.get("return_5d", 0.0)),
            "monthly_return_pct": float(latest.get("return_20d", 0.0)),
            "rsi_14": float(latest.get("rsi_14", 50.0)),
            "macd": float(latest.get("macd", 0.0)),
            "atr_14": float(latest.get("atr_14", 0.0)),
            "volatility_20d": float(latest.get("volatility_20d", 0.0)),
            "sma_20": float(latest.get("sma_20", price)),
            "sma_50": float(latest.get("sma_50", price)),
            "sma_200": float(latest.get("sma_200", price)),
            "dist_sma_200_pct": float(latest.get("dist_sma_200_pct", 0.0)),
            "dist_52w_high_pct": float(latest.get("dist_52w_high_pct", 0.0)),
            "dist_52w_low_pct": float(latest.get("dist_52w_low_pct", 0.0)),
            "drawdown_pct": float(latest.get("drawdown_pct", 0.0)),
        }
        return analysis
