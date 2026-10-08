"""
Gold/Silver Ratio & Relative-Value Engine.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any


class GoldSilverRatio:
    """
    Computes Gold/Silver ratio, historical percentiles, Z-scores, mean-reversion metrics, and relative return statistics.
    """

    @staticmethod
    def calculate_ratio(gold_df: pd.DataFrame, silver_df: pd.DataFrame) -> pd.DataFrame:
        if gold_df.empty or silver_df.empty:
            return pd.DataFrame()

        g = gold_df[["date", "close"]].rename(columns={"close": "gold_price"})
        s = silver_df[["date", "close"]].rename(columns={"close": "silver_price"})

        merged = pd.merge(g, s, on="date", how="inner").sort_values("date")
        merged["gold_silver_ratio"] = merged["gold_price"] / merged["silver_price"].replace(0, np.nan)

        # Calculate Gold return minus Silver return (spread)
        merged["gold_return_1d"] = merged["gold_price"].pct_change(1)
        merged["silver_return_1d"] = merged["silver_price"].pct_change(1)
        merged["spread_return_1d"] = merged["gold_return_1d"] - merged["silver_return_1d"]

        # Rolling ratio stats (252 days)
        window = 252
        merged["ratio_sma_252"] = merged["gold_silver_ratio"].rolling(window=window, min_periods=30).mean()
        merged["ratio_std_252"] = merged["gold_silver_ratio"].rolling(window=window, min_periods=30).std()
        merged["ratio_zscore"] = (merged["gold_silver_ratio"] - merged["ratio_sma_252"]) / merged["ratio_std_252"].replace(0, np.nan)

        # Rolling Percentile
        def calc_percentile(s):
            if len(s) < 10:
                return 50.0
            return (s.iloc[-1] > s).mean() * 100.0

        merged["ratio_percentile_252"] = (
            merged["gold_silver_ratio"].rolling(window=window, min_periods=30).apply(calc_percentile, raw=False)
        )

        return merged

    @staticmethod
    def analyze_latest_ratio(ratio_df: pd.DataFrame) -> Dict[str, Any]:
        if ratio_df.empty:
            return {}

        latest = ratio_df.iloc[-1]
        curr_ratio = float(latest["gold_silver_ratio"])
        zscore = float(latest.get("ratio_zscore", 0.0))
        percentile = float(latest.get("ratio_percentile_252", 50.0))
        mean_252 = float(latest.get("ratio_sma_252", curr_ratio))

        # Regime classification
        if zscore > 2.0:
            regime = "Extremely High (Gold Outperforming / Silver Undervalued)"
            relative_value_bias = "Bullish Silver / Bearish Ratio"
        elif zscore < -2.0:
            regime = "Extremely Low (Silver Outperforming / Gold Undervalued)"
            relative_value_bias = "Bullish Gold / Bullish Ratio"
        elif zscore > 1.0:
            regime = "Elevated Ratio"
            relative_value_bias = "Slight Silver Bias"
        elif zscore < -1.0:
            regime = "Depressed Ratio"
            relative_value_bias = "Slight Gold Bias"
        else:
            regime = "Neutral / Historical Mean"
            relative_value_bias = "Neutral"

        return {
            "date": str(latest["date"]),
            "current_ratio": curr_ratio,
            "mean_252d": mean_252,
            "zscore": zscore,
            "percentile_252d": percentile,
            "regime": regime,
            "relative_value_bias": relative_value_bias,
        }
