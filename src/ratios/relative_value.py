import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

class RelativeValueAnalyzer:
    """
    Dedicated analyzer for Gold/Silver ratio and relative value relationships.
    Calculates:
    - Gold/Silver ratio (gold_price / silver_price)
    - Historical percentile, Z-score, rolling spread
    - Mean-reversion behavior and forward return statistics following extreme ratio situations.
    """
    def __init__(self, zscore_window: int = 252, extreme_zscore_threshold: float = 2.0):
        self.zscore_window = zscore_window
        self.extreme_threshold = extreme_zscore_threshold

    def calculate_ratio_series(
        self,
        gold_df: pd.DataFrame,
        silver_df: pd.DataFrame,
        gold_col: str = "close",
        silver_col: str = "close"
    ) -> pd.DataFrame:
        """
        Calculates daily Gold/Silver ratio and derived metrics.
        """
        if gold_df.empty or silver_df.empty:
            return pd.DataFrame()

        g = gold_df[["timestamp", gold_col]].rename(columns={gold_col: "gold_price"})
        s = silver_df[["timestamp", silver_col]].rename(columns={silver_col: "silver_price"})

        merged = pd.merge(g, s, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
        merged["gold_silver_ratio"] = merged["gold_price"] / merged["silver_price"]

        ratio = merged["gold_silver_ratio"]
        merged["ratio_sma_50"] = ratio.rolling(window=50, min_periods=1).mean()
        merged["ratio_sma_200"] = ratio.rolling(window=200, min_periods=1).mean()

        mean = ratio.rolling(window=self.zscore_window, min_periods=20).mean()
        std = ratio.rolling(window=self.zscore_window, min_periods=20).std()
        merged["ratio_zscore"] = (ratio - mean) / (std + 1e-10)

        # Gold vs Silver returns spread
        merged["gold_ret_1d"] = merged["gold_price"].pct_change() * 100.0
        merged["silver_ret_1d"] = merged["silver_price"].pct_change() * 100.0
        merged["ret_spread_1d"] = merged["gold_ret_1d"] - merged["silver_ret_1d"]

        # Forward return spread for relative value evaluation
        horizons = [1, 3, 5, 10, 20, 60]
        for h in horizons:
            g_fwd = (merged["gold_price"].shift(-h) - merged["gold_price"]) / merged["gold_price"] * 100.0
            s_fwd = (merged["silver_price"].shift(-h) - merged["silver_price"]) / merged["silver_price"] * 100.0
            merged[f"fwd_gold_ret_{h}d"] = g_fwd
            merged[f"fwd_silver_ret_{h}d"] = s_fwd
            merged[f"fwd_spread_ret_{h}d"] = g_fwd - s_fwd

        return merged

    def analyze_ratio_extremes(
        self,
        ratio_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Identifies situations where Gold/Silver ratio reached extreme levels (|Z| > threshold)
        and evaluates historical subsequent forward outcomes.
        """
        if ratio_df.empty or "ratio_zscore" not in ratio_df.columns:
            return {}

        latest = ratio_df.iloc[-1]
        curr_ratio = float(latest["gold_silver_ratio"]) if not pd.isna(latest["gold_silver_ratio"]) else None
        curr_z = float(latest["ratio_zscore"]) if not pd.isna(latest["ratio_zscore"]) else None

        # Percentile rank in historical series
        percentile = (ratio_df["gold_silver_ratio"] < curr_ratio).mean() * 100.0 if curr_ratio else None

        # Analyze high ratio extremes (Gold expensive vs Silver)
        high_extremes = ratio_df[ratio_df["ratio_zscore"] > self.extreme_threshold]
        # Analyze low ratio extremes (Silver expensive vs Gold)
        low_extremes = ratio_df[ratio_df["ratio_zscore"] < -self.extreme_threshold]

        forward_stats = {}
        for h in [1, 3, 5, 10, 20, 60]:
            col = f"fwd_spread_ret_{h}d"
            if col in ratio_df.columns:
                high_ret = high_extremes[col].dropna()
                low_ret = low_extremes[col].dropna()
                forward_stats[f"{h}d"] = {
                    "high_ratio_mean_fwd_spread": round(float(high_ret.mean()), 2) if len(high_ret) > 0 else 0.0,
                    "low_ratio_mean_fwd_spread": round(float(low_ret.mean()), 2) if len(low_ret) > 0 else 0.0,
                }

        return {
            "current_ratio": round(curr_ratio, 2) if curr_ratio else None,
            "current_zscore": round(curr_z, 2) if curr_z else None,
            "historical_percentile": round(percentile, 1) if percentile else None,
            "long_term_mean": round(float(ratio_df["gold_silver_ratio"].mean()), 2) if not ratio_df.empty else None,
            "forward_extreme_spread_stats": forward_stats
        }
