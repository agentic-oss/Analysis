"""
Silver Analysis & Gold/Silver Relative Value Module.
Calculates Gold/Silver Ratio, historical percentiles, Z-scores, mean-reversion behavior,
spread dynamics, and historical forward-return statistics when extremes are reached.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np


class SilverRelativeValueAnalyzer:
    """Analyzes Silver vs Gold relative value, Gold/Silver ratio metrics, and forward spread stats."""

    def __init__(self, zscore_window: int = 252):
        self.zscore_window = zscore_window

    def calculate_ratio_metrics(
        self,
        gold_df: pd.DataFrame,
        silver_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculates Gold/Silver ratio, rolling averages, Z-score, and return spreads.
        Returns combined DataFrame with relative value indicators.
        """
        if gold_df.empty or silver_df.empty:
            return pd.DataFrame()

        g = gold_df[["date", "close"]].rename(columns={"close": "gold_close"})
        s = silver_df[["date", "close"]].rename(columns={"close": "silver_close"})
        merged = pd.merge(g, s, on="date", how="inner").sort_values("date").reset_index(drop=True)

        merged["gold_silver_ratio"] = merged["gold_close"] / merged["silver_close"].replace(0, np.nan)
        ratio = merged["gold_silver_ratio"]

        # Rolling statistics
        merged["ratio_sma_20"] = ratio.rolling(window=20, min_periods=1).mean()
        merged["ratio_sma_50"] = ratio.rolling(window=50, min_periods=1).mean()
        merged["ratio_sma_200"] = ratio.rolling(window=200, min_periods=1).mean()

        mean_w = ratio.rolling(window=self.zscore_window, min_periods=30).mean()
        std_w = ratio.rolling(window=self.zscore_window, min_periods=30).std().replace(0, np.nan)
        merged["gold_silver_ratio_zscore"] = (ratio - mean_w) / std_w

        # Daily returns and spread (Gold return - Silver return)
        merged["gold_ret_1d"] = merged["gold_close"].pct_change() * 100.0
        merged["silver_ret_1d"] = merged["silver_close"].pct_change() * 100.0
        merged["gold_silver_spread_1d"] = merged["gold_ret_1d"] - merged["silver_ret_1d"]

        # Forward return targets (1D, 3D, 5D, 10D, 20D, 60D) without look-ahead bias in features
        for h in [1, 3, 5, 10, 20, 60]:
            merged[f"future_gold_ret_{h}d"] = (merged["gold_close"].shift(-h) / merged["gold_close"] - 1.0) * 100.0
            merged[f"future_silver_ret_{h}d"] = (merged["silver_close"].shift(-h) / merged["silver_close"] - 1.0) * 100.0
            merged[f"future_ratio_{h}d"] = merged["gold_silver_ratio"].shift(-h)

        return merged

    def analyze_ratio_extremes(self, ratio_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates current ratio state, historical percentiles, and forward return expectations."""
        if ratio_df.empty or "gold_silver_ratio" not in ratio_df.columns:
            return {}

        curr = ratio_df.iloc[-1]
        ratio_series = ratio_df["gold_silver_ratio"].dropna()

        curr_ratio = float(curr["gold_silver_ratio"])
        pct_rank = float((ratio_series < curr_ratio).mean() * 100.0)
        zscore = float(curr["gold_silver_ratio_zscore"]) if pd.notna(curr["gold_silver_ratio_zscore"]) else 0.0

        # Mean-reversion signal detection
        signal = "NEUTRAL"
        if zscore >= 2.0:
            signal = "SILVER_OUTPERFORM_EXPECTED" # Ratio is extremely high, expect silver to catch up
        elif zscore <= -2.0:
            signal = "GOLD_OUTPERFORM_EXPECTED" # Ratio is extremely low, expect gold to outperform

        # Historical forward statistics when z-score was similar (|z - z_curr| < 0.5)
        sim_mask = (ratio_df["gold_silver_ratio_zscore"] - zscore).abs() < 0.5
        sim_sample = ratio_df[sim_mask]

        fwd_stats = {}
        for h in [1, 3, 5, 10, 20, 60]:
            g_col = f"future_gold_ret_{h}d"
            s_col = f"future_silver_ret_{h}d"
            if g_col in sim_sample.columns and not sim_sample[g_col].dropna().empty:
                g_rets = sim_sample[g_col].dropna()
                s_rets = sim_sample[s_col].dropna()
                fwd_stats[f"{h}d"] = {
                    "sample_count": len(g_rets),
                    "gold_mean_ret": float(g_rets.mean()),
                    "silver_mean_ret": float(s_rets.mean()),
                    "gold_win_rate": float((g_rets > 0).mean() * 100.0),
                    "silver_win_rate": float((s_rets > 0).mean() * 100.0)
                }

        return {
            "current_ratio": curr_ratio,
            "historical_percentile": pct_rank,
            "zscore": zscore,
            "signal": signal,
            "mean_ratio_252d": float(ratio_series.tail(252).mean()),
            "forward_extreme_statistics": fwd_stats
        }
