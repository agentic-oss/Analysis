"""
Gold/Silver relative value analysis and ratio metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


class RelativeValueAnalysis:
    """Analyzes Gold/Silver ratio, spreads, Z-scores, and historical forward statistics."""

    def __init__(self, lookback_window: int = 252):
        self.lookback_window = lookback_window

    def analyze_ratio(self, gold_df: pd.DataFrame, silver_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Gold/Silver ratio metrics over time:
        - gold_silver_ratio = gold_close / silver_close
        - rolling mean, std, Z-score, percentile rank
        - spread = gold_return - silver_return
        - forward returns (1D, 3D, 5D, 10D, 20D, 60D)
        """
        if gold_df.empty or silver_df.empty:
            return pd.DataFrame()

        g = gold_df[["date", "close"]].rename(columns={"close": "gold_close"})
        s = silver_df[["date", "close"]].rename(columns={"close": "silver_close"})

        df = pd.merge(g, s, on="date", how="inner").sort_values("date").reset_index(drop=True)
        df["gold_silver_ratio"] = df["gold_close"] / df["silver_close"].replace(0, np.nan)

        # Rolling ratio stats
        w = self.lookback_window
        df["ratio_mean_252d"] = df["gold_silver_ratio"].rolling(w, min_periods=30).mean()
        df["ratio_std_252d"] = df["gold_silver_ratio"].rolling(w, min_periods=30).std()
        df["ratio_zscore_252d"] = (df["gold_silver_ratio"] - df["ratio_mean_252d"]) / df["ratio_std_252d"].replace(0, np.nan)

        # Returns & spreads
        df["gold_ret_1d"] = df["gold_close"].pct_change()
        df["silver_ret_1d"] = df["silver_close"].pct_change()
        df["spread_1d"] = df["gold_ret_1d"] - df["silver_ret_1d"]
        df["spread_vol_20d"] = df["spread_1d"].rolling(20, min_periods=5).std() * np.sqrt(252)

        # Forward returns for gold, silver, and ratio for empirical forward stats calculation
        for horizon in [1, 3, 5, 10, 20, 60]:
            df[f"future_gold_ret_{horizon}d"] = df["gold_close"].pct_change(horizon).shift(-horizon)
            df[f"future_silver_ret_{horizon}d"] = df["silver_close"].pct_change(horizon).shift(-horizon)
            df[f"future_ratio_ret_{horizon}d"] = df["gold_silver_ratio"].pct_change(horizon).shift(-horizon)

        return df

    @staticmethod
    def calculate_forward_return_stats(subset_df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Calculates conditional forward return statistics on a filtered historical subset."""
        if subset_df.empty or target_col not in subset_df.columns:
            return {}

        valid_returns = subset_df[target_col].dropna()
        if len(valid_returns) == 0:
            return {}

        pos_count = (valid_returns > 0).sum()
        neg_count = (valid_returns < 0).sum()
        total_count = len(valid_returns)

        return {
            "sample_count": total_count,
            "mean_return_pct": float(valid_returns.mean() * 100),
            "median_return_pct": float(valid_returns.median() * 100),
            "std_dev_pct": float(valid_returns.std() * 100),
            "win_rate_pct": float((pos_count / total_count) * 100),
            "loss_rate_pct": float((neg_count / total_count) * 100),
            "max_gain_pct": float(valid_returns.max() * 100),
            "max_loss_pct": float(valid_returns.min() * 100),
            "percentile_25_pct": float(valid_returns.quantile(0.25) * 100),
            "percentile_75_pct": float(valid_returns.quantile(0.75) * 100),
        }
