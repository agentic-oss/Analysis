import pandas as pd
import numpy as np
from typing import Dict, Any, List

class RelativeValueAnalyzer:
    """Analyzes Gold/Silver ratio, z-scores, percentiles, and mean-reversion behavior."""

    @staticmethod
    def analyze_gold_silver_ratio(gold_series: pd.Series, silver_series: pd.Series) -> pd.DataFrame:
        """
        Calculates:
        - Ratio: gold_price / silver_price
        - Return difference: gold_return - silver_return
        - Rolling Z-Score (60d, 252d)
        - Historical percentile rank (252d)
        - Mean reversion indicators
        """
        df = pd.DataFrame()
        df["gold_close"] = gold_series.astype(float)
        df["silver_close"] = silver_series.astype(float)

        # Ratio
        df["gold_silver_ratio"] = df["gold_close"] / df["silver_close"].replace(0, np.nan)

        # Returns
        gold_ret = df["gold_close"].pct_change()
        silver_ret = df["silver_close"].pct_change()
        df["gold_silver_spread_ret"] = gold_ret - silver_ret

        # Rolling statistics (60d and 252d)
        for w in [60, 252]:
            mean = df["gold_silver_ratio"].rolling(window=w, min_periods=10).mean()
            std = df["gold_silver_ratio"].rolling(window=w, min_periods=10).std()
            df[f"ratio_mean_{w}d"] = mean
            df[f"ratio_zscore_{w}d"] = (df["gold_silver_ratio"] - mean) / std.replace(0, np.nan)

        # 252d Percentile Rank
        def _percentile_rank(window):
            if len(window) < 10 or pd.isna(window.iloc[-1]):
                return np.nan
            val = window.iloc[-1]
            return (window < val).mean() * 100.0

        df["ratio_percentile_252d"] = df["gold_silver_ratio"].rolling(window=252, min_periods=10).apply(_percentile_rank, raw=False)

        # Mean-reversion signals
        df["ratio_extreme_high"] = df["ratio_zscore_252d"] > 2.0
        df["ratio_extreme_low"] = df["ratio_zscore_252d"] < -2.0

        return df
