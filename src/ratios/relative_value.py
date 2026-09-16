import numpy as np
import pandas as pd
from typing import Dict, Any


class RelativeValueAnalytics:
    """Calculates Gold/Silver ratio, relative returns, Z-scores, and mean-reversion metrics."""

    @staticmethod
    def calculate_gold_silver_ratio(gold_df: pd.DataFrame, silver_df: pd.DataFrame) -> pd.DataFrame:
        """Merges Gold and Silver price data and computes ratio metrics."""
        if gold_df.empty or silver_df.empty:
            return pd.DataFrame()

        g = gold_df[["date", "close"]].rename(columns={"close": "gold_price"})
        s = silver_df[["date", "close"]].rename(columns={"close": "silver_price"})

        merged = pd.merge(g, s, on="date", how="inner").sort_values("date").reset_index(drop=True)
        merged["gold_silver_ratio"] = merged["gold_price"] / merged["silver_price"]

        # Statistics over rolling windows
        merged["gs_ratio_sma_50"] = merged["gold_silver_ratio"].rolling(50, min_periods=5).mean()
        merged["gs_ratio_sma_200"] = merged["gold_silver_ratio"].rolling(200, min_periods=10).mean()

        std_200 = merged["gold_silver_ratio"].rolling(200, min_periods=10).std()
        merged["gs_ratio_zscore"] = (merged["gold_silver_ratio"] - merged["gs_ratio_sma_200"]) / (std_200 + 1e-8)

        # Historical Percentile (252-day expanding/rolling)
        merged["gs_ratio_percentile_252"] = (
            merged["gold_silver_ratio"]
            .rolling(252, min_periods=20)
            .apply(lambda x: (x.iloc[-1] > x).mean() * 100.0 if len(x) > 0 else 50.0, raw=False)
        )

        # Spread & Return Differences
        merged["gold_return_1d"] = merged["gold_price"].pct_change(1)
        merged["silver_return_1d"] = merged["silver_price"].pct_change(1)
        merged["spread_return_1d"] = merged["gold_return_1d"] - merged["silver_return_1d"]
        merged["spread_volatility_20d"] = merged["spread_return_1d"].rolling(20, min_periods=5).std() * np.sqrt(252)

        # Signal generation
        merged["ratio_extreme_high"] = merged["gold_silver_ratio"] > 85.0
        merged["ratio_extreme_low"] = merged["gold_silver_ratio"] < 65.0

        return merged
