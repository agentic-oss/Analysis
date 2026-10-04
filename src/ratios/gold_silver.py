import pandas as pd
import numpy as np
from typing import Dict, Any

def analyze_gold_silver_ratio(
    gold_df: pd.DataFrame,
    silver_df: pd.DataFrame,
    lookback_window: int = 252
) -> pd.DataFrame:
    """Calculate Gold/Silver ratio, percentiles, Z-score, rolling spread, and forward relative return statistics."""
    if gold_df.empty or silver_df.empty:
        return pd.DataFrame()

    g = gold_df[["date", "close"]].rename(columns={"close": "gold_price"})
    s = silver_df[["date", "close"]].rename(columns={"close": "silver_price"})

    merged = pd.merge(g, s, on="date", how="inner").sort_values("date").reset_index(drop=True)
    merged["gold_silver_ratio"] = merged["gold_price"] / merged["silver_price"]

    # Rolling statistics
    ratio = merged["gold_silver_ratio"]
    rolling_mean = ratio.rolling(window=lookback_window, min_periods=20).mean()
    rolling_std = ratio.rolling(window=lookback_window, min_periods=20).std()

    merged["ratio_sma_252"] = rolling_mean
    merged["ratio_zscore"] = (ratio - rolling_mean) / rolling_std.replace(0, np.nan)

    # Historical percentile rank (252-day rolling percentile)
    def percentile_rank(window):
        if len(window) < 5:
            return 50.0
        val = window.iloc[-1]
        return (window < val).mean() * 100.0

    merged["ratio_percentile_252"] = ratio.rolling(window=lookback_window, min_periods=20).apply(percentile_rank, raw=False)

    # Return Spread: Gold Return minus Silver Return
    gold_ret = merged["gold_price"].pct_change()
    silver_ret = merged["silver_price"].pct_change()
    merged["return_spread_1d"] = gold_ret - silver_ret
    merged["spread_volatility_20d"] = merged["return_spread_1d"].rolling(20, min_periods=5).std()

    # Extreme conditions for mean-reversion analysis
    merged["is_ratio_extreme_high"] = merged["ratio_zscore"] > 2.0
    merged["is_ratio_extreme_low"] = merged["ratio_zscore"] < -2.0

    return merged
