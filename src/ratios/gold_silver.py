import logging
import pandas as pd
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)

class GoldSilverRatioAnalyzer:
    """Analyzes Gold/Silver ratio, percentiles, z-scores, rolling spreads, and forward return behavior."""

    @staticmethod
    def analyze_ratio(gold_df: pd.DataFrame, silver_df: pd.DataFrame) -> pd.DataFrame:
        if gold_df.empty or silver_df.empty:
            return pd.DataFrame()

        merged = pd.merge(
            gold_df[["date", "close"]].rename(columns={"close": "gold_price"}),
            silver_df[["date", "close"]].rename(columns={"close": "silver_price"}),
            on="date",
            how="inner"
        )

        merged["gold_silver_ratio"] = merged["gold_price"] / merged["silver_price"].replace(0, np.nan)
        ratio = merged["gold_silver_ratio"]

        # Rolling statistics
        merged["ratio_sma_50"] = ratio.rolling(window=50, min_periods=1).mean()
        merged["ratio_sma_200"] = ratio.rolling(window=200, min_periods=1).mean()

        mean_252 = ratio.rolling(window=252, min_periods=1).mean()
        std_252 = ratio.rolling(window=252, min_periods=1).std()
        merged["ratio_zscore_252d"] = (ratio - mean_252) / std_252.replace(0, np.nan)

        # Percentile rank in 252d window
        merged["ratio_percentile_252d"] = ratio.rolling(window=252, min_periods=1).apply(
            lambda s: (s < s.iloc[-1]).mean() * 100.0 if len(s) > 0 else 50.0, raw=False
        )

        # Return spread: Gold return - Silver return
        gold_ret = merged["gold_price"].pct_change(1)
        silver_ret = merged["silver_price"].pct_change(1)
        merged["gold_return_1d"] = gold_ret
        merged["silver_return_1d"] = silver_ret
        merged["gold_silver_return_spread_1d"] = gold_ret - silver_ret
        merged["spread_volatility_20d"] = merged["gold_silver_return_spread_1d"].rolling(window=20, min_periods=1).std()

        # Forward return statistics for extreme ratio situations
        for horizon in [1, 3, 5, 10, 20, 60]:
            merged[f"future_gold_return_{horizon}d"] = merged["gold_price"].pct_change(horizon).shift(-horizon)
            merged[f"future_silver_return_{horizon}d"] = merged["silver_price"].pct_change(horizon).shift(-horizon)
            merged[f"future_ratio_return_{horizon}d"] = merged["gold_silver_ratio"].pct_change(horizon).shift(-horizon)

        return merged

    @staticmethod
    def get_latest_summary(ratio_df: pd.DataFrame) -> Dict[str, Any]:
        if ratio_df.empty:
            return {}
        last = ratio_df.iloc[-1]
        return {
            "date": last["date"],
            "current_ratio": float(last["gold_silver_ratio"]),
            "ratio_sma_50": float(last["ratio_sma_50"]),
            "ratio_sma_200": float(last["ratio_sma_200"]),
            "ratio_zscore": float(last.get("ratio_zscore_252d", 0.0)),
            "ratio_percentile": float(last.get("ratio_percentile_252d", 50.0)),
            "return_spread_1d": float(last.get("gold_silver_return_spread_1d", 0.0))
        }
