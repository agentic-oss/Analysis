"""
Gold & Silver Specific Analysis and Gold/Silver Ratio Engine.
Calculates Gold/Silver ratio, historical percentiles, Z-scores, mean-reversion,
and multi-factor rolling correlations across macro drivers (DXY, Yields, Oil, Equities, VIX).
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd


class RatioAnalysisEngine:
    """Gold/Silver ratio and relative value analysis."""

    def calculate_ratio_metrics(
        self,
        gold_df: pd.DataFrame,
        silver_df: pd.DataFrame,
        lookback_window: int = 252,
    ) -> pd.DataFrame:
        """Calculates Gold/Silver ratio and relative statistical metrics."""
        merged = pd.merge(
            gold_df[["date", "close"]].rename(columns={"close": "gold"}),
            silver_df[["date", "close"]].rename(columns={"close": "silver"}),
            on="date",
            how="inner",
        ).sort_values("date")

        merged["gold_silver_ratio"] = merged["gold"] / merged["silver"]

        # Ratio rolling statistics
        merged["ratio_sma_50"] = merged["gold_silver_ratio"].rolling(50).mean()
        merged["ratio_sma_200"] = merged["gold_silver_ratio"].rolling(200).mean()

        rolling_mean = merged["gold_silver_ratio"].rolling(lookback_window).mean()
        rolling_std = merged["gold_silver_ratio"].rolling(lookback_window).std()
        merged["ratio_zscore"] = (merged["gold_silver_ratio"] - rolling_mean) / (rolling_std + 1e-9)

        # Ratio percentile over lookback
        def calc_percentile(s):
            if len(s) < 2:
                return 50.0
            return (s.iloc[-1] > s).mean() * 100.0

        merged["ratio_percentile_252"] = (
            merged["gold_silver_ratio"].rolling(lookback_window).apply(calc_percentile, raw=False)
        )

        # Relative return spread (Gold return - Silver return)
        merged["gold_return_1d"] = merged["gold"].pct_change(1)
        merged["silver_return_1d"] = merged["silver"].pct_change(1)
        merged["return_spread_1d"] = merged["gold_return_1d"] - merged["silver_return_1d"]
        merged["spread_volatility_20d"] = merged["return_spread_1d"].rolling(20).std()

        return merged


class CrossMarketCorrelationEngine:
    """Calculates multi-factor rolling correlations between gold/silver and macro drivers."""

    def compute_rolling_correlations(
        self,
        asset_series: pd.Series,
        macro_df: pd.DataFrame,
        windows: List[int] = [20, 60, 120, 252],
    ) -> pd.DataFrame:
        """
        Computes rolling correlation between asset_series and macro_df columns over multiple windows.
        """
        df = pd.DataFrame({"asset": asset_series}).join(macro_df)
        results = pd.DataFrame(index=df.index)

        asset_ret = df["asset"].pct_change()

        for col in macro_df.columns:
            col_ret = df[col].pct_change() if macro_df[col].max() > 5.0 else df[col].diff()
            for w in windows:
                corr_col = f"corr_{col}_{w}d"
                results[corr_col] = asset_ret.rolling(w).corr(col_ret)

        return results
