"""
Gold/Silver Ratio & Relative-Value Analysis Module.
Calculates Gold/Silver ratio, Z-scores, historical percentiles, rolling spread, spread volatility,
mean reversion behavior, and forward relative return distributions.
"""
import logging
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RelativeValueAnalyzer:
    """
    Analyzes relative value metrics between Gold and Silver.
    Computes:
      - Gold/Silver ratio (gold_price / silver_price)
      - Return spread (gold_return - silver_return)
      - Historical percentile distribution
      - Z-score (20-day, 60-day, 252-day)
      - Spread volatility
      - Mean reversion signals
      - Forward relative return statistics (1D, 3D, 5D, 10D, 20D, 60D)
    """

    @staticmethod
    def compute_relative_value(gold_df: pd.DataFrame, silver_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges Gold and Silver daily data on date and computes relative metrics.
        """
        if gold_df.empty or silver_df.empty:
            return pd.DataFrame()

        g = gold_df[["date", "close"]].rename(columns={"close": "gold_price"}).sort_values("date")
        s = silver_df[["date", "close"]].rename(columns={"close": "silver_price"}).sort_values("date")

        merged = pd.merge(g, s, on="date", how="inner").dropna()
        if merged.empty:
            return pd.DataFrame()

        merged["gold_return_1d"] = merged["gold_price"].pct_change(1)
        merged["silver_return_1d"] = merged["silver_price"].pct_change(1)
        merged["return_spread_1d"] = merged["gold_return_1d"] - merged["silver_return_1d"]

        # Gold / Silver Ratio
        merged["gold_silver_ratio"] = merged["gold_price"] / merged["silver_price"]

        # Rolling statistics (60-day & 252-day window for Z-score)
        for win in [20, 60, 252]:
            mean = merged["gold_silver_ratio"].rolling(window=win, min_periods=5).mean()
            std = merged["gold_silver_ratio"].rolling(window=win, min_periods=5).std().replace(0, np.nan)
            merged[f"ratio_zscore_{win}d"] = (merged["gold_silver_ratio"] - mean) / std
            merged[f"spread_volatility_{win}d"] = merged["return_spread_1d"].rolling(window=win, min_periods=5).std() * np.sqrt(252)

        # Historical percentile calculation (expanding or rolling window)
        merged["ratio_percentile_historical"] = merged["gold_silver_ratio"].expanding(min_periods=20).rank(pct=True) * 100
        merged["ratio_mean_252d"] = merged["gold_silver_ratio"].rolling(window=252, min_periods=20).mean()

        # Extreme ratio flags
        merged["extreme_ratio_high"] = merged["gold_silver_ratio"] > 85.0
        merged["extreme_ratio_low"] = merged["gold_silver_ratio"] < 65.0

        # Mean reversion setup signal
        merged["mean_reversion_signal"] = np.where(
            merged["ratio_zscore_60d"] > 2.0, "favor_silver",
            np.where(merged["ratio_zscore_60d"] < -2.0, "favor_gold", "neutral")
        )

        return merged

    @staticmethod
    def calculate_forward_spread_outcomes(
        rv_df: pd.DataFrame,
        horizons: List[int] = [1, 3, 5, 10, 20, 60]
    ) -> Dict[str, Any]:
        """
        Calculates future return spread outcomes for Gold vs Silver when the ratio is at extreme levels.
        """
        if rv_df.empty or "gold_silver_ratio" not in rv_df.columns:
            return {}

        df = rv_df.copy().sort_values("date").reset_index(drop=True)
        results = {}

        for h in horizons:
            # Shift backwards for future returns (look-ahead target creation for analysis only)
            df[f"future_gold_return_{h}d"] = df["gold_price"].pct_change(h).shift(-h)
            df[f"future_silver_return_{h}d"] = df["silver_price"].pct_change(h).shift(-h)
            df[f"future_spread_return_{h}d"] = df[f"future_gold_return_{h}d"] - df[f"future_silver_return_{h}d"]

        latest_ratio = df["gold_silver_ratio"].iloc[-1]
        latest_zscore = df["ratio_zscore_60d"].iloc[-1] if "ratio_zscore_60d" in df.columns else 0.0

        for h in horizons:
            col = f"future_spread_return_{h}d"
            valid_series = df[col].dropna()

            if len(valid_series) == 0:
                continue

            results[f"horizon_{h}d"] = {
                "mean_spread_return": float(valid_series.mean()),
                "median_spread_return": float(valid_series.median()),
                "pos_spread_prob": float((valid_series > 0).mean()),
                "neg_spread_prob": float((valid_series < 0).mean()),
                "p10": float(valid_series.quantile(0.10)),
                "p90": float(valid_series.quantile(0.90))
            }

        return {
            "latest_ratio": float(latest_ratio),
            "latest_zscore_60d": float(latest_zscore),
            "forward_horizons": results
        }
